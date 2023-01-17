%pylab inline

import warnings
warnings.filterwarnings('ignore')

import pandas as pd

import seaborn as sns
sns.set()


from statsmodels.tsa.stattools import adfuller


from statsmodels.tsa.stattools import acf, pacf
from copy import deepcopy


from datetime import datetime

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)
# Functions :

"""Plots a simple serie in PLOTLY."""
def jsplot(dates , values , mode = 'lines+markers'):

    data = [go.Scatter(
              x=dates,
              y=values,
              mode = mode)]

    iplot(data)
    
    
"""Plot multiple series in PLOTLY:"""
def jsplot_multiple(dates , values , mode = 'lines+markers'):

    data = []
    for col in values.columns:
        splot = go.Scatter(
                        x=dates,
                        y=values[col],
                        mode = mode,
                        name = str(col) )
        data.append(splot)

    iplot(data)
    
    
"""Function that test the stationarity of a Time series by
computing and plotting rolling statistics, and then by performing
An augmented Dickey Fuller test.""" 

def test_stationarity(timeseries , window = 50):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window).mean()
    rolstd = timeseries.rolling(window).std()

    #Plot rolling statistics:
    fig = plt.figure(figsize=(12, 8))
    orig = plt.plot(timeseries,label='Original')
    mean = plt.plot(rolmean, color='red' , label='Rolling Mean')
    std = plt.plot(rolstd, label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show()
    
    #Perform Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:')
    try:
        dftest = adfuller(timeseries.dropna(), autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput) 
    except:
        print('test failed')
        

        
"""Performs Acp - Pacp Analysis on a time serie."""
def acp_pacp(timeseries , nlags = 30):
    lag_acf = acf(timeseries, nlags=nlags)
    lag_pacf = pacf(timeseries, nlags=nlags, method='ols')
    
    print('lag_acf')
    fig = plt.figure(figsize=(7 , 6))

    sns.barplot( np.arange(len(lag_acf)) , lag_acf , palette = 'GnBu_d')
    
    
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')

    plt.show()
    print('lag_pacf')
    fig = plt.figure(figsize=(7, 6))

    sns.barplot( np.arange(len(lag_pacf)) , lag_pacf , palette = 'GnBu_d')

    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(timeseries)),linestyle='--',color='gray')

    plt.show()
caracs = pd.read_csv('../input/caracteristics.csv' , encoding = 'latin-1') # Caracteristics of the accidents.
places = pd.read_csv('../input/places.csv' ) # Places features.
users = pd.read_csv('../input/users.csv' ) # Users involved in the accdient features.
vehicles = pd.read_csv('../input/vehicles.csv') # Vehicles features.
holidays = pd.read_csv('../input/holidays.csv') # Vehicles features.
dtsers = caracs.loc[(caracs.dep.isin([750])) , ['Num_Acc' , 'jour' , 'mois' , 'an']]

dtsers['day'] = pd.to_datetime((2000+dtsers.an)*10000+dtsers.mois*100+dtsers.jour,format='%Y%m%d')
dtsers.drop(['jour' , 'mois' , 'an'] , axis = 1 ,inplace = True)


dtsers = dtsers.groupby('day' , as_index = False).count()

# Dummy Variable Holiday
dtsers['isholiday'] = 0
dtsers.loc[dtsers.day.isin(holidays.ds) , 'isholiday'] = 1

# Week day and month
dtsers['weekday'] = dtsers.day.dt.weekday
dtsers['month'] = dtsers.day.dt.month
# Dummification
dtsers = pd.get_dummies(dtsers , columns = ['weekday' , 'month'])

print(' the 3 last years of the time series:')
jsplot(dtsers.day[3500:] , dtsers.Num_Acc[3500:] )
# Some statistics :
test_stationarity(dtsers.Num_Acc , window = 28)
acp_pacp(dtsers.Num_Acc)
tempas = caracs.loc[caracs.dep == 750 , ['Num_Acc' , 'hrmn']]
tempas['hour'] = tempas['hrmn'].apply(lambda x:str(x).zfill(4)[:2])


grave_accs = users[users.grav.isin([2,3]) ].Num_Acc

tempas['gravity'] = 0
tempas.loc[tempas.Num_Acc.isin(grave_accs),'gravity'] = 1


occs = tempas.drop('hrmn' , axis = 1).groupby('hour' , as_index = False).agg({'Num_Acc' : 'count' , 'gravity' : 'sum'})



trace1 = go.Area(
    r=list(occs.Num_Acc),
    t=list(occs.hour),
    name='Total Number of accidents',
    marker=dict(
        color='rgb(106,81,163)'
    )
)

trace2 = go.Area(
    r=list(occs.gravity),
    t=list(occs.hour),
    name='Grave accidents',
    marker=dict(
        color='rgb(158,154,200)'
    )
)

data = [trace1 , trace2]

layout = go.Layout(
    title='Repartition of accidents per Hour',
    autosize = False,
    width = 1000,
    height = 500,
    orientation=-90
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
tempas = caracs.loc[caracs.dep == 750 ,['Num_Acc']]
tempas['date'] = pd.to_datetime((2000+caracs.an)*10000+caracs.mois*100+caracs.jour,format='%Y%m%d')
tempas['weekday'] = tempas['date'].dt.weekday.apply(lambda x:str(x).zfill(2))

tempas['gravity'] = 0
tempas.loc[tempas.Num_Acc.isin(grave_accs),'gravity'] = 1


occs = tempas.drop('date' , axis = 1).groupby('weekday' , as_index = False).agg({'Num_Acc' : 'count' , 'gravity' : 'sum'})



trace1 = go.Area(
    r=list(occs.Num_Acc),
    t=list(occs.weekday),
    name='Total Number of accidents',
    marker=dict(
        color='rgb(106,81,163)'
    )
)

trace2 = go.Area(
    r=list(occs.gravity),
    t=list(occs.weekday),
    name='Grave accidents',
    marker=dict(
        color='rgb(158,154,200)'
    )
)

data = [trace1 , trace2]

layout = go.Layout(
    title='Repartition of accidents per weekday',
    autosize = False,
    width = 1000,
    height = 500,
    orientation=-90
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
tempas = caracs.loc[caracs.dep == 750 ,['Num_Acc' , 'mois']]
tempas['mois'] = tempas['mois'].apply(lambda x:str(x).zfill(2))

tempas['gravity'] = 0
tempas.loc[tempas.Num_Acc.isin(grave_accs),'gravity'] = 1


occs = tempas.groupby('mois' , as_index = False).agg({'Num_Acc' : 'count' , 'gravity' : 'sum'})



trace1 = go.Area(
    r=list(occs.Num_Acc),
    t=list(occs.mois),
    name='Total Number of accidents',
    marker=dict(
        color='rgb(106,81,163)'
    )
)

trace2 = go.Area(
    r=list(occs.gravity),
    t=list(occs.mois),
    name='Grave accidents',
    marker=dict(
        color='rgb(158,154,200)'
    )
)

data = [trace1 , trace2]

layout = go.Layout(
    title='Repartition of accidents per Hour',
    autosize = False,
    width = 1000,
    height = 500,
    orientation=-90
)
fig = go.Figure(data=data, layout=layout)
iplot(fig)
def evaluate(y_true , y_pred , dates):
    
    try:
        true_value , prediction = y_true.sum(axis = 1), y_pred.sum(axis=1).round()
    except:
        true_value , prediction = y_true, y_pred.round()
    
    print('Mean Absolute Error   :' , round(abs(true_value - prediction).mean() , 2)) 
    print('Root Mean Square Error:' , round(sqrt(((true_value - prediction)**2).mean()) , 2) )
    print('Mean Percentage Error :' , round((abs(true_value - prediction)/true_value).mean() , 2)  )
    
    error = pd.Series(true_value - prediction)
    
    #density plot :
    print('Error Density :')
    error.plot.density()
    plt.show()
    
    # mean of error and correlation :
    print('Mean Error                       :' , round(mean(error) , 2 ))
    print('True Value And error Correlation :' , round(np.corrcoef(error , true_value)[0 , 1] , 2))
    
    # plot :
    
    to_plot = pd.DataFrame({'target' : y_true.reshape(-1) , 'prediction' : y_pred.reshape(-1)})
    
    jsplot_multiple(dates , to_plot)
# Naive Model :

new , old = (dtsers.loc[dtsers.day.dt.year == 2016 , ['day' , 'Num_Acc']].reset_index(drop = True) ,
             dtsers.loc[dtsers.day.dt.year == 2015 , ['day' , 'Num_Acc']].reset_index(drop = True)[:365])

old.columns = ['day' , 'old']

new['weekofyear'] , new['dayofweek'] = new.day.dt.weekofyear , new.day.dt.dayofweek
old['weekofyear'] , old['dayofweek'] = old.day.dt.weekofyear , old.day.dt.dayofweek

merged = new.merge(old , on = ['weekofyear' , 'dayofweek'])


evaluate(merged.Num_Acc.values , merged.old.values , dtsers.day[-365:])
from fbprophet import Prophet
#Initialisation of the model.
model = Prophet(holidays = holidays , yearly_seasonality=True , weekly_seasonality=True, daily_seasonality=False)

#train & test set.
histo , new = dtsers[dtsers.day.dt.year < 2016].reset_index(drop = True) , dtsers[dtsers.day.dt.year 
                                                                                  == 2016].reset_index(drop = True)

# We rename the columns before fitting the model to Prophet.
ncols = histo.columns.values
ncols[0] , ncols[1] = 'ds' , 'y'

histo.columns , new.columns = ncols , ncols

# We fit the model.
model.fit(histo)


# Prediction
ypred = model.predict(new)['yhat'].round()

# Evaluation
evaluate(new.y.values , ypred.values , dtsers.day[-365:])
import keras
from keras.models import Sequential , load_model
from keras.layers import Dense , LSTM, Dropout , Conv1D , MaxPooling1D , Reshape , Activation
from keras.layers import Masking , TimeDistributed, Bidirectional
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.callbacks import History , ModelCheckpoint
def reshape_timeseries(series , target_ids, window_size , take_curr = True , scale = True):
    
    
    # Converting the dataset to a suitable format :
    X = series.values
    Y = series.iloc[ : , target_ids].values
    
    # Scaling the data  
    if scale:
        maxes = Y.max(axis = 0)
        Y = np.divide( Y , maxes)
        X = MinMaxScaler().fit_transform(X)
    
    # Conversion to time series with keras object
    ts = TimeseriesGenerator(X , Y , length = window_size , batch_size = X.shape[0])
    X , Y = ts[0]
    
    # Masking
    if take_curr:
        for timestep in X[: , window_size - 1]:
            timestep[target_ids] = [-2 for i in target_ids]
    else:
        X = X[: , :-1]
        
    if scale:
        return X , Y , maxes

    return X,Y
def model(X , Y , lr = 0.001,
          lstm_layers = [] , lstm_dropout = [],
          dense_layers = [] , dense_dropout = [] ,
          ntest_day = 365 , epochs = 10 , batch_size = 32):
        
        
    # training and testing set :
    length , timesteps , features = X.shape[0] , X.shape[1] , X.shape[2]
    target_shape = Y.shape[1]
    
    # Validation rate to pass to the Sequential Model :
    val_rate = ntest_day/length
    
    
    ############################################ Model :
    
    checkpoint = ModelCheckpoint('model' , save_best_only=True)
    
    model = Sequential()
    
    # Masking Layer.
    model.add(Masking(mask_value = -2 , input_shape=(X.shape[1],  X.shape[2])    ))
    
    
    # BI-LSTM Layers.
    for i in range(len(lstm_layers)):
        rsequs  = not (i == (len(lstm_layers) - 1))
        model.add(Bidirectional( LSTM(lstm_layers[i] , return_sequences = rsequs) ,input_shape=(X.shape[1], X.shape[2]) ) )
        model.add(Dropout(lstm_dropout[i]))


    # Dense Layers.
    for i in range(len(dense_layers)): 
        model.add(Dense(dense_layers[i]) )
        model.add(Dropout(dense_dropout[i]))
        model.add(Activation('relu'))
        
    
    model.add(Dense(target_shape))
    Nadam = keras.optimizers.Nadam(lr = lr , beta_1=0.9, beta_2=0.999, epsilon=1e-08)#, schedule_decay=0.0004)
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    print('Model Summary:')
    print(model.summary())
    
    # fitting the data
    print('\n\n Training :')
    model.fit(X, Y, epochs= epochs, batch_size=batch_size, validation_split = val_rate, callbacks = [checkpoint])
    
    
    # loading best_model
    model = load_model('model')
    
    return model
X , Y , maxes  = reshape_timeseries(dtsers.iloc[:, 1:] , [0], window_size = 28 , take_curr = True , scale = True)
ntest_day = 365

nmodel = model(X , Y , lr = 0.002, lstm_layers = [20 ] , lstm_dropout = [.3 ] ,
               dense_layers = [500] , dense_dropout = [.5] , batch_size = 64 , epochs = 20)

# Computing Validation scores : MAE - RMSE - MPE
y_predict = nmodel.predict(X[- ntest_day:]) * maxes
y_true = Y[- ntest_day:] * maxes

evaluate(y_true , y_predict , dtsers.day[-365:])
def long_term_prediction(model , X , nb_target):
#Function also adapted to multiple targets    
    predictions = []
    new_line = X[0].reshape(1 , *X.shape[1:])
    pred = model.predict(new_line)
    predictions.append(pred)
    
    for line in X[1:]:
        old_line = deepcopy(line)
        old_line[-2 , :nb_target] = pred
        pred= model.predict(old_line.reshape(1 , *X.shape[1:]))
        predictions.append(pred)
        
    return np.array(predictions).reshape(-1 , nb_target )
# Computing Validation scores : MAE - RMSE - MPE
y_predict = long_term_prediction(nmodel , X[- ntest_day:] , 1)* maxes
y_true = Y[- ntest_day:] * maxes
    
evaluate(y_true , y_predict , dtsers.day[-365:])
cdtsers = caracs.loc[(caracs.dep.isin([750])) , ['Num_Acc' , 'dep', 'com', 'jour' , 'mois' , 'an']]


cdtsers['day'] = pd.to_datetime((2000+cdtsers.an)*10000+cdtsers.mois*100+cdtsers.jour,format='%Y%m%d')
cdtsers.drop(['jour' , 'mois' , 'an'] , axis = 1 ,inplace = True)

def correct(x):
    if x>100:
        return x - 100
    return x

cdtsers.com = cdtsers.com.apply( correct )

cdtsers = cdtsers.groupby(['day' , 'dep' , 'com'] , as_index = False).count()

cdtsers = cdtsers.pivot_table(index = ['day' , 'dep'] , columns = 'com' , values = 'Num_Acc').reset_index()

cdtsers.fillna(0).head()

cdtsers['isholiday'] = 0
cdtsers.loc[cdtsers.day.isin(holidays.ds) , 'isholiday'] = 1



cdtsers['weekday'] = cdtsers.day.dt.weekday
cdtsers['month'] = cdtsers.day.dt.month
cdtsers = pd.get_dummies(cdtsers , columns = ['weekday' , 'month'])


cdtsers.drop([56 , 'dep'] , axis = 1 , inplace = True)
cdtsers.fillna(0 , inplace = True)
X , Y , maxes  = reshape_timeseries(cdtsers.iloc[: , 1:] , list(range(19)), window_size = 28 , take_curr = True , scale = True)

ntest_day = 365

nmodel = model(X , Y , lr = 0.005, lstm_layers = [64 , 64] , lstm_dropout = [.2 , .2] ,
               dense_layers = [64] , dense_dropout = [.2] , batch_size = 64 , epochs = 20)
# Evaluating:
y_predict = (nmodel.predict(X[- ntest_day:]) * maxes).sum(axis = 1)
y_true = (Y[- ntest_day:] * maxes).sum(axis = 1)

evaluate(y_true , y_predict , cdtsers.day[-365:])
# Evaluating on long term:
y_predict =( long_term_prediction(nmodel , X[- ntest_day:] , 19)* maxes).sum(axis = 1)
y_true = (Y[- ntest_day:] * maxes).sum(axis = 1)
    
evaluate(y_true , y_predict , cdtsers.day[-365:])
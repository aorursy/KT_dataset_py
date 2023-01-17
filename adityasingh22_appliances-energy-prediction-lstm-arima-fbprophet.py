import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import preprocessing, model_selection, metrics
#import the dataset in dataframe

data = pd.read_csv("../input/KAG_energydata_complete.csv")
data.head()
print('The number of rows in dataset is - ' , data.shape[0])

print('The number of columns in dataset is - ' , data.shape[1])
#Number of null values in all columns

data.isnull().sum().sort_values(ascending = True)
from sklearn.model_selection import train_test_split



# 75% of the data is usedfor the training of the models and the rest is used for testing

train, test = train_test_split(data,test_size=0.25,random_state=40)
# Divide the columns based on type for clear column management 



col_time=["date"]



col_temp = ["T1","T2","T3","T4","T5","T6","T7","T8","T9"]



col_hum = ["RH_1","RH_2","RH_3","RH_4","RH_5","RH_6","RH_7","RH_8","RH_9"]



col_weather = ["T_out", "Tdewpoint","RH_out","Press_mm_hg",

                "Windspeed","Visibility"] 

col_light = ["lights"]



col_randoms = ["rv1", "rv2"]



col_target = ["Appliances"]
# Seperate dependent and independent variables 

feature_vars = train[ col_time + col_temp + col_hum + col_weather + col_light + col_randoms ]

target_vars = train[col_target]
# Check the distribution of values in lights column

feature_vars.lights.value_counts()
# Due to lot of zero enteries this column is of not much use and will be ignored in rest of the model

_ = feature_vars.drop(['lights'], axis=1 , inplace= True) ;
feature_vars.head(2)
# plotly

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go



# To understand the timeseries variation of the applaince energy consumption

visData = go.Scatter( x= data.date  ,  mode = "lines", y = data.Appliances )

layout = go.Layout(title = 'Appliance energy consumption pattern' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))

fig = go.Figure(data=[visData],layout=layout)



iplot(fig)
# Adding column to mark weekdays (0) and weekends(1) for time series evaluation , 

# decided not to use it for model evaluation as it has least impact



data['WEEKDAY'] = ((pd.to_datetime(data['date']).dt.dayofweek)// 5 == 1).astype(float)

# There are 5472 weekend recordings 

data['WEEKDAY'].value_counts()
# Find rows with weekday 

temp_weekday =  data[data['WEEKDAY'] == 0]

# To understand the timeseries variation of the applaince energy consumption

visData = go.Scatter( x= temp_weekday.date  ,  mode = "lines", y = temp_weekday.Appliances )

layout = go.Layout(title = 'Appliance energy consumption pattern on weekdays' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))

fig = go.Figure(data=[visData],layout=layout)



iplot(fig)
# Find rows with weekend 



temp_weekend =  data[data['WEEKDAY'] == 1]



# To understand the timeseries variation of the applaince energy consumption

visData = go.Scatter( x= temp_weekend.date  ,  mode = "lines", y = temp_weekend.Appliances )

layout = go.Layout(title = 'Appliance energy consumption pattern on weekend' , xaxis=dict(title='Date'), yaxis=dict(title='(Wh)'))

fig = go.Figure(data=[visData],layout=layout)



iplot(fig)
# Histogram of all the features to understand the distribution

feature_vars.hist(bins = 20 , figsize= (12,16)) ;
# focussed displots for RH_6 , RH_out , Visibility , Windspeed due to irregular distribution

f, ax = plt.subplots(2,2,figsize=(12,8))

vis1 = sns.distplot(feature_vars["RH_6"],bins=10, ax= ax[0][0])

vis2 = sns.distplot(feature_vars["RH_out"],bins=10, ax=ax[0][1])

vis3 = sns.distplot(feature_vars["Visibility"],bins=10, ax=ax[1][0])

vis4 = sns.distplot(feature_vars["Windspeed"],bins=10, ax=ax[1][1])
# Distribution of values in Applainces column

f = plt.figure(figsize=(12,5))

plt.xlabel('Appliance consumption in Wh')

plt.ylabel('Frequency')

sns.distplot(target_vars , bins=10 ) ;
#Appliance column range with consumption less than 200 Wh

print('Percentage of the appliance consumption is less than 200 Wh')

print(((target_vars[target_vars <= 200].count()) / (len(target_vars)))*100 )
# Use the weather , temperature , applainces and random column to see the correlation

train_corr = train[col_temp + col_hum + col_weather +col_target+col_randoms]

corr = train_corr.corr()

# Mask the repeated values

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

  

f, ax = plt.subplots(figsize=(16, 14))

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, annot=True, fmt=".2f" , mask=mask,)

    #Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

    #Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

    #show plot

plt.show()
def get_redundant_pairs(df):

    '''Get diagonal and lower triangular pairs of correlation matrix'''

    pairs_to_drop = set()

    cols = df.columns

    for i in range(0, df.shape[1]):

        for j in range(0, i+1):

            pairs_to_drop.add((cols[i], cols[j]))

    return pairs_to_drop



# Function to get top correlations 



def get_top_abs_correlations(df, n=5):

    au_corr = df.corr().abs().unstack()

    labels_to_drop = get_redundant_pairs(df)

    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)

    return au_corr[0:n]



print("Top Absolute Correlations")

print(get_top_abs_correlations(train_corr, 40))
#Split training dataset into independent and dependent varibales

train_X = train[feature_vars.columns]

train_y = train[target_vars.columns]

train_X.drop(['date'], axis=1, inplace=True)
#Boruta algorith for feature selection

from sklearn.ensemble import RandomForestRegressor

from boruta import BorutaPy

from datetime import datetime



X=train_X.values

y=train_y.values

y=y.ravel()



def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))



rfc = RandomForestRegressor(n_estimators=100, max_depth=6, criterion='mse')

boruta_selector = BorutaPy(rfc, n_estimators='auto', verbose=2)

start_time = timer(None)

boruta_selector.fit(X, y)

timer(start_time)
boruta_selector.support_
boruta_selector.ranking_
#Split testing dataset into independent and dependent varibales

test_X = test[feature_vars.columns]

test_y = test[target_vars.columns]
# Due to conlusion made above below columns are removed

train_X.drop(["rv1","rv2","Visibility","T6","T9"],axis=1 , inplace=True)



# Due to conlusion made above below columns are removed

test_X.drop(["rv1","rv2","Visibility","T6","T9","date"], axis=1, inplace=True)
train_X.columns
test_X.columns
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()



# Create test and training set by including Appliances column



train = train[list(train_X.columns.values) + col_target ]



test = test[list(test_X.columns.values) + col_target ]



# Create dummy test and training set to hold scaled values



sc_train = pd.DataFrame(columns=train.columns , index=train.index)



sc_train[sc_train.columns] = sc.fit_transform(train)



sc_test= pd.DataFrame(columns=test.columns , index=test.index)



sc_test[sc_test.columns] = sc.fit_transform(test)
sc_train.head()
sc_test.head()
# Remove Appliances column from traininig set



train_X =  sc_train.drop(['Appliances'] , axis=1)

train_y = sc_train['Appliances']



test_X =  sc_test.drop(['Appliances'] , axis=1)

test_y = sc_test['Appliances']
train_X.head()
train_y.head()
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor



models = [

           ['RandomForest ',RandomForestRegressor()],

           ['ExtraTreeRegressor :',ExtraTreesRegressor()]

]
# Run all the proposed models and update the information in a list model_data

import time

from math import sqrt

from sklearn.metrics import mean_squared_error



model_data = []

for name,curr_model in models :

    curr_model_data = {}

    curr_model.random_state = 78

    curr_model_data["Name"] = name

    start = time.time()

    curr_model.fit(train_X,train_y)

    end = time.time()

    curr_model_data["Train_Time"] = end - start

    curr_model_data["Train_R2_Score"] = metrics.r2_score(train_y,curr_model.predict(train_X))

    curr_model_data["Test_R2_Score"] = metrics.r2_score(test_y,curr_model.predict(test_X))

    curr_model_data["Test_RMSE_Score"] = sqrt(mean_squared_error(test_y,curr_model.predict(test_X)))

    model_data.append(curr_model_data)
model_data
df = pd.DataFrame(model_data)

df
df.plot(x="Name", y=['Test_R2_Score' , 'Train_R2_Score' , 'Test_RMSE_Score'], kind="bar" , title = 'R2 Score Results' , figsize= (10,8)) ;
from sklearn.model_selection import GridSearchCV

param_grid = [{

              'max_depth': [80, 150, 200,250],

              'n_estimators' : [100,150,200,250],

              'max_features': ["auto", "sqrt", "log2"]

            }]

reg = ExtraTreesRegressor(random_state=40)

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = reg, param_grid = param_grid, cv = 5, n_jobs = -1 , scoring='r2' , verbose=2)

grid_search.fit(train_X, train_y)
# Tuned parameter set

grid_search.best_params_
# Best possible parameters for ExtraTreesRegressor

grid_search.best_estimator_



# R2 score on training set with tuned parameters



grid_search.best_estimator_.score(train_X,train_y)
# R2 score on test set with tuned parameters

grid_search.best_estimator_.score(test_X,test_y)
# RMSE score on test set with tuned parameters



np.sqrt(mean_squared_error(test_y, grid_search.best_estimator_.predict(test_X)))
# Get sorted list of features in order of importance

feature_indices = np.argsort(grid_search.best_estimator_.feature_importances_)
importances = grid_search.best_estimator_.feature_importances_

indices = np.argsort(importances)[::-1]

names = [train_X.columns[i] for i in indices]

# Create plot

plt.figure(figsize=(10,6))



# Create plot title

plt.title("Feature Importance")



# Add bars

plt.bar(range(train_X.shape[1]), importances[indices])



# Add feature names as x-axis labels

plt.xticks(range(train_X.shape[1]), names, rotation=90)



# Show plot

plt.show()
from math import sqrt

from sklearn.model_selection import train_test_split

from numpy import concatenate

from pandas import read_csv

from pandas import DataFrame

from pandas import concat

from pandas import to_datetime

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error

from keras.models import Sequential

from keras.layers import LSTM

from keras.layers import Dense

from matplotlib import pyplot
# convert series to supervised learning

def series_to_supervised(dataset, n_in=1, n_out=1, dropnan=True):

    num_vars = 1 if type(dataset) is list else dataset.shape[1]

    dataframe = DataFrame(dataset)

    cols, names = list(), list()

    

    # input sequence (t-n, ....t-1)

    for i in range(n_in, 0, -1):

        cols.append(dataframe.shift(i))

        names += [('var%d(t-%d)' % (j+1, i)) for j in range(num_vars)]

    # forecast sequence (t, t+1 .... t+n)

    for i in range(0, n_out):

        cols.append(dataframe.shift(-i))

        if i == 0:

            names += [('var%d(t)' % (j+1)) for j in range(num_vars)]

        else:

            names += [('var%d(t+%d)' % (j+1, i)) for j in range(num_vars)]

    

    # put it all together 

    agg = concat(cols, axis=1)

    agg.columns = names

    

    # drop rows with NaN values

    if dropnan:

        agg.dropna(inplace=True)

    return agg
feature=["RH_out","RH_8","RH_1","T3","RH_3","T2","Press_mm_hg","RH_2","RH_7","T8","RH_6","RH_4","RH_5","T_out","RH_9",

             "T4","T7","Tdewpoint","Windspeed","T1","T5"]

data1 = data[col_target + col_time + feature]
import pandas as pd



data1["date"]=pd.to_datetime(data1["date"])

data1 = data1.set_index(['date'], drop=True)

data1.head()
values=data1.values

values.shape
# normalize features

scaler = MinMaxScaler(feature_range=(0,1))

scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, 1, 1)
reframed.head()
reframed.drop(reframed.columns[[22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43]], axis=1, inplace=True)
values = reframed.values
X = values[:,:21]

Y = values[:,21]
X.shape
X
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.3)



# reshape input to be 3D [samples, timesteps, features]

X_Train = X_Train.reshape((X_Train.shape[0], 1, X_Train.shape[1]))

X_Test = X_Test.reshape((X_Test.shape[0], 1, X_Test.shape[1]))
# network architecture

model = Sequential()

model.add(LSTM(50, input_shape=(X_Train.shape[1], X_Train.shape[2])))

model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')



# fit

history = model.fit(X_Train, Y_Train, epochs=70, batch_size=10, validation_data=(X_Test, Y_Test), verbose=2, shuffle=False)
pyplot.plot(history.history['loss'], label='Train')

pyplot.plot(history.history['val_loss'], label='Test')

pyplot.legend()

pyplot.show()
sca=DataFrame(scaled)

sca.head()
lstm_test_mse = model.evaluate(X_Test, Y_Test, batch_size=1)

print('Test MSE: %f'%lstm_test_mse)
from sklearn.metrics import r2_score



y_pred_test_lstm = model.predict(X_Test)

y_train_pred_lstm = model.predict(X_Train)

print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(Y_Train, y_train_pred_lstm)))

print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(Y_Test, y_pred_test_lstm)))
lstm_y_pred_test = model.predict(X_Test)

plt.figure(figsize=(10, 6))

plt.plot(Y_Test, label='True')

plt.plot(y_pred_test_lstm, label='LSTM')

plt.title("LSTM's Prediction")

plt.xlabel('Observation')

plt.ylabel('Appliances scaled')

plt.legend()

plt.show();
# make a prediction

yhat = model.predict(X_Test)

X_Test = X_Test.reshape((X_Test.shape[0], 21))

# invert scaling for forecast

inv_yhat = np.concatenate((yhat, X_Test[:, -21:]), axis=1)

inv_yhat = scaler.inverse_transform(inv_yhat)

inv_yhat = inv_yhat[:,0]

# invert scaling for actual

Y_Test = Y_Test.reshape((len(Y_Test), 1))

inv_y = np.concatenate((Y_Test, X_Test[:, -21:]), axis=1)

inv_y = scaler.inverse_transform(inv_y)

inv_y = inv_y[:,0]

# calculate RMSE

rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))

print('Test RMSE: %.3f' % rmse)
# Import libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib as mpl

from scipy import stats

import statsmodels.api as sm

import warnings

from itertools import product

from datetime import datetime

warnings.filterwarnings('ignore')

plt.style.use('seaborn-poster')
df = pd.read_csv("../input/KAG_energydata_complete.csv")
data=df[['date','Appliances']]

data.head()
# Unix-time to 

df.Timestamp = pd.to_datetime(df.date)



# Resampling to daily frequency

df.index = df.Timestamp

df = df.resample('D').mean()



# Resampling to monthly frequency

df_month = df.resample('M').mean()

df.head()
fig = plt.figure(figsize=[65, 17])

plt.suptitle('Appliances enegy consumption', fontsize=22)



plt.subplot(221)

plt.plot(df.Appliances, '-', label='By Days')

plt.legend()



plt.subplot(222)

plt.plot(df_month.Appliances, '-', label='By Months')

plt.legend()
print("Dickey–Fuller test: p=%f" % sm.tsa.stattools.adfuller(df.Appliances)[1])
from pylab import rcParams

rcParams['figure.figsize'] = 18, 8



decomposition = sm.tsa.seasonal_decompose(df.Appliances, model='additive')

fig = decomposition.plot()

plt.show()
import itertools

p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(df.Appliances,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
mod = sm.tsa.statespace.SARIMAX(df.Appliances,

                                order=(0, 1, 1),

                                seasonal_order=(1, 0, 0, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)



results = mod.fit()



print(results.summary().tables[1])
pred = results.get_prediction(start=pd.to_datetime('2016-04-20'), dynamic=False)

pred_ci = pred.conf_int()



ax = df.Appliances['2016':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))



ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)



ax.set_xlabel('Date')

ax.set_ylabel('Appliances consumption')

plt.legend()



plt.show()
y_forecasted = pred.predicted_mean

y_truth = df.Appliances['2016-04-20':]



mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error of our forecasts is {}'.format(round(np.sqrt(mse), 2)))
pred_uc = results.get_forecast(steps=15)

pred_ci = pred_uc.conf_int()



ax = df.Appliances.plot(label='observed', figsize=(14, 7))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Appliances consumption')



plt.legend()

plt.show()
import pandas as pd

from fbprophet import Prophet



data = data.rename(columns={'date': 'ds', 'Appliances': 'y'})

model = Prophet(interval_width=0.95)

model.fit(data)



future = model.make_future_dataframe(periods=10, freq='H')

forecast = model.predict(future)



plt.figure(figsize=(30, 6))

model.plot(future, xlabel = 'Date', ylabel = 'comsumption')

plt.title('Appliances consumption');
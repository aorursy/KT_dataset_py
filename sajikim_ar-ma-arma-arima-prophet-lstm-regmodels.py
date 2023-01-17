import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

%matplotlib inline



from math import sqrt

from sklearn.preprocessing import MinMaxScaler

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import mean_squared_error, mean_absolute_error



# Analysis imports

from pandas.plotting import lag_plot

from pylab import rcParams

from statsmodels.tsa.seasonal import seasonal_decompose

from pandas import DataFrame

from pandas import concat



# Modelling imports

from statsmodels.tsa.arima_model import ARIMA

from fbprophet import Prophet

from keras.models import Sequential

from keras.layers import LSTM, GRU, Dense

from keras.layers import Dropout

from keras.preprocessing.sequence import TimeseriesGenerator

from keras.layers.core import Activation

import tensorflow as tf

from keras.initializers import glorot_uniform



from sklearn.linear_model import LinearRegression

from sklearn.svm import SVR

from sklearn.model_selection import train_test_split



import random

from numpy.random import seed



import statsmodels.api as sm

import itertools

import warnings

warnings.filterwarnings('ignore')
def show_graph(train, test=None, pred=None, title=None):

    

    fig = plt.figure(figsize=(20, 5))



    # entire data

    ax1 = fig.add_subplot(121)

    ax1.set_xlabel('Dates')

    ax1.set_ylabel('Price')

    ax1.plot(train.index, train['Price'], color='green', label='Train price')

    if test is not None:

        ax1.plot(test.index, test['Price'], color='red', label='Test price')

    if pred is not None:

        if 'yhat' in pred.columns:

            ax1.plot(pred.index, pred['yhat'], color = 'blue', label = 'Predicted price')

            ax1.fill_between(pred.index, pred['yhat_lower'], pred['yhat_upper'], color='grey', label="Band Range")

        else:

            ax1.plot(pred.index, pred['Price'], color='blue', label='Predicted price')

    ax1.legend()

    if title is not None:

        plt.title(title + ' (Entire)')

    plt.grid(True)



    # zoom data

    period=50

    period=int(0.2*len(train))

    ax2 = fig.add_subplot(122)

    ax2.set_xlabel('Dates')

    ax2.set_ylabel('Price')

    ax2.plot(train.index[-period:], train['Price'].tail(period), color='green', label='Train price')

    if test is not None:

        ax2.plot(test.index, test['Price'], color='red', label='Test price')

    if pred is not None:

        if 'yhat' in pred.columns:

            ax2.plot(pred.index, pred['yhat'], color = 'blue', label = 'Predicted price')

            ax2.fill_between(pred.index, pred['yhat_lower'], pred['yhat_upper'], color='grey', label="Band Range")

        else:

            ax2.plot(pred.index, pred['Price'], color='blue', label='Predicted price')

    ax2.legend()

    if title is not None:

        plt.title(title + ' (Recent ' + str(period) + ')')

    plt.grid(True)



    fig.subplots_adjust(wspace=0.1, hspace=0.1)

    plt.show()

    

def make_future_dates(last_date, period):

    prediction_dates=pd.date_range(last_date, periods=period+1, freq='B')

    return prediction_dates[1:]



def calculate_accuracy(forecast, actual, algorithm):

    mse  = round(mean_squared_error(actual, forecast),4)

    mae  = round(mean_absolute_error(actual, forecast),4)

    rmse = round(sqrt(mean_squared_error(actual, forecast)),4)

    return ({'algorithm':algorithm, 'mse':mse, 'mae':mae, 'rmse': rmse})
def get_data_from_EIA_local():

    df = pd.read_csv("../input/cushing-ok-wti-spot-price-fob/Cushing_OK_WTI_Spot_Price_FOB_20200706.csv", header=4, parse_dates=[0])

    df.columns=["Date", "Price"]

    df.set_index('Date', inplace=True)

    df.sort_index(inplace=True)

    return df
df_org=get_data_from_EIA_local()

data=df_org['2019-07-06':'2020-07-06'].copy()

data.Price["2020-04-20"]=(data.Price["2020-04-17"] + data.Price["2020-04-21"]) / 2



acc_sum=[]

df_preds=pd.DataFrame({"Date":make_future_dates('2020-07-06',34)})

df_preds=df_preds.set_index('Date', drop=True)



# Display OIL price

plt.figure(figsize=(10,5))

plt.xlabel('Dates')

plt.ylabel('Price')

plt.plot(data['Price']);

plt.grid(True)

plt.show()
# Show LAG

fig = plt.figure(figsize=(10, 6))

lag_plot(data['Price'], lag=5)

plt.title('Lag')

plt.grid(True)

plt.legend();



# Show Diff

data_diff = data - data.shift() 

data_diff = data_diff.dropna()

plt.figure(figsize=(10, 6))

plt.title('Diff')

plt.grid(True)

plt.plot(data_diff);



fig = plt.figure(figsize=(8, 6))



# Show ACF

ax1 = fig.add_subplot(211)

sm.graphics.tsa.plot_acf(data_diff, lags=40, ax=ax1)



# Show PACF

ax2 = fig.add_subplot(212)

sm.graphics.tsa.plot_pacf(data_diff, lags=40, ax=ax2)



plt.tight_layout()
result = seasonal_decompose(data.Price[-1000:], model='additive', freq=30)

plt.figure(figsize=(16,10))

fig = result.plot()

plt.show()
values = DataFrame(data['Price'].values)

dataframe = concat([values.shift(1),values.shift(5),values.shift(10),values.shift(30), values], axis=1)

dataframe.columns = ['t', 't+1', 't+5', 't+10', 't+30']

result = dataframe.corr()

print(result)
adf_result = sm.tsa.stattools.adfuller(data['Price'].values, autolag ='AIC')

adf = pd.Series(adf_result[0:4], index = ['Test Statistic', 'p-　　value', '#Lags Used', 'Number of Observations Used'])

print(adf)
split = int(0.80*len(data))

train_data, test_data = data[0:split], data[split:]

show_graph(train_data,test_data,title='Train & Test')
def evaluate_arima_model(train, test, order, maxlags=8, ic='aic'):

    # feature Scaling

    stdsc = StandardScaler()

    train_std = stdsc.fit_transform(train.values.reshape(-1, 1))

    test_std = stdsc.transform(test.values.reshape(-1, 1))

    # prepare training dataset

    history = [x for x in train_std]

    # make predictions

    predictions = list()

    # rolling forecasts

    for t in range(len(test_std)):

        # predict

        model = ARIMA(history, order=order)

        model_fit = model.fit(maxlags=maxlags, ic=ic, disp=0)

        yhat = model_fit.forecast()[0]

        # invert transformed prediction

        predictions.append(yhat)

        # observation

        history.append(test_std[t])

    # inverse transform

    predictions = stdsc.inverse_transform(np.array(predictions).reshape((-1)))

    # calculate mse

    mse = mean_squared_error(test, predictions)

    return predictions, mse



def evaluate_arima_models(train, test, p_values, d_values, q_values):

    best_score, best_cfg = float("inf"), None

    pdq = list(itertools.product(p_values, d_values, q_values))

    for order in pdq:

        try:

            predictions, mse = evaluate_arima_model(train, test, order)

            if mse < best_score:

                best_score, best_cfg = mse, order

            print('Model(%s) mse=%.3f' % (order,mse))

        except:

            continue

    print('Best Model(%s) mse=%.3f' % (best_cfg, best_score)) 

    return best_cfg



def predict_arima_model(train, period, order, maxlags=8, ic='aic'):

    # Feature Scaling

    stdsc = StandardScaler()

    train_std = stdsc.fit_transform(train.values.reshape(-1, 1))

    # fit model

    model = ARIMA(train_std, order=order)

    model_fit = model.fit(maxlags=maxlags, ic=ic, disp=0)

    # make prediction

    yhat = model_fit.predict(len(train), len(train) + period -1, typ='levels')

    # inverse transform

    yhat = stdsc.inverse_transform(np.array(yhat).flatten())

    return yhat
model_name='AR Model'



# evaluate parameters

p_values = range(1, 4)

d_values = [0]

q_values = [0]

#evaluate_arima_models(train_data['Price'], test_data['Price'], p_values, d_values, q_values)



# predict test period with best parameter

predictions, mse = evaluate_arima_model(train_data['Price'], test_data['Price'],(1, 0, 0))

df_pred = pd.DataFrame({'Price':predictions},index=test_data.index)



# calculate performance metrics

acc = calculate_accuracy(predictions, test_data['Price'], model_name)

print(acc)

acc_sum.append(acc)



# show result

show_graph(train_data,test_data,df_pred,title=model_name+'\nTest period prediction')



# predict future period with best parameter

forecast_out = 34

future_dates = make_future_dates(data.index[-1], forecast_out)

predictions = predict_arima_model(data,len(future_dates),(1, 0, 0))

df_pred = pd.DataFrame({'Price':predictions},index=future_dates)



# show result

show_graph(data,None,df_pred,title=model_name+'\nFuture period prediction')



df_preds[model_name] = df_pred['Price']
model_name='MA Model'



# evaluate parameters

p_values = [0]

d_values = [0]

q_values = range(1, 4)

#evaluate_arima_models(train_data['Price'], test_data['Price'], p_values, d_values, q_values)



# predict test period with best parameter

predictions, mse = evaluate_arima_model(train_data['Price'], test_data['Price'],(0, 0, 1))

df_pred = pd.DataFrame({'Price':predictions},index=test_data.index)



# calculate performance metrics

acc = calculate_accuracy(predictions, test_data['Price'], model_name)

print(acc)

acc_sum.append(acc)



# show result

show_graph(train_data, test_data, df_pred, title=model_name + '\nTest period prediction')



# predict future period with best parameter

forecast_out = 34

future_dates = make_future_dates(data.index[-1], forecast_out)

predictions = predict_arima_model(data,len(future_dates),(0, 0, 1))

df_pred = pd.DataFrame({'Price':predictions},index=future_dates)



# show result

show_graph(data,None,df_pred,title=model_name+'\nFuture period prediction')



df_preds[model_name] = df_pred['Price']
model_name='ARMA Model'



# evaluate parameters

p_values = range(0, 1, 2)

d_values = [0]

q_values = range(0, 1, 2)

#evaluate_arima_models(train_data['Price'].tail, test_data['Price'], p_values, d_values, q_values)



# predict test period with best parameter

#predictions, mse = evaluate_arima_model(train_data['Price'], test_data['Price'],(1, 0, 1))

#df_pred = pd.DataFrame({'Price':predictions},index=test_data.index)



## calculate performance metrics

#acc = calculate_accuracy(predictions, test_data['Price'], model_name)

#print(acc)

#acc_sum.append(acc)



# show result

#show_graph(train_data, test_data, df_pred, title=model_name + '\nTest period prediction')



# predict future period with best parameter

forecast_out = 34

future_dates = make_future_dates(data.index[-1], forecast_out)

predictions = predict_arima_model(data,len(future_dates),(2, 0, 1))

df_pred = pd.DataFrame({'Price':predictions},index=future_dates)



# show result

show_graph(data,None,df_pred,title=model_name+'\nFuture period prediction')



df_preds[model_name] = df_pred['Price']
model_name='ARIMA Model'



# evaluate parameters

p_values = [1, 2, 4, 6, 8, 10]

d_values = range(0, 3)

q_values = range(1, 3)

#evaluate_arima_models(train_data['Price'], test_data['Price'], p_values, d_values, q_values)



# predict test period with best parameter

predictions, mse = evaluate_arima_model(train_data['Price'], test_data['Price'],(2, 1, 1))

df_pred = pd.DataFrame({'Price':predictions},index=test_data.index)



# calculate performance metrics

acc = calculate_accuracy(predictions, test_data['Price'],model_name)

print(acc)

acc_sum.append(acc)



# show result

show_graph(train_data, test_data, df_pred, title=model_name + '\nTest period prediction')



# predict future period with best parameter

forecast_out = 34

future_dates = make_future_dates(data.index[-1], forecast_out)

predictions = predict_arima_model(data,len(future_dates),(2, 1, 1))

df_pred = pd.DataFrame({'Price':predictions},index=future_dates)



# show result

show_graph(data,None,df_pred,title=model_name+'\nFuture period prediction')



df_preds[model_name] = df_pred['Price']
model_name='Facebook Prophet'



def predict_prophet(train,period):

    # create model

    prop = Prophet(growth='logistic',

                    n_changepoints=40,

                    changepoint_range=1,

                    changepoint_prior_scale=0.5,

                    weekly_seasonality=False,

                    yearly_seasonality=False

                  )

    # prepare training dataset

    ph_df_train = pd.DataFrame({'y':train['Price'].values, 'ds':train.index})

    ph_df_train['cap'] = 100

    ph_df_train['floor'] = 0

    prop.fit(ph_df_train)

    # create future dates

    future_prices = prop.make_future_dataframe(periods=period, freq = 'd')

    future_prices['cap'] = 100

    future_prices['floor'] = 0

    # predict prices

    forecast = prop.predict(future_prices)

    predicted=forecast[-period:]

    return predicted



# predict test period

predictions = predict_prophet(train_data,len(test_data))

predictions.index = test_data.index



# calculate performance metrics

acc = calculate_accuracy(predictions['yhat'], test_data['Price'], model_name)

print(acc)

acc_sum.append(acc)



# show result

show_graph(train_data, test_data, predictions, title=model_name + '\nTest period prediction')



# predict future period

forecast_out = 34

future_dates = make_future_dates(data.index[-1], forecast_out)

predictions = predict_prophet(data,len(future_dates))

predictions.index = future_dates



# show result

show_graph(data,None,predictions,title=model_name+'\nFuture period prediction')



df_preds[model_name] = predictions['yhat']
def set_random_seed(seed):

    random.seed(seed)

    np.random.seed(seed)

    tf.random.set_seed(seed)



def create_lstm_data(train,test,look_back):

    train_lstm = train

    test_lstm = test

    train_gen = TimeseriesGenerator(train_lstm, train_lstm, length=look_back, batch_size=20)     

    test_gen = TimeseriesGenerator(test_lstm, test_lstm, length=look_back, batch_size=1)

    return train_gen, test_gen



def create_lstm_model(neurons, activ_func="linear",

                dropout=0.10, loss="mean_squared_error", optimizer="adam"):

    set_random_seed(20200715)

    model = Sequential()

    

    model.add(LSTM(neurons,

                   input_shape=(look_back,1),

                   kernel_initializer=glorot_uniform(seed=20200715)

                  ))

    model.add(Dropout(dropout))

    model.add(Dense(units=1,

                   kernel_initializer=glorot_uniform(seed=20200715)

                  ))

    model.add(Activation(activ_func))



    model.compile(loss=loss, optimizer=optimizer)

    return model



def predict_lstm_model(data, period, model):

    prediction_list = data[-look_back:]

    

    for _ in range(period):

        x = prediction_list[-look_back:]

        x = x.reshape((1, look_back, 1))

        out = model.predict(x)[0][0]

        prediction_list = np.append(prediction_list, out)

        

    prediction_list = prediction_list[look_back-1:]

    prediction_list = prediction_list[1:]

    return prediction_list



def show_lstm_history(history):

    loss = history.history['loss']

    epochs = range(1, len(loss) + 1)

    plt.figure()

    plt.plot(epochs, loss,  label='Training loss')

    plt.title('validation loss')

    plt.legend()

    plt.show()
# feature Scaling

stdsc = StandardScaler()

train_lstm = stdsc.fit_transform(train_data.values.reshape(-1, 1))

test_lstm = stdsc.transform(test_data.values.reshape(-1, 1))



# create data

look_back = 7

train_gen, test_gen = create_lstm_data(train_lstm, test_lstm, look_back)



# create model

model = create_lstm_model(300)

model.summary()



# training

history = model.fit_generator(train_gen, epochs=100, verbose=1, shuffle=False)

show_lstm_history(history)
model_name='LSTM'



# predict test period

prediction = model.predict_generator(test_gen)



# inverse transform

prediction = stdsc.inverse_transform(prediction.reshape((-1)))

df_pred = pd.DataFrame({'Price':prediction},index=test_data[:len(prediction)].index)



# calculate performance metrics

acc = calculate_accuracy(prediction, test_data[:len(prediction)], model_name)

print(acc)

acc_sum.append(acc)



# show result

show_graph(train_data,test_data,df_pred,title=model_name+'\nTest period prediction')



# predict future period

forecast_out = 34

train_lstm = stdsc.transform(data.values.reshape(-1, 1))

prediction = predict_lstm_model(train_lstm, forecast_out, model)



# inverse transform

prediction = prediction.reshape((-1))

prediction = stdsc.inverse_transform(np.array(prediction).flatten())



# show result

future_dates = make_future_dates(data.index[-1], forecast_out)

df_pred = pd.DataFrame({'Price':prediction},index=future_dates)

show_graph(data, None, df_pred, title=model_name + '\nFuture period prediction')



df_preds[model_name] = df_pred['Price']
def prepare_data(data2, forecast_out, test_size):

    label = np.roll(data2, -forecast_out).reshape((-1))

    X = data2; 

    X_lately = X[-forecast_out:]

    X = X[:-forecast_out] 

    y = label[:-forecast_out] 

    if test_size == 0:

        X_train, X_test , Y_train, Y_test = X, np.empty(0), y, np.empty(0)

    else:

        X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=test_size,shuffle=False) 

    return [X_train, X_test , Y_train, Y_test , X_lately];
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, SGDRegressor

from sklearn.linear_model import PassiveAggressiveRegressor, ARDRegression, RidgeCV

from sklearn.linear_model import TheilSenRegressor, RANSACRegressor, HuberRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.svm import SVR, LinearSVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.experimental import enable_hist_gradient_boosting

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor

from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import Pipeline

from sklearn.cross_decomposition import PLSRegression



reg_dict = {"LinearRegression": LinearRegression(),

            #"Ridge": Ridge(),

            "Lasso": Lasso(),

            "ElasticNet": ElasticNet(), 

            #"Polynomial_deg2": Pipeline([('poly', PolynomialFeatures(degree=2)),('linear', LinearRegression())]),

            #"Polynomial_deg3": Pipeline([('poly', PolynomialFeatures(degree=3)),('linear', LinearRegression())]),

            #"Polynomial_deg4": Pipeline([('poly', PolynomialFeatures(degree=4)),('linear', LinearRegression())]),

            #"Polynomial_deg5": Pipeline([('poly', PolynomialFeatures(degree=5)),('linear', LinearRegression())]),

            #"KNeighborsRegressor": KNeighborsRegressor(n_neighbors=3),

            #"DecisionTreeRegressor": DecisionTreeRegressor(),

            #"RandomForestRegressor": RandomForestRegressor(),

            #"SVR_rbf": SVR(kernel='rbf', C=1e3, gamma=0.1, epsilon=0.1, degree=3),

            "SVR_linear": SVR(kernel='linear', C=1e3, gamma=0.1, epsilon=0.1, degree=3),

            #"GaussianProcessRegressor": GaussianProcessRegressor(),

            #"SGDRegressor": SGDRegressor(),

            #"MLPRegressor": MLPRegressor(hidden_layer_sizes=(10,10), max_iter=100, early_stopping=True, n_iter_no_change=5),

            #"ExtraTreesRegressor": ExtraTreesRegressor(n_estimators=100), 

            ##"PLSRegression": PLSRegression(n_components=34),

            #"PassiveAggressiveRegressor": PassiveAggressiveRegressor(max_iter=100, tol=1e-3),

            "TheilSenRegressor": TheilSenRegressor(random_state=0),

            "RANSACRegressor": RANSACRegressor(random_state=0),

            #"HistGradientBoostingRegressor": HistGradientBoostingRegressor(),

            #"AdaBoostRegressor": AdaBoostRegressor(random_state=0, n_estimators=100),

            #"BaggingRegressor": BaggingRegressor(base_estimator=SVR(), n_estimators=10),

            #"GradientBoostingRegressor": GradientBoostingRegressor(random_state=0),

            #"VotingRegressor": VotingRegressor([('lr', LinearRegression()), ('rf', RandomForestRegressor(n_estimators=10))]),

            #"StackingRegressor": StackingRegressor(estimators=[('lr', RidgeCV()), ('svr', LinearSVR())], final_estimator=RandomForestRegressor(n_estimators=10)),

            #"ARDRegression": ARDRegression(),

            "HuberRegressor": HuberRegressor(),

            }
for reg_name, reg in reg_dict.items():



    # prepare data

    forecast_out = 34

    x_train, x_test, y_train, y_test, X_lately = prepare_data(data,forecast_out,0.2)



    # feature Scaling

    stdsc = StandardScaler()

    x_train_std = stdsc.fit_transform(x_train)

    y_train_std = stdsc.transform(y_train.reshape(-1, 1))

    x_test_std = stdsc.transform(x_test)

    y_test_std = stdsc.transform(y_test.reshape(-1, 1))



    # create and train model

    reg.fit(x_train_std, y_train_std)



    # Predict test period

    prediction = reg.predict(x_test_std)



    # inverse transform

    prediction = stdsc.inverse_transform(prediction.reshape((-1)))



    # calculate performance metrics

    acc = calculate_accuracy(prediction, y_test, reg_name)

    print(acc)

    acc_sum.append(acc)



    # show result

    future_dates1 = data.index[forecast_out:forecast_out+len(x_train)]

    future_dates2 = data.index[forecast_out+len(x_train):]

    df_train = pd.DataFrame({'Price':y_train},index=future_dates1)

    df_test = pd.DataFrame({'Price':y_test},index=future_dates2)

    df_pred = pd.DataFrame({'Price':prediction},index=future_dates2)

    show_graph(df_train,df_test,df_pred,title=reg_name+'\nTest period prediction')





    # prepare data

    forecast_out = 34

    x_train, x_test, y_train, y_test, X_lately = prepare_data(data,forecast_out,0.0)



    # feature Scaling

    stdsc = StandardScaler()

    x_train_std = stdsc.fit_transform(x_train)

    y_train_std = stdsc.transform(y_train.reshape(-1, 1))

    X_lately_std = stdsc.transform(X_lately)



    # create and train model

    reg.fit(x_train_std, y_train_std)



    # Predict future period

    prediction = reg.predict(X_lately_std)



    # inverse transform

    prediction = stdsc.inverse_transform(prediction.reshape((-1)))



    # show result

    future_dates = make_future_dates(data.index[-1], forecast_out)

    df_pred = pd.DataFrame({'Price':prediction},index=future_dates)

    show_graph(data,None,df_pred,title=reg_name+'\nFuture period prediction')



    df_preds[reg_name] = df_pred['Price']
df_sum = pd.DataFrame(acc_sum)

df_sum = df_sum.sort_values('mae', ascending=True)

df_sum = df_sum.reset_index(drop=True)

df_sum
df_preds
# display

plt.figure(figsize=(16, 8))

plt.plot(data.index[-100:], data['Price'].tail(100),label="Train")

for col in df_preds.columns:

    plt.plot(df_preds.index[-len(df_preds):], df_preds[col][-len(df_preds):],label=col)

    

plt.vlines([data.index[-1]], 0, 60, "red", linestyles='dashed')

plt.text([data.index[-1]], 60, 'Today', backgroundcolor='white', ha='center', va='center')

plt.vlines([data.index[-1-75]], 0, 60, "red", linestyles='dashed')

plt.text([data.index[-1-75]], 60, '75 days before', backgroundcolor='white', ha='center', va='center')

plt.vlines([df_preds.index[-1]], 0, 60, "red", linestyles='dashed')

plt.text([df_preds.index[-1]], 60, '34 days after', backgroundcolor='white', ha='center', va='center')



plt.title('Predictions (Raw Price)')

plt.xlabel('Date')

plt.ylabel('Price')

plt.legend(loc='lower right',ncol=1)

plt.grid(True)

plt.show()
# ma75 calculation 

df_ma75 = df_org.copy() 

for col in df_preds.columns:

    df_ma75[col] = df_org['Price'].copy() 

df_ma75 = pd.concat([df_ma75, df_preds])

df_ma75 = df_ma75.rolling(75).mean()

df_ma75[-len(df_preds):]
def disp_all(mode='entire'):

    plt.figure(figsize=(16, 8))

    for col in df_ma75.columns:

        if col != "Price":

            plt.plot(df_ma75.index[-len(df_preds):], df_ma75[col][-len(df_preds):],label=col)



    if mode is 'entire':

        plt.plot(df_ma75.index[-100:], df_ma75['Price'].tail(100),label="Train-MA75")

        plt.plot(data.index[-100:], data['Price'].tail(100),label="Train-Raw")

        plt.vlines([data.index[-1]], 0, 60, "red", linestyles='dashed')

        plt.text([data.index[-1]], 60, 'Today', backgroundcolor='white', ha='center', va='center')

        plt.vlines([data.index[-1-75]], 0, 60, "red", linestyles='dashed')

        plt.text([data.index[-1-75]], 60, '75 days before', backgroundcolor='white', ha='center', va='center')

        plt.vlines([df_ma75.index[-1]], 0, 60, "red", linestyles='dashed')

        plt.text([df_ma75.index[-1]], 60, '34 days after', backgroundcolor='white', ha='center', va='center')



    plt.title('Predictions (MA75 Price '+mode+')')

    plt.xlabel('Date')

    plt.ylabel('Price')

    plt.legend(loc='best',ncol=2)

    plt.grid(True)

    plt.show()

    

disp_all('entire')

disp_all('zoom')
template = pd.read_csv('../input/ntt-data-global-ai-challenge-06-2020/sampleSubmission0710_updated.csv', header=0, parse_dates=[0])

template.drop("Price",axis=1,inplace=True)



for col in df2.columns:

    submission = pd.merge(template, df2[col], on='Date', how='left')

    submission.rename(columns={col: 'Price'},inplace=True)

    if submission["Price"].isnull().any():

        #print("[Warning] NaN found in " + col +  " ("+str(submission["Price"].isnull().sum()) + ")")

        submission["Price"].fillna(submission["Price"].mean(),inplace=True)

    submission["Price"] = submission["Price"].round(9)

    submission.to_csv("submission_" + col + ".csv", index=False)
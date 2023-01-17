!pip install pmdarima
import pandas as pd

import numpy as np

from matplotlib import pyplot

import seaborn as sns

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from statsmodels.tsa.arima_model import ARIMA

from statsmodels.tsa.statespace.sarimax import SARIMAX

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.preprocessing import MinMaxScaler

from pmdarima.arima import auto_arima

import statsmodels.api as sm

from statsmodels.tsa.stattools import adfuller

%matplotlib inline
# METHOD SELECTION

###########################################################################################

dataset= pd.read_csv('../input/queens-zillow/Queens_zillow.csv')
steps=-1

dataset_pred = dataset.copy()

dataset_pred['Actual']=dataset_pred['Prices'].shift(steps)

dataset_pred.head(3)
dataset_pred=dataset_pred.dropna()
dataset_pred['Date'] =pd.to_datetime(dataset_pred['Date'])

dataset_pred.index= dataset_pred['Date']
dataset_pred.head()
dataset_pred['Prices'].plot(color='green', figsize=(15,2))

pyplot.legend(['Next month value', 'Prices'])

pyplot.title("Zillow queens 3 bedroom rent")
sc_in = MinMaxScaler(feature_range=(0, 1))

scaled_input = sc_in.fit_transform(dataset_pred[['Prices']])

scaled_input = pd.DataFrame(scaled_input)

X= scaled_input
sc_out = MinMaxScaler(feature_range=(0, 1))

scaler_output = sc_out.fit_transform(dataset_pred[['Actual']])

scaler_output =pd.DataFrame(scaler_output)

y=scaler_output
X.rename(columns={0:'Prices'}, inplace=True)

X.head(2)
y.rename(columns={0:'Price_next_month'}, inplace= True)

y.index=dataset_pred.index

y.head(2)
train_size=int(len(dataset) *0.7)

test_size = int(len(dataset)) - train_size

train_X, train_y = X[:train_size].dropna(), y[:train_size].dropna()

test_X, test_y = X[train_size:].dropna(), y[train_size:].dropna()


seas_d=sm.tsa.seasonal_decompose(X['Prices'],model='add',freq=10);

fig=seas_d.plot()

fig.set_figheight(4)

pyplot.show()
def test_adf(series, title=''):

    dfout={}

    dftest=sm.tsa.adfuller(series.dropna(), autolag='AIC', regression='ct')

    for key,val in dftest[4].items():

        dfout[f'critical value ({key})']=val

    if dftest[1]<=0.05:

        print("Strong evidence against Null Hypothesis")

        print("Reject Null Hypothesis - Data is Stationary")

        print("Data is Stationary", title)

    else:

        print("Strong evidence for  Null Hypothesis")

        print("Accept Null Hypothesis - Data is not Stationary")

        print("Data is NOT Stationary for", title)
y_test=y['Price_next_month'][:train_size].dropna()

test_adf(y_test, "Rent Price")
test_adf(y_test.diff(), "Rent Price")
fig,ax= pyplot.subplots(2,1, figsize=(10,5))

fig=sm.tsa.graphics.plot_acf(y_test, lags=60, ax=ax[0])

fig=sm.tsa.graphics.plot_pacf(y_test, lags=60, ax=ax[1])

pyplot.show()
step_wise=auto_arima(train_y, 

                     exogenous= train_X,

                     start_p=1, start_q=1, 

                     max_p=7, max_q=7, 

                     d=1, max_d=7,

                     trace=True, 

                     error_action='ignore', 

                     suppress_warnings=True, 

                     stepwise=True)
step_wise.summary()

# END OF METHOD SELECTION

###########################################################################################
# METHODS ##############################################################################

# ARIMA 

def parser(x):

    splitted = x.split('-')

    return pd.datetime.strptime(splitted[0] + "-" + splitted[1] + "-20" + splitted[2], '%d-%b-%Y')



series = pd.read_csv('../input/queens-zillow/Queens_zillow.csv',header=0,parse_dates=[0],index_col=0, squeeze=True, date_parser=parser)

series

X = series.values



size = int(74)

train, test = X[0:size], X[size:len(X)]



hist = [x for x in train]

predictions = list()



# for t in range(len(test)):

#     model = ARIMA(hist, order=(0, 1, 1))

#     model_fit = model.fit(disp=0)

#     output = model_fit.forecast()

#     yhat = output[0]

#     predictions.append(yhat)

#     obs = test[t]

#     hist.append(obs)

#     print('predicted=%f, expected=%f' % (yhat, obs))



SARIMAX

for t in range(len(test)):

    model = SARIMAX(hist, order=(0, 1, 0))

    model_fit = model.fit(disp=0)

    output = model_fit.forecast()

    yhat = output[0]

    predictions.append(yhat)

    obs = test[t]

    hist.append(obs)

    print('predicted=%f, expected=%f' % (yhat, obs))



error = mean_squared_error(test, predictions)

r2_sc = r2_score(test, predictions)

print('Test MSE: %.2f' % error)

print('r2_score: %.2f' % r2_sc)



pyplot.plot(test)

pyplot.plot(predictions, color='red')

pyplot.show()



# ###########################################################################################
# UNTESTED CODE

# model= SARIMAX(train_y, 

#                exog=train_X,

#                order=(0,1,1),

#                enforce_invertibility=False, 

#                enforce_stationarity=False)



# results= model.fit()



# predictions= results.predict(start =train_size, end=train_size+test_size+(steps)-1,exog=test_X)



# forecast_1= results.forecast(steps=test_size-1, exog=test_X)



# act= pd.DataFrame(scaler_output.iloc[train_size:, 0])



# predictions=pd.DataFrame(predictions)

# predictions.reset_index(drop=True, inplace=True)

# predictions.index=test_X.index

# predictions['Actual'] = act['Price_next_month']

# predictions.rename(columns={0:'Pred'}, inplace=True)



# predictions['Actual'].plot(figsize=(20,8), legend=True, color='blue')

# predictions['Pred'].plot(legend=True, color='red', figsize=(20,8))



# forecasting= pd.DataFrame(forecast_1)

# forecasting.reset_index(drop=True, inplace=True)

# forecasting.index=test_X.index

# forecasting['Actual'] =scaler_output.iloc[train_size:, 0]

# forecasting.rename(columns={0:'Forecast'}, inplace=True)



# forecasting['Forecast'].pyplot(legend=True)

# forecasting['Actual'].pyplot(legend=True)

# END OF UNTESTED CODE
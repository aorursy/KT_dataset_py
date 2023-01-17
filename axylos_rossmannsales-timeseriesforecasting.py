# Importing Required Libraries
# Packages for data manipulation
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Packages for Visualisations
import matplotlib.pyplot as plt
import seaborn as sns 
%matplotlib inline

# Packages for dealing with warning
import warnings
warnings.filterwarnings("ignore")

#Packages for SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
train = pd.read_csv('../input/rossmann-store-sales/train.csv', index_col = 'Date')
store = pd.read_csv('../input/rossmann-store-sales/store.csv')
test = pd.read_csv('../input/rossmann-store-sales/test.csv', index_col = 'Date')
# Exploring Test data
test.head()
test.dtypes
test['Open'].fillna(value = 0 , inplace = True)
test.shape
test.Open = test.Open.astype('int64')
test.StateHoliday = test.StateHoliday.astype('str')
test.dtypes
# Label Encoding the 'StateHoliday' Column
label_encoder = LabelEncoder()

test1 = test.copy()
test1['StateHoliday'] = label_encoder.fit_transform(test1['StateHoliday'])
test1.head()
test1.dtypes
test1['Store'].value_counts()
test1['Store'].nunique()
Store_list = test1['Store'].unique()
print(Store_list)
#train.head()
store.head()
train.shape
store.shape
# Dropping unrequired columns in train
# We will not be using store data as we are doing independent time series forecasting for individual stores.
train.drop(['Customers'], axis = 1, inplace = True)
train.head()
train.sort_index(inplace = True)
#train[train['StateHoliday']==0].StateHoliday ='0'
train.StateHoliday = train.StateHoliday.astype(str)
train['StateHoliday'].value_counts()
train.sort_values(['Date','Store'], inplace = True)
train.head()
cols = train.columns.tolist()
new_cols = ['Sales','Store','DayOfWeek','Open', 'Promo','StateHoliday','SchoolHoliday']
train = train[new_cols]

#Label encoding 'StateHoliday' column
train1 = train.copy()
train.sort_index(inplace = True)
train1['StateHoliday'] = label_encoder.fit_transform(train['StateHoliday'])
train1.tail()
train1.shape
train1['StateHoliday'].value_counts()
storeno1 = train1[train1['Store']==1]
storeno1.asfreq(freq ='D', fill_value = 0) 
storeno1.head()
#storeno1.shape
#storeno1.dtypes
#storeno1['StateHoliday'].value_counts()
train_size=int(len(storeno1) *0.7)
test_size = int(len(storeno1)) - train_size

store1 = storeno1[:train_size]
validation_store1 = storeno1[train_size:]
plt.plot(store1.index,store1.Sales)
# Perfomring the Augmented Dicky-fuller test to check stationarity
results1 = adfuller(store1['Sales'])
results2 = adfuller(store1.Sales.diff().dropna())
results3 = adfuller(store1.Sales.diff().diff().dropna())
print(results1)
print(results2)
print(results3)

#Since the test static is very negative and p-value is close to zero, we can confirm stationarity of the data. 
#Hence,the order of difference is 1.
# Seasonal Behaviour
decomp_results = seasonal_decompose(store1['Sales'], freq=7)
type(decomp_results)
decomp_results.plot()
plt.show()

# Clearly, we see that there is a weekly and yearly seasonal pattern. But, we will focus only on the weekly sesonal pattern

# Plotting ACF and PACF to find order of differencing
# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))

# Make ACF plot
plot_acf(store1['Sales'].diff().dropna(), lags=25, zero=False, ax=ax1)

# Make PACF plot
plot_pacf(store1['Sales'].diff().dropna(), lags=25, zero=False, ax=ax2)

plt.show()

# From the ACF and PACF plots, we can say that a good estimate of order is p:1 , q:2
# Plotting ACF and PACF to find order of seasonal differencing
# Create figure
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,8))

# Make ACF plot
plot_acf(store1['Sales'].diff(7).dropna(), lags=[7,14,21,28,35,42,49,56], zero=False, ax=ax1)

# Make PACF plot
plot_pacf(store1['Sales'].diff(7).dropna(), lags=[7,14,21,28,35,42,49,56], zero=False, ax=ax2)

plt.show()

# By using a seasonal differencing of 1 , we were able to make the data stationary.
# From the ACF and PACF plots, we are not able to conclude the optimal values of P,Q
# # Choosing best model using pmdarima
# # AIC chooses better predictive models. BIC choose good explanatory models.We will chose AIC criteria.

# # Could use the following if pmdarima was available
# # results = pm.auto_arima( store1, # data
# #                          d=2, # non-seasonal difference order
# #                          start_p=0, # initial guess for p
# #                          start_q=2, # initial guess for q
# #                          max_p=2, # max value of p to test
# #                          max_q=2, # max value of q to test
                        
# #                          seasonal=True, # is the time series seasonal
# #                          m=7, # the seasonal period
# #                          D=1, # seasonal difference order
# #                          start_P=1, # initial guess for P
# #                          start_Q=1, # initial guess for Qa
# #                          max_P=3, # max value of P to test
# #                          max_Q=3, # max value of Q to test
                        
# #                         information_criterion='aic', # used to select best model
# #                         trace=True, # print results whilst training
# #                         error_action='ignore', # ignore orders that don't work
# #                         stepwise=False, # apply intelligent order search
# #                        )

# #Using iterative method
# order_aic_bic =[]

# # Loop over non-seasonal AR order
# for p in range(3):
#     # Loop over non-seasonal MA order
#     for q in range(3):
#         # Loop over seasonal AR order
#         for P in range(3):
#             # Loop over seasonal MA order
#             for Q in range(3):
#                 try:
#                     # Fit model
#                     model = SARIMAX(store1.iloc[:,0], order=(p,1,q), seasonal_order=(P,1,Q,7) , exog = store1.iloc[:,1:])
#                     results = model.fit()
#                     # Add order and scores to list
#                     order_aic_bic.append((p, q,P,Q, results.aic, results.bic))
#                 except:
#                     # Add order and scores to list
#                     order_aic_bic.append((p, q,P,Q, None, None))
            
# # Make DataFrame of model order and AIC/BIC scores
# order_df = pd.DataFrame(order_aic_bic, columns=['p','q','P','Q', 'aic', 'bic'])

# Sort by AIC
print(order_df.sort_values('aic'))
# Sort by BIC
print(order_df.sort_values('bic'))

# From the results, we will choose p=1 ,q = 2,P = 1,Q = 1 as they have the lowest aic and bic scores 
# Final model
model_final = SARIMAX(store1.iloc[:,0], order=(1,1,2), seasonal_order=(1,1,1,7) , exog = store1.iloc[:,1:])
results = model_final.fit()

# Results diagnostics
results.plot_diagnostics()
plt.show()
# Results summary
print(results.summary())
predictions= results.predict(start =train_size, end=train_size+test_size-1,exog= validation_store1.iloc[:,1:], dynamic = True)

predictions=pd.DataFrame(predictions)
predictions.reset_index(inplace=True)
predictions.index = validation_store1.index
predictions['Actual'] = validation_store1['Sales']
predictions.rename(columns={0:'Pred'}, inplace=True)
predictions['Actual'].plot(figsize=(20,8), legend=True, color='blue')
predictions['Pred'].plot(legend=True, color='red', figsize=(20,8))
# rmse = np.sqrt(mean_squared_error(predictions.Actual, predictions.Pred))
# print("RMSE :" + str(rmse))

new  = predictions[predictions['Actual']!=0]
new.head()

loss = np.sqrt(np.mean(np.square(((new.Actual - new.Pred) / new.Actual)), axis=0))
print("RMSPE :" + str(loss))
    
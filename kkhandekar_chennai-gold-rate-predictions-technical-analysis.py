!pip install pmdarima -q
# Generic

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import gc, warnings, re

warnings.filterwarnings("ignore")



# Plotting

import matplotlib.pyplot as plt

import plotly.express as px



from tabulate import tabulate



# Sklearn

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn import metrics as mt



from sklearn.linear_model import LinearRegression

from sklearn import neighbors

from sklearn.model_selection import GridSearchCV



# Arima Model

import pmdarima as pm



# Prophet

from fbprophet import Prophet



# Keras

from keras.models import Sequential

from keras.layers import Dense, Dropout, LSTM
# Data Load

url = '../input/gold-rate-history-in-tamilnadu-india/gold_rate_history.csv'

df = pd.read_csv(url,index_col='Date', header='infer', parse_dates=True, infer_datetime_format=True)



# Dropping Unwanted Columns

unwanted_cols = ['Country','State','Location']

df.drop(unwanted_cols, axis=1, inplace=True)



# Renaming Columns

df.rename(columns={"Pure Gold (24 k)": "Pure_Gold_24k",

                   "Standard Gold (22 K)": "Std_Gold_22k",

                  },inplace=True)



# Total Records

print("Total Records: ", df.shape[0])



# Inspect

df.head()
# Scaling Data

cols = df.columns

idx = df.index

scaler = MinMaxScaler(feature_range=(0,1))



df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=cols, index=idx)
# Visualisation



fig = px.line(df, x=df.index, y=df.columns,

              title='Gold Prices in Chennai (2006-2020)')



fig.update_xaxes(tickangle=45)

fig.show()
fig = px.scatter(df, x=df.index, y=df.columns, marginal_x="box", marginal_y="violin", title="Chennai Gold Rate Marginal Distribution Plot")

fig.show()
# Creating a seperate dataframe

df_lr = df_scaled.copy()



# Converting Date Index to Column for Feature Extraction

df_lr.reset_index(level=0, inplace=True)



# Time Feature Extraction

df_lr['year']=df_lr['Date'].dt.year 

df_lr['month']=df_lr['Date'].dt.month 

df_lr['day']=df_lr['Date'].dt.day

df_lr['quarter']=df_lr['Date'].dt.quarter

df_lr['weekofyear']=df_lr['Date'].dt.weekofyear

df_lr['weekday']=df_lr['Date'].dt.weekday



# Dropping Date Column

df_lr.drop('Date',axis=1,inplace=True)
# Feature Engineering & Split

features = ['year','month','day','quarter','weekofyear','weekday']



target_24k = ['Pure_Gold_24k']

target_22k = ['Std_Gold_22k']



X = df_lr[features]

y_24k = df_lr[target_24k]

y_22k = df_lr[target_22k]



size = 0.1  #validation size



X_train_24k, X_val_24k, y_train_24k, y_val_24k = train_test_split(X, y_24k, test_size=size, random_state=42)

X_train_22k, X_val_22k, y_train_22k, y_val_22k = train_test_split(X, y_22k, test_size=size, random_state=42)

tab_data = []



# Modeling, Training & Prediction

lr_model = LinearRegression()



'''24k Gold'''

lr_model.fit(X_train_24k,y_train_24k)

y_pred_24k = lr_model.predict(X_val_24k)



# Evaluate

rmse = mt.mean_squared_error(y_val_24k,y_pred_24k)

r2_score = mt.r2_score(y_val_24k,y_pred_24k)



tab_data.append(['PureGold_24k','{:.2}'.format(rmse), '{:.2}'.format(r2_score)])



'''22k Gold'''

lr_model.fit(X_train_22k,y_train_22k)

y_pred_22k = lr_model.predict(X_val_22k)



# Evaluate

rmse = mt.mean_squared_error(y_val_22k,y_pred_22k)

r2_score = mt.r2_score(y_val_22k,y_pred_22k)



tab_data.append(['StdGold_22k','{:.2}'.format(rmse), '{:.2}'.format(r2_score)])



print(tabulate(tab_data, headers=['RMSE','R2_Score'], tablefmt="pretty"))
# Free Memory

gc.collect()

tab_data.clear()
#using gridsearch to find the best parameter



params = {'n_neighbors':[2,3,4,5,6,7,8,9]}



knn = neighbors.KNeighborsRegressor()

knn_model = GridSearchCV(knn, params, cv=5)



'''24k Gold'''

knn_model.fit(X_train_24k,y_train_24k)

y_pred_24k = knn_model.predict(X_val_24k)



# Evaluate

rmse = mt.mean_squared_error(y_val_24k,y_pred_24k)

r2_score = mt.r2_score(y_val_24k,y_pred_24k)



tab_data.append(['PureGold_24k','{:.2}'.format(rmse), '{:.2}'.format(r2_score)])





'''22k Gold'''

knn_model.fit(X_train_22k,y_train_22k)

y_pred_22k = knn_model.predict(X_val_22k)



# Evaluate

rmse = mt.mean_squared_error(y_val_22k,y_pred_22k)

r2_score = mt.r2_score(y_val_22k,y_pred_22k)



tab_data.append(['StdGold_22k','{:.2}'.format(rmse), '{:.2}'.format(r2_score)])



print(tabulate(tab_data, headers=['RMSE','R2_Score'], tablefmt="pretty"))

# Free Memory

gc.collect()

tab_data.clear()
df_ar = df.copy()



# Split

tr_pct = int(0.9 * len(df_ar))  # 90% for training



train_df = df_ar[:tr_pct]

val_df = df_ar[tr_pct:]



'''Pure Gold 24k'''

tr_24k = train_df['Pure_Gold_24k']

val_24k = val_df['Pure_Gold_24k']



'''Std Gold 22k'''

tr_22k = train_df['Std_Gold_22k']

val_22k = val_df['Std_Gold_22k']
'''Forecasting'''



# Fit & Forecasting (Pure Gold 24k)

arima_model = pm.auto_arima(tr_24k, seasonal=True, m=12)

forecast_24k = arima_model.predict(val_24k.shape[0])



# Fit & Forecasting (Pure Gold 22k)

arima_model = pm.auto_arima(tr_22k, seasonal=True, m=12)

forecast_22k = arima_model.predict(val_22k.shape[0])

# Dataframe

forecast24 = pd.DataFrame(forecast_24k, index=val_df.index,columns=['Forecast24k'])

forecast22 = pd.DataFrame(forecast_22k, index=val_df.index,columns=['Forecast22k'])



# Merge Dataframe

forecast = pd.merge(forecast22,forecast24, left_index=True, right_index=True)



# Inspect

forecast.head()
fig = px.line(forecast, x=forecast.index, y=forecast.columns, title='Chennai Gold Rate Forecasting (ARIMA)')

fig.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

fig.suptitle('Chennai Gold Rate Forecast (ARIMA)', fontsize=20)



ax1.plot(train_df['Pure_Gold_24k'])

ax1.plot(val_df['Pure_Gold_24k'])

ax1.plot(forecast['Forecast24k'])

ax1.set_title("Pure Gold 24k", fontsize=10)



ax2.plot(train_df['Std_Gold_22k'])

ax2.plot(val_df['Std_Gold_22k'])

ax2.plot(forecast['Forecast22k'])

ax2.set_title("Std Gold 22k", fontsize=10)
# Free Memory

gc.collect()
df_pr.head()
# Creating a copy

df_pr = df.copy()



# Converting Date Index to Column for Feature Extraction

df_pr.reset_index(level=0, inplace=True)



# Splitting the dataframe into two

df_pr_24k = df_pr[['Date','Pure_Gold_24k']]

df_pr_22k = df_pr[['Date','Std_Gold_22k']]



# Renaming Columns

df_pr_24k.rename(columns={'Date':'ds','Pure_Gold_24k':'y'},inplace=True)

df_pr_22k.rename(columns={'Date':'ds','Std_Gold_22k':'y'},inplace=True)
'''Pure Gold 24k dataset'''



# Split

tr_pct = int(0.9 * len(df_pr))  # 90% for training



train_df_24k = df_pr_24k[:tr_pct]

val_df_24k = df_pr_24k[tr_pct:]



'''Std Gold 22k dataset'''



# Split



train_df_22k = df_pr_22k[:tr_pct]

val_df_22k = df_pr_22k[tr_pct:]
# Instantiating Model

proph = Prophet()
'''Forecasting'''



# Pure Gold 24k

proph.fit(train_df_24k)



gold24k_price = proph.make_future_dataframe(periods=len(val_df_24k))

forecast24 = proph.predict(gold24k_price)

# Re-Instantiating Model

proph = Prophet()
'''Forecasting'''



# Std Gold 22k

proph.fit(train_df_22k)



gold22k_price = proph.make_future_dataframe(periods=len(val_df_22k))

forecast22 = proph.predict(gold22k_price)
# Adding Predictions to the Validation Dataset

val_df_24k['Predictions'] = 0

val_df_24k['Predictions'] = forecast24['yhat']



val_df_22k['Predictions'] = 0

val_df_22k['Predictions'] = forecast22['yhat']
# Plot

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))

fig.suptitle('Chennai Gold Rate Forecast (Prophet)', fontsize=20)



ax1.plot(train_df_24k['y'])

ax1.plot(val_df_24k[['y', 'Predictions']])

ax1.set_title("Pure Gold 24k", fontsize=10)



ax2.plot(train_df_22k['y'])

ax2.plot(val_df_22k[['y', 'Predictions']])

ax2.set_title("Std Gold 22k", fontsize=10)
# Free Memory

gc.collect()
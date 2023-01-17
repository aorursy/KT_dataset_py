# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip install sktime
!pip install PyAstronomy
df_orders = pd.read_csv("/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv")
df_order_payments = pd.read_csv("/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv")
df_order_review = pd.read_csv("/kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv")
df_order_itens = pd.read_csv("/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv")

df_selers = pd.read_csv("/kaggle/input/brazilian-ecommerce/olist_sellers_dataset.csv")

df_custumers = pd.read_csv("/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv")

df_products = pd.read_csv("/kaggle/input/brazilian-ecommerce/olist_products_dataset.csv")

df_geolocation = pd.read_csv("/kaggle/input/brazilian-ecommerce/olist_geolocation_dataset.csv")

df_category_name = pd.read_csv("/kaggle/input/brazilian-ecommerce/product_category_name_translation.csv")
df_order_itens.head()
df_products.head()
df_orders.head()
df_product_order = df_products.merge(df_order_itens, how="left")
df_product_order
df_product_order = df_product_order.merge(df_orders, how='right')
df_product_order
df_pd_converted = df_product_order[df_product_order.order_status != 'canceled']
df_pd_converted
df_product_order_filtered = df_pd_converted[['product_id','product_category_name','order_item_id','price']]
df_product_order_filtered
df_category = df_product_order_filtered.groupby(['product_category_name']).sum()
df_category.sort_values('price',ascending=False)
import plotly.express as px
fig = px.line(df_category, y='order_item_id')
fig.show()
df_pdt_pred_filtered = df_pd_converted[['product_id','price','order_purchase_timestamp']]
df_pdt_pred_filtered
from datetime import datetime as dt
df_pdt_pred_filtered["order_purchase_timestamp"] = df_pdt_pred_filtered["order_purchase_timestamp"].apply(lambda d: (dt.strptime(d, '%Y-%m-%d %H:%M:%S')))
#df_pdt_pred_filtered["time"] = df_pdt_pred_filtered["order_purchase_timestamp"].apply(lambda d: (str(dt.date(d).year) + '-' + str(dt.date(d).month)))
df_pdt_pred_filtered["time"] = df_pdt_pred_filtered["order_purchase_timestamp"].apply(lambda d: dt(year=d.year, month=d.month, day=d.day))
df_pdt_pred_filtered.drop(columns='order_purchase_timestamp', inplace = True)
df_pdt_pred_filtered
df_pdt_pred_filtered.sort_values('time')
df_pdt_count = df_pdt_pred_filtered.groupby('product_id').count()
df_pdt_count.rename(columns={'price': 'count'}, inplace = True)
df_pdt_count.sort_values(by='count', inplace=True, ascending=False)
df_pdt_count.head()
#df_pdt_pred_group = df_pdt_pred_filtered.groupby([pd.Grouper(key='time', freq='W-Mon'), 'product_id']).count()
df_pdt_pred_group = df_pdt_pred_filtered.groupby(['time', 'product_id']).count()
#df_pdt_pred_group = df_pdt_pred_filtered.groupby([pd.Grouper(key='time', freq='M'), 'product_id']).count()
df_pdt_pred_group.sort_values(by='time', inplace=True)
df_pdt_pred_group.rename(columns={'price': 'count'}, inplace = True)
df_pdt_pred_group
df_pdt_pred_group.reset_index(inplace=True)
df_pdt_1 = df_pdt_pred_group.where(df_pdt_pred_group['product_id']=='aca2eb7d00ea1a7b8ebd4e68314663af').dropna()
df_pdt_1
fig = px.line(df_pdt_1, y='count', x='time')
fig.show()
X = df_pdt_1.iloc[:, 0].values
Y = df_pdt_1.iloc[:, 2]
X = X.reshape(-1, 1)
X.shape, Y.shape
from sktime.forecasting.model_selection import temporal_train_test_split

x_train, x_test, y_train, y_test = temporal_train_test_split(X, Y, test_size=0.1)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
x_test
import xgboost as xgb

param_dist = {'objective':'reg:squarederror', 'n_estimators':150, 'max_depth':15, 'num_parellel_tree':10}
model = xgb.XGBRegressor(**param_dist)

model.fit(x_train, y_train)
import plotly.graph_objects as go
time = [k for k in range(y_test.shape[0])]

fig = go.Figure()

fig.add_trace(go.Scatter(
    name="Real data",
    x=time, 
    y=y_test))

fig.add_trace(go.Scatter(
    name="Prediction model",
    x=time, 
    y=model.predict(x_test)))

fig.update_layout(title_text="Product count prediction")
fig.update_xaxes(title_text='Time')
fig.update_yaxes(title_text='Count')

fig.show()
df_pdt_1.shape
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional
from tensorflow.keras.optimizers import SGD
import math
#Separando treino de teste
n_test = 40
training_set = df_pdt_1.iloc[:(df_pdt_1.shape[0] - n_test),2].values
test_set = df_pdt_1.iloc[(df_pdt_1.shape[0] - n_test):,2].values
#Normalizar os dados
sc = MinMaxScaler(feature_range=(0,1))
training_set = training_set.reshape(-1,1)
test_set = test_set.reshape(-1,1)
training_set_norm = training_set
test_set_norm = test_set
#training_set_norm = sc.fit_transform(training_set)
#test_set_norm = sc.fit_transform(test_set)
training_set_norm.shape
X_train = []
y_train = []

n = 4

for i in range (n, training_set_norm.shape[0]):
  X_train.append( training_set[ i-n:i, 0 ] )
  y_train.append( training_set[ i, 0 ])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train.shape
X_train_3D = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
y_train.shape
# The LSTM architecture
regressor = Sequential()
# First LSTM layer with Dropout regularisation
regressor.add(LSTM(units=10, return_sequences=True, input_shape=(X_train_3D.shape[1], 1)))
regressor.add(Dropout(0.25))
# Second LSTM layer
regressor.add(LSTM(units=10, return_sequences=True))
regressor.add(Dropout(0.25))
# Third LSTM layer
regressor.add(LSTM(units=10, return_sequences=True))
regressor.add(Dropout(0.25))
# Fourth LSTM layer
regressor.add(LSTM(units=4))
regressor.add(Dropout(0.25))
# The output layer
regressor.add(Dense(units=1))
# Compiling the RNN
regressor.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['mse','mae'])
# Fitting to the training set
historico = regressor.fit(X_train_3D,y_train,epochs=50,batch_size=5)
import matplotlib.pyplot as plt
plt.plot(historico.history['mse'])
plt.plot(historico.history['mae'])
#Preparando X de teste para prever as ações
X_test = []
for i in range(n,test_set_norm.shape[0]):
    X_test.append( test_set[ i-n:i, 0 ] )
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
predicted = regressor.predict(X_test)
predicted[:10]
plt.plot(predicted, color='blue',label='Predicted IBM Stock Price')
plt.plot(test_set_norm, color='red',label='Real IBM Stock Price')
plt.plot(test_set_norm, color='red',label='Real IBM Stock Price')
plt.plot(predicted, color='blue',label='Predicted IBM Stock Price')
plt.title('IBM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('IBM Stock Price')
plt.legend()
plt.show()
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import gc
import matplotlib.pyplot as plt
sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
val = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
df = df[['date','item_id','shop_id','item_cnt_day']]
df["item_cnt_day"].clip(0.,20.,inplace=True)
df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()

test = pd.merge(val,df,on=['item_id','shop_id'], how='left').fillna(0)

test = test.drop(labels=['ID','item_id','shop_id'],axis=1)
scaler = MinMaxScaler(feature_range=(0, 1))
sales["item_price"] = scaler.fit_transform(sales["item_price"].values.reshape(-1,1))
df2 = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).mean().reset_index()
df2 = df2[['date','item_id','shop_id','item_price']].pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()

price = pd.merge(val,df2,on=['item_id','shop_id'], how='left').fillna(0)
price = price.drop(labels=['ID','item_id','shop_id'],axis=1)
y_train = test['2015-10']
x_sales = test.drop(labels=['2015-10'],axis=1)
x_sales = x_sales.values.reshape((x_sales.shape[0], x_sales.shape[1], 1))
x_prices = price.drop(labels=['2015-10'],axis=1)
x_prices= x_prices.values.reshape((x_prices.shape[0], x_prices.shape[1], 1))
X = np.append(x_sales,x_prices,axis=2)

y = y_train.values.reshape((214200, 1))
del y_train, x_sales; gc.collect()

test = test.drop(labels=['2013-01'],axis=1)
x_test_sales = test.values.reshape((test.shape[0], test.shape[1], 1))
x_test_prices = price.drop(labels=['2013-01'],axis=1)
x_test_prices = x_test_prices.values.reshape((x_test_prices.shape[0], x_test_prices.shape[1], 1))
test = np.append(x_test_sales,x_test_prices,axis=2)
del x_test_sales,x_test_prices, price; gc.collect()

model_lstm = Sequential()
model_lstm.add(LSTM(32, input_shape=(X.shape[1], X.shape[2]),return_sequences=True))
model_lstm.add(Dropout(0.5))
model_lstm.add(LSTM(32))
model_lstm.add(Dropout(0.5))
model_lstm.add(Dense(1))
model_lstm.compile(optimizer="adam", loss='mse', metrics=["mse"])
VALID = True
LSTM_PARAM = {"batch_size":128,
              "verbose":2,
              "epochs":10}

if VALID is True:
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=1, shuffle=False)
    
    callbacks_list=[EarlyStopping(monitor="val_loss",min_delta=.001, patience=3,mode='auto')]
    hist = model_lstm.fit(X_train, y_train,
                          validation_data=(X_valid, y_valid),
                          callbacks=callbacks_list,
                          **LSTM_PARAM)
    pred = model_lstm.predict(test)
    
    best = np.argmin(hist.history["val_loss"])

if VALID is False:

    hist = model_lstm.fit(X,y,**LSTM_PARAM)
    pred = model_lstm.predict(X)
    
print("\Output Submission")
submission = pd.DataFrame(pred,columns=['item_cnt_month'])
submission.to_csv('submission.csv',index_label='ID')
print(submission.head())
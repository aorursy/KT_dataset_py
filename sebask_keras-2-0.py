# Python 3 libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, ThresholdedReLU, MaxPooling2D, Embedding, Dropout
from keras.optimizers import Adam, SGD, RMSprop
from keras import backend as K
from keras.wrappers.scikit_learn import KerasRegressor

# Import data
sales = pd.read_csv('../input/sales_train.csv', parse_dates=['date'], infer_datetime_format=True, dayfirst=True)
shops = pd.read_csv('../input/shops.csv')
items = pd.read_csv('../input/items.csv')
cats = pd.read_csv('../input/item_categories.csv')
val = pd.read_csv('../input/test.csv')

# Rearrange the raw data to be monthly sales by item-shop
df = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()
df = df[['date','item_id','shop_id','item_cnt_day']]
df = df.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()

# Merge data from monthly sales to specific item-shops in test data
test = pd.merge(val,df,on=['item_id','shop_id'], how='left')
test = test.fillna(0)
display(test.head(3))

# Strip categorical data so keras only sees raw timeseries
test = test.drop(labels=['ID','item_id','shop_id'],axis=1)

# Rearrange the raw data to be monthly average price by item-shop
df2 = sales.groupby([sales.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).mean().reset_index()
df2 = df2[['date','item_id','shop_id','item_price']]
df2 = df2.pivot_table(index=['item_id','shop_id'], columns='date',values='item_price',fill_value=0).reset_index()

# Merge data from average prices to specific item-shops in test data
price = pd.merge(val,df2,on=['item_id','shop_id'], how='left')
price = price.fillna(0)
display(price.head(3))
price = price.drop(labels=['ID','item_id','shop_id'],axis=1)
# Create x and y training sets from oldest data points
y_train = test['2015-10']
x_sales = test.drop(labels=['2015-10'],axis=1)
x_prices = price.drop(labels=['2015-10'],axis=1)

# Turn training sets into numpy matrix
x_sales = x_sales.as_matrix()
x_sales = x_sales.reshape((214200, 33, 1))
x_prices = x_prices.as_matrix()
x_prices = x_prices.reshape((214200, 33, 1))
#x_train = np.append(x_sales,x_prices,axis=2)
x_train = x_sales
display(x_train.shape)

y_train = y_train.as_matrix()
y_train = y_train.reshape((214200, 1))

# Transform test set into numpy matrix
x_test_sales = test.drop(labels=['2013-01'],axis=1)
x_test_sales = x_test_sales.as_matrix()
x_test_sales = x_test_sales.reshape((214200, 33, 1))
x_test_prices = price.drop(labels=['2013-01'],axis=1)
x_test_prices = x_test_prices.as_matrix()
x_test_prices = x_test_prices.reshape((214200, 33, 1))
#x_test = np.append(x_test_sales,x_test_prices,axis=2)
x_test = x_test_sales
display(x_test.shape)
# Define the model layers
model_lstm = Sequential()
model_lstm.add(LSTM(100, input_shape=(33, 1)))
model_lstm.add(Dropout(0.8))
model_lstm.add(Dense(1))
algorithm = RMSprop(lr=0.1)
def root_mse(y_true, y_pred):
        return np.abs(K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)))
model_lstm.compile(optimizer=algorithm, loss='mse', metrics=['accuracy'])
hist = model_lstm.fit(x_train, y_train, epochs=3)
y_pre = model_lstm.predict(x_test)
preds = pd.DataFrame(y_pre,columns=['item_cnt_month'])
#preds.round()
display(preds.head())
display(preds.describe())
display(test['2015-10'].describe())
preds.to_csv('submission.csv',index_label='ID')

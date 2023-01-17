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
import datetime
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import catboost
from catboost import Pool
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

%matplotlib inline
sns.set(style="darkgrid")
pd.set_option('display.float_format', lambda x: '%.2f' % x)
warnings.filterwarnings("ignore")
df_sample = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
df_sample[0:5]
df_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
df_train[0:5]
df_train.shape
df_shop =pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
df_shop[0:5]
df_item = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
df_item[0:5]
df_test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
df_test.tail()
df_test[0:5]
df_item_categories =pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
df_item_categories[0:5]
df_shop.isnull().values.any()
df_train = df_train.merge(df_shop,on = ['shop_id'])
df_train[0:5]
df_shop['shop_name'][df_shop['shop_id'] == 59]
df_train = df_train.merge(df_item,on = ['item_id'])
df_train[0:5]
df_item[df_item['item_id'] == 22154]
df_train = df_train.merge(df_item_categories,on =['item_category_id'])
df_train[0:5]
df_item_categories[df_item_categories['item_category_id'] == 37]
df_train.isnull().values.any()
df_train.describe()
df_train.date = df_train.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
df_train['date'].describe()
df_train['date'].min().date()
df_train['date'].max().date()
# the skewed box plot shows us the presence of outliers
%matplotlib notebook
%matplotlib inline
sns.boxplot(y="item_price", data =df_train)
plt.show()
#calculating 0-100th percentile to find a the correct percentile value for removal of outliers
for i in range(0,100,10):
    var = df_train["item_price"].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))
print ("100 percentile value is ",var[-1])
#looking further from the 99th percecntile
for i in range(90,100):
    var = df_train["item_price"].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))
print ("100 percentile value is ",var[-1])
#calculating speed values at each percntile 99.0,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100
for i in np.arange(0.0, 1.0, 0.1):
    var =df_train["item_price"].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(99+i,var[int(len(var)*(float(99+i)/100))]))
print("100 percentile value is ",var[-1])
plt.plot(var[-5:])
plt.show()
plt.plot(var[-2:])
plt.show()
# the skewed box plot shows us the presence of outliers
%matplotlib notebook
%matplotlib inline
sns.boxplot(y="item_cnt_day", data =df_train)
plt.show()
#calculating 0-100th percentile to find a the correct percentile value for removal of outliers
for i in range(0,100,10):
    var = df_train["item_cnt_day"].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))
print ("100 percentile value is ",var[-1])
#looking further from the 99th percecntile
for i in range(90,100):
    var = df_train["item_cnt_day"].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(i,var[int(len(var)*(float(i)/100))]))
print ("100 percentile value is ",var[-1])
#calculating speed values at each percntile 99.0,99.1,99.2,99.3,99.4,99.5,99.6,99.7,99.8,99.9,100
for i in np.arange(0.0, 1.0, 0.1):
    var =df_train["item_cnt_day"].values
    var = np.sort(var,axis = None)
    print("{} percentile value is {}".format(99+i,var[int(len(var)*(float(99+i)/100))]))
print("100 percentile value is ",var[-1])
plt.plot(var[-20:])
plt.show()
plt.plot(var[-3:])
plt.show()
df_train = df_train[df_train['item_price']<50000]
df_train.shape
df_train = df_train[df_train['item_cnt_day']<750]
df_train = df_train[df_train['item_cnt_day']>=0]
df_train.shape

print('Data set size before filter valid:', df_train.shape)
# Only shops that exist in test set.
df_train = df_train[df_train['shop_id'].isin(df_test['shop_id'].unique())]
# Only items that exist in test set.
df_train = df_train[df_train['item_id'].isin(df_test['item_id'].unique())]
print('Data set size after filter valid:', df_train.shape)
df_train[0:5]
# Aggregate to monthly level the sales
df_train_groupby= df_train.groupby(["date_block_num","shop_id","item_id"])[
    "item_price","item_cnt_day"].agg({"item_price":"mean","item_cnt_day":"sum"})
df_train_groupby = df_train_groupby.reset_index()
df_train_groupby[0:5]
df_train_groupby.shape
a =  list(df_train_groupby.item_cnt_day)
a
df_test[0:5]
df_test.shape
final_data = pd.merge(df_test,df_train_groupby,on = ['item_id','shop_id'],how = 'left')
final_data[0:5]
final_data.shape
final_data.isnull().values.any()
final_data.fillna(0,inplace = True)
final_data.isnull().values.any()
final_data.drop(['shop_id','item_id'],inplace = True, axis = 1)

final_data[0:5]
#We will create pivot table.
# Rows = each shop+item code
# Columns will be out time sequence
pivot_data = final_data.pivot_table(index='ID',values = 'item_cnt_day' ,columns='date_block_num',fill_value = 0 )
pivot_data.head()
row = 34
pivot_data.values[0:1][0][row-3-1: row -1]
sum(a[25:])
a  =(pivot_data[0:1].values)[0]
def simple_average(pivot_data,row_no,col_no,window_size):
    predicted_values = []
    for i in range(row_no):
        temp = pivot_data.values[i:i+1][0]
        #print(temp[col_no-window_size-1:col_no-1])
        predict_value = int(sum(temp[col_no-window_size-1:col_no-1])/window_size)
        #print(predict_value)
        predicted_values.append(predict_value)
        
    return predicted_values
def actual_values(pivot_data):
    actual = []
    temp = pivot_data.shape[0]
    for i in range(temp):
        temp = pivot_data.values[i:i+1][0]
        actual.extend([int(temp[-1:])])
    return actual
actual = actual_values(pivot_data)
def rms_error(predicted,actual):
    final =   list(np.array(actual) - np.array(store))
    mse = sum([e**2 for e in final])/len(final)
    return mse
# here we have predicted the october item_cnt using the previous three months
error_values = []
window_lenght = 6
for i in range(2,window_lenght):
    store = simple_average(pivot_data,pivot_data.shape[0],pivot_data.shape[1],i)
    error = rms_error(store,actual)
    error_values.extend([error])
# here we have use window lenght from 2 to 5
print(error_values)
def predict_nov_simple_average(pivot_data,row_no,col_no,window_size):
    predicted_values = []
    for i in range(row_no):
        temp = pivot_data.values[i:i+1][0]
        #print(temp[col_no-window_size:col_no])
        predict_value = int(sum(temp[col_no-window_size:col_no])/window_size)
        #print(predict_value)
        predicted_values.append(predict_value)
        
    return predicted_values
# we are getting the best result from window_size of 3 so we will use window_size of 3 to predict value for november data
predicted_nov_savg = predict_nov_simple_average(pivot_data,pivot_data.shape[0],pivot_data.shape[1],3)
len(predicted_nov_savg)
predicted_nov_savg
def weighted_average(pivot_data,row_no,col_no,window_size):
    predicted_values = []
    for i in range(row_no):
        temp = pivot_data.values[i:i+1][0]
        #print(temp[col_no-window_size-1:col_no-1])
        temp = temp[col_no-window_size-1:col_no-1]
        store = 0
        for j in range(1,window_size+1):
            store = store + j*temp[j-1]
        predict_value = int(store/((window_size*(window_size+1))/2))
        #print(predict_value)
        predicted_values.append(predict_value)
        
    return predicted_values
# here we have predicted the october item_cnt using the previous three months
error_values = []
window_lenght = 6
for i in range(2,window_lenght):
    store = weighted_average(pivot_data,pivot_data.shape[0],pivot_data.shape[1],i)
    error = rms_error(store,actual)
    error_values.extend([error])
error_values
#here the window size of 4 gives the best result so we will use window size of 4 to predict the item_cnt for month november
def predict_nov_weighted_average(pivot_data,row_no,col_no,window_size):
    predicted_values = []
    for i in range(row_no):
        temp = pivot_data.values[i:i+1][0]
        temp = temp[col_no-window_size:col_no]
        #print(temp[col_no-window_size:col_no])
        store =0
        for j in range(1,window_size+1):
            store = store + j*temp[j-1]
        predict_value = int(store/((window_size*(window_size+1))/2))
        #print(predict_value)
        predicted_values.append(predict_value)
        
    return predicted_values
predicted_nov_weighted_average = predict_nov_weighted_average(pivot_data,pivot_data.shape[0],pivot_data.shape[1],4)
len(predicted_nov_weighted_average)
predicted_nov_weighted_average
def exponential_average(pivot_data,row_no,col_no,alpha):
    predicted_values = []
    for i in range(row_no):
        temp = pivot_data.values[i:i+1][0]
        #print(temp[:-1])
        temp = temp[:-1]
        store = 0
        for j in range(len(temp)):
            store = store*(1-alpha) + alpha*temp[j]
        #print(predict_value)
        store = int(store)
        predicted_values.append(store)
        
    return predicted_values
# here we have predicted the october item_cnt using the previous three months
error_values = []
window_lenght = [0.3,0.4,0.5,0.6,0.7]
for i in window_lenght:
    store = exponential_average(pivot_data,pivot_data.shape[0],pivot_data.shape[1],i)
    error = rms_error(store,actual)
    error_values.extend([error])
error_values
def exponential_average_nov(pivot_data,row_no,col_no,alpha):
    predicted_values = []
    for i in range(row_no):
        temp = pivot_data.values[i:i+1][0]
        #print(temp[:-1])
        #temp = temp[:-1]
        store = 0
        for j in range(len(temp)):
            store = store*(1-alpha) + alpha*temp[j]
        #print(predict_value)
        store = int(store)
        predicted_values.append(store)
        
    return predicted_values
predicted_nov_exponential_average = exponential_average_nov(pivot_data,pivot_data.shape[0],pivot_data.shape[1],0.4)
len(predicted_nov_exponential_average)
# one thing to consider while autoregressive model is that the value of lag means how many previous values you are going to use
# we can plot autocorelation to find out the value of lag.
# here we are plotting the curve for id 0
fig, axs = plt.subplots(2, 2)
axs[0,0].plot(pivot_data.values[0:1][0])
axs[0,1].plot(pivot_data.values[1:2][0])

axs[1,0].plot(pivot_data.values[2:3][0])
axs[1,1].plot(pivot_data.values[3:4][0])

#plt.plot(series)
plt.show()
from statsmodels.graphics.tsaplots import plot_acf
#fig, axs = plt.subplots(2, 2)

plot_acf(pivot_data.values[0:1][0],lags = 33)
plot_acf(pivot_data.values[1:2][0],lags = 33)
plot_acf(pivot_data.values[2:3][0],lags = 33)
plot_acf(pivot_data.values[3:4][0],lags = 33)

plt.show()
def auto_regressive(pivot_data,row_no,lag):
    predicted_values = []
    for i in range(row_no):
        data = pivot_data.values[i:i+1][0]
        #print(data[:-1])
        model = AutoReg(data[:-1],lags = lag).fit()
        #print(model.params)
        predict_value = model.predict(start=len(data)-1,end=len(data)-1)
        #print(predict_value)
        predicted_values.append(int(predict_value))  
        
    return predicted_values
'''def auto_regressive_2(pivot_data,row_no,lag):
    predicted_values = []
    for i in range(row_no):
        data = pivot_data.values[i:i+1][0]
        data = data[:-1]
        window = len(data)
        #print(window)
        model = AutoReg(data[:-1],lags = lag).fit()
        parameters = model.params
        yhat = parameters[0]
        for j in range(lag):
            yhat += parameters[j+1] * data[window-j-1]
        #print(model.params)
        #print(yhat)
        #predict_value = model.predict(start=len(data)-1,end=len(data)-1)
        #print(predict_value)
        predicted_values.append(int(yhat))  
        
    return predicted_values'''
'''store = auto_regressive_2(pivot_data,pivot_data.shape[0],4)'''
'''print(rms_error(store,actual))'''
# here we have multiple time series for each id 
# we will fit model on series one by one 
# and then check oveall mean squared error
from statsmodels.tsa.ar_model import AutoReg

error_values = []
lag = [2,3,4,5]
for i in lag:
    store = auto_regressive(pivot_data,pivot_data.shape[0],i)
    error = rms_error(store,actual)
    error_values.extend([error])

# the result are not good while using auto regressive model
error_values
# X we will keep all columns execpt the last one 
X_train = np.expand_dims(pivot_data.values[:,:-1],axis = 2)
# the last column is our prediction
y_train = pivot_data.values[:,-1:]
X_train.shape
y_train.shape
# for test we keep all the columns execpt the first one
X_test = np.expand_dims(pivot_data.values[:,1:],axis = 2)
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout
from keras.models import load_model, Model

# our defining sales model 
model = Sequential()
model.add(LSTM(units = 64,input_shape = (33,1)))
#sales_model.add(LSTM(units = 64,activation='relu'))
model.add(Dense(128))
model.add(Dropout(0.3))
model.add(Dense(64))
model.add(Dropout(0.2))

model.add(Dense(1))

model.summary()
model.compile(loss = 'mse',optimizer = 'adam', metrics = ['mean_squared_error'])
model.fit(X_train,y_train,batch_size = 300,epochs = 10)
output_november_sales = model.predict(X_test)
output_november_sales[0:10]

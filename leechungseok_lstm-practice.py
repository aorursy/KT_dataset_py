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
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
def predict_stock(stock, column, period, timestep):
  stock = stock
  column = column
  period = period
  timestep = timestep
  
  select_stock(stock, period, timestep)
  prepro(df, column, timestep)
  result = pred_stock(period, timestep)
  return result
# 필요한 변수 = ['High', 'Low', 'Close']
# 종목 선택 & 앞으로 예측할 기간 선택

def select_stock(stock, period, timestep):
    df = pdr.get_data_yahoo(stock, '2000-01-01')
    tempdf = df.copy()[['High', 'Low', 'Close']]
    #타겟 설정
    tempdf['Target_YMD'] = 0
    tempdf['Target_High'] = 0
    tempdf['Target_Low'] = 0
    tempdf['Target_Close'] = 0
    for i in range(len(tempdf)):
        try:
            tempdf['Target_YMD'].iloc[i] = str(tempdf.index[i+period]).split(' ')[0]
            tempdf['Target_High'].iloc[i] = tempdf['High'].iloc[i+period]
            tempdf['Target_Low'].iloc[i] = tempdf['Low'].iloc[i+period]
            tempdf['Target_Close'].iloc[i] = tempdf['Close'].iloc[i+period]
        except:
            continue

    tempdf_index = []
    for i in range(len(tempdf)):
        indexword =  str(tempdf.index[i]).split(' ')[0]
        tempdf_index.append(indexword)

    tempdf.index = tempdf_index
    global future_test
    future_test = tempdf[-period-(timestep-1):][['High', 'Low', 'Close']] # 3은 shift 횟수
    tempdf = tempdf[:-period]
    
    return tempdf
# df = select_stock(stock_code, period) 
def prepro(df, column, timestep):
    x_data = df[['High', 'Low', 'Close']]
    #x_data log&MinMaxScale
    x_data = np.log(x_data)
    global sc_x
    sc_x = MinMaxScaler()
    x_data = pd.DataFrame(sc_x.fit_transform(x_data), index=x_data.index, columns=x_data.columns)
    #Time step 결정
    lst = []
    lst.append('High')
    for step in range(1,timestep):
        x_data['High_{}'.format(step)] = x_data['High'].shift(step)
        lst.append('High_'+str(step))
    lst.append('Low')
    for step in range(1,timestep):
        x_data['Low_{}'.format(step)] = x_data['Low'].shift(step)
        lst.append('Low_'+str(step))
    lst.append('Close')
    for step in range(1,timestep):
        x_data['Close_{}'.format(step)] = x_data['Close'].shift(step)
        lst.append('Close_'+str(step))
    x_data = x_data.dropna()
    x_data = x_data[lst]
    
    #y_data
    #y columns 선택변수로 만들 것 
    target_word = 'Target_'+str(column)
    y_data = df[target_word][(timestep-1):] # Timestep의 갯수만큼 
    y_ymd = df['Target_YMD'][(timestep-1):]
    y_data = np.log(y_data)
    global sc_y 
    sc_y = MinMaxScaler()
    y_data = sc_y.fit_transform(np.array(y_data).reshape(-1,1))
    #train/test set split  _ test_size = 1000
    global x_test_t
    global y_test
    y_train = y_data[:-1000]
    y_test = y_data[-1000:]
    x_train = x_data[:-1000]
    x_test = x_data[-1000:]

    #x_data reshape(len, feature, timestep)
    x_train_t = np.array(x_train).reshape(len(x_train), 3, timestep)
    x_test_t = np.array(x_test).reshape(len(x_test), 3, timestep)
    
    #Machine Learning Model
    global model
    model = tf.keras.models.Sequential()
    model.add(LSTM(10, input_shape=(x_train_t.shape[1], x_train_t.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    model.fit(x_train_t, y_train, epochs=10, batch_size =1, verbose =1)
    
    pred = model.predict(x_test_t)
    pred = np.exp(sc_y.inverse_transform(pred))
    predict = pd.DataFrame(pred)
    predict.columns = ['Predict_{}'.format(column)]
    predict['Real_{}'.format(column)] = np.exp(sc_y.inverse_transform(y_test))
    predict.index = y_ymd[-1000:]
    predict['Error_{}'.format(column)] = predict['Real_{}'.format(column)] - predict['Predict_{}'.format(column)]
    global adj
    lennum = round((len(predict))* (2/3))
    adj = predict['Error_{}'.format(column)][-int(lennum):].mean()
    print('Mean Error:', adj)
    return predict
    
def pred_stock(period,timestep):
  model.fit(x_test_t, y_test, epochs=10, batch_size =1, verbose =1)
  x_data = np.log(future_test)
  x_data = pd.DataFrame(sc_x.transform(x_data), index= x_data.index, columns=x_data.columns)
  #Time step 결정
  lst = []
  lst.append('High')
  for step in range(1,timestep):
      x_data['High_{}'.format(step)] = x_data['High'].shift(step)
      lst.append('High_'+str(step))
  lst.append('Low')
  for step in range(1,timestep):
      x_data['Low_{}'.format(step)] = x_data['Low'].shift(step)
      lst.append('Low_'+str(step))
  lst.append('Close')
  for step in range(1,timestep):
      x_data['Close_{}'.format(step)] = x_data['Close'].shift(step)
      lst.append('Close_'+str(step))
  x_data = x_data.dropna()
  x_data = x_data[lst] 
  #x_data reshape(len, feature, timestep) 
  x_data_t = np.array(x_data).reshape(len(x_data), 3, timestep) 

  #Prediction
  pred = model.predict(x_data_t)
  pred = np.exp(sc_y.inverse_transform(pred))
  
  tmplst=[]
  for i in range(1,period+1):
    tmpwd = future_test.index[-1] + " +"+str(i)+"일"
    tmplst.append(tmpwd)

  predict = pd.DataFrame(pred, columns = ['Col'])
  predict['Adj_Col'] = predict['Col']+ adj
  predict['Date'] = tmplst

  return predict
# df = select_stock(stock_code, period)  
def prepro2(df, column, timestep):   # x_data.shape(len(x_train), timestep, 3)
    x_data = df[['High', 'Low', 'Close']]
    #x_data log&MinMaxScale
    x_data = np.log(x_data)
    global sc_x
    sc_x = MinMaxScaler()
    x_data = pd.DataFrame(sc_x.fit_transform(x_data), index=x_data.index, columns=x_data.columns)
    #Time step 결정
    for step in range(1,timestep):
        x_data['High_{}'.format(step)] = x_data['High'].shift(step)
    for step in range(1,timestep):
        x_data['Low_{}'.format(step)] = x_data['Low'].shift(step)
    for step in range(1,timestep):
        x_data['Close_{}'.format(step)] = x_data['Close'].shift(step)
    x_data = x_data.dropna()
        
    #y_data
    #y columns 선택변수로 만들 것 
    target_word = 'Target_'+str(column)
    y_data = df[target_word][(timestep-1):] # Timestep의 갯수만큼 
    y_ymd = df['Target_YMD'][(timestep-1):]
    y_data = np.log(y_data)
    global sc_y 
    sc_y = MinMaxScaler()
    y_data = sc_y.fit_transform(np.array(y_data).reshape(-1,1))
    #train/test set split  _ test_size = 1000
    global x_test_t
    global y_test
    y_train = y_data[:-1000]
    y_test = y_data[-1000:]
    x_train = x_data[:-1000]
    x_test = x_data[-1000:]

    #x_data reshape(len, feature, timestep)
    x_train_t = np.array(x_train).reshape(len(x_train), timestep, 3)
    x_test_t = np.array(x_test).reshape(len(x_test), timestep, 3)
    
    #Machine Learning Model
    global model
    model = tf.keras.models.Sequential()
    model.add(LSTM(10, input_shape=(x_train_t.shape[1], x_train_t.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    model.fit(x_train_t, y_train, epochs=50, batch_size =1, verbose =1)
    
    pred = model.predict(x_test_t)
    pred = np.exp(sc_y.inverse_transform(pred))
    predict = pd.DataFrame(pred)
    predict.columns = ['Predict_{}'.format(column)]
    predict['Real_{}'.format(column)] = np.exp(sc_y.inverse_transform(y_test))
    predict.index = y_ymd[-1000:]
    predict['Error_{}'.format(column)] = predict['Real_{}'.format(column)] - predict['Predict_{}'.format(column)]
    global adj
    lennum = round((len(predict))* (2/3))
    adj = predict['Error_{}'.format(column)][-int(lennum):].mean()
    print('Mean Error:', adj)
    return predict
    

def pred_stock2(period,timestep):
  model.fit(x_test_t, y_test, epochs=50, batch_size =1, verbose =1) 
  x_data = np.log(future_test)
  x_data = pd.DataFrame(sc_x.transform(x_data), index= x_data.index, columns=x_data.columns)
  #Time step 결정
  for step in range(1,timestep):
      x_data['High_{}'.format(step)] = x_data['High'].shift(step)
  for step in range(1,timestep):
      x_data['Low_{}'.format(step)] = x_data['Low'].shift(step)
  for step in range(1,timestep):
      x_data['Close_{}'.format(step)] = x_data['Close'].shift(step)
  x_data = x_data.dropna()
  #x_data reshape(len, feature, timestep) 
  x_data_t = np.array(x_data).reshape(len(x_data), timestep, 3) 

  #Prediction
  pred = model.predict(x_data_t)
  pred = np.exp(sc_y.inverse_transform(pred))
  
  tmplst=[]
  for i in range(1,period+1):
    tmpwd = future_test.index[-1] + " +"+str(i)+"일"
    tmplst.append(tmpwd)

  predict = pd.DataFrame(pred, columns = ['Col'])
  predict['Adj_Col'] = predict['Col']+ adj
  predict['Date'] = tmplst

  return predict
# 테스트 셋 주기적으로 업데이트하는 함수
# 트레이닝 미리 해놓을 것

def setup(timestep, column):
    global x_data
    global y_data
    x_data = df[['High', 'Low', 'Close']]
    #x_data log&MinMaxScale
    x_data = np.log(x_data)
    global sc_x
    sc_x = MinMaxScaler()
    x_data = pd.DataFrame(sc_x.fit_transform(x_data), index=x_data.index, columns=x_data.columns)
    #Time step 결정
    lst = []
    lst.append('High')
    for step in range(1,timestep):
        x_data['High_{}'.format(step)] = x_data['High'].shift(step)
        lst.append('High_'+str(step))
    lst.append('Low')
    for step in range(1,timestep):
        x_data['Low_{}'.format(step)] = x_data['Low'].shift(step)
        lst.append('Low_'+str(step))
    lst.append('Close')
    for step in range(1,timestep):
        x_data['Close_{}'.format(step)] = x_data['Close'].shift(step)
        lst.append('Close_'+str(step))
    x_data = x_data.dropna()
    x_data = x_data[lst]
    
    #y_data
    #y columns 선택변수로 만들 것 
    target_word = 'Target_'+str(column)
    y_data = df[target_word][(timestep-1):] # Timestep의 갯수만큼 
    y_ymd = df['Target_YMD'][(timestep-1):]
    y_data = np.log(y_data)
    global sc_y 
    sc_y = MinMaxScaler()
    y_data = sc_y.fit_transform(np.array(y_data).reshape(-1,1))
    #train/test set split  _ test_size = 1000

    y_train = y_data[:-1000]
    x_train = x_data[:-1000]

    #x_data reshape(len, feature, timestep)
    x_train_t = np.array(x_train).reshape(len(x_train), 3, timestep)
    
    #Machine Learning Model
    global model
    model = tf.keras.models.Sequential()
    model.add(LSTM(10, input_shape=(x_train_t.shape[1], x_train_t.shape[2])))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    model.fit(x_train_t, y_train, epochs=10, batch_size =1, verbose =1)



def update(repeat):
  repeat = repeat
  first = 1000 -repeat
  second = first - repeat
  y_ymd = df['Target_YMD'][(timestep-1):]
  y_test = y_data[-1000:-first]
  x_test = x_data[-1000:-first]
  x_test_t = np.array(x_test).reshape(len(x_test), 3, timestep)
  pred = model.predict(x_test_t)
  pred = np.exp(sc_y.inverse_transform(pred))
  pred = pd.DataFrame(pred, index= y_ymd[-1000:-first], columns=['Prediction'])
  totalpred = pd.DataFrame()
  totalpred = pd.concat([totalpred, pred])

  repeatnum = int(1000 / repeat)
  for i in range(repeatnum):  #
    model.fit(x_test_t, y_test, epochs=10, batch_size =1, verbose =1)  
    if second > 0:
      x_test = x_data[-first:-second]
      y_test = y_data[-first:-second]
      x_test_t = np.array(x_test).reshape(len(x_test), 3, timestep)
      pred = model.predict(x_test_t)
      pred = np.exp(sc_y.inverse_transform(pred))
      pred = pd.DataFrame(pred, index= y_ymd[-first:-second], columns=['Prediction'])
      totalpred = pd.concat([totalpred, pred])
    elif second == 0:
      x_test = x_data[-first:]
      y_test = y_data[-first:]
      x_test_t = np.array(x_test).reshape(len(x_test), 3, timestep)
      pred = model.predict(x_test_t)
      pred = np.exp(sc_y.inverse_transform(pred))
      pred = pd.DataFrame(pred, index= y_ymd[-first:], columns=['Prediction'])
      totalpred = pd.concat([totalpred, pred])
    first = first-(repeat)
    second = second-(repeat)



  totalpred['Real']= np.exp(sc_y.inverse_transform(y_data[-1000:]))
  totalpred['Error'] = totalpred[totalpred.columns[1]] - totalpred[totalpred.columns[0]]
  return totalpred
def pred_stock2(period,timestep):
  x_data = np.log(future_test)
  x_data = pd.DataFrame(sc_x.transform(x_data), index= x_data.index, columns=x_data.columns)
  #Time step 결정
  lst = []
  lst.append('High')
  for step in range(1,timestep):
      x_data['High_{}'.format(step)] = x_data['High'].shift(step)
      lst.append('High_'+str(step))
  lst.append('Low')
  for step in range(1,timestep):
      x_data['Low_{}'.format(step)] = x_data['Low'].shift(step)
      lst.append('Low_'+str(step))
  lst.append('Close')
  for step in range(1,timestep):
      x_data['Close_{}'.format(step)] = x_data['Close'].shift(step)
      lst.append('Close_'+str(step))
  x_data = x_data.dropna()
  x_data = x_data[lst] 
  #x_data reshape(len, feature, timestep) 
  x_data_t = np.array(x_data).reshape(len(x_data), 3, timestep) 

  #Prediction
  pred = model.predict(x_data_t)
  pred = np.exp(sc_y.inverse_transform(pred))
  
  tmplst=[]
  for i in range(1,period+1):
    tmpwd = future_test.index[-1] + " +"+str(i)+"일"
    tmplst.append(tmpwd)

  predict = pd.DataFrame(pred, columns = ['Col'])
  predict['Date'] = tmplst

  return predict
period = 10 
timestep = 2
df = select_stock('AAPL', period, timestep)
check = prepro2(df, 'Close', timestep)
check
result = pred_stock2(period, timestep)
result
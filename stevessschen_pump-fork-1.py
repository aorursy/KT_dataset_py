import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import copy
import random
#註：time_left欄位為訓練資料的Y，代表剩餘分鐘數

#載入訓練檔，並將空白填滿0
df = pd.read_csv('../input/train-test-data/train_sensor.csv')
df = df.drop(['timestamp','sensor_15','sensor_50','Unnamed: 0','machine_status'],axis=1)
df =df.fillna(value=0)

#batch_size128
batch_size=128
df.head()
#設定正規化
def normalize(train2):
  for i in train2:
    max11=valueList[i][0]
    min11=valueList[i][1]
    mean11=valueList[i][2]
    train2[i]= train2[i].apply(lambda x: (x - mean11) / (max11 - min11))
  #train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train2
#設定反正規化
def unnormalize(train):
  train2 = copy.deepcopy(train) 
  for i in range(len(train)):
      train2[i]=train[i]*(valueList['time_left'][0]-valueList['time_left'][1])+valueList['time_left'][2]
  return train2
#儲存正規化前的數值，用以正規化或反正規化
def save_normalize(df):
    ValueList=copy.deepcopy(df[:][:3])
    for i in df:
        max1=np.max(df[i][:])
        min1=np.min(df[i][:])
        mean1=np.mean(df[i][:])
        ValueList[i][0]=max1
        ValueList[i][1]=min1
        ValueList[i][2]=mean1
    #../input/train-test-data/ValueList.csv
    ValueList.to_csv('./ValueList.csv')
#切分為訓練集或測試集
def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val
#建立訓練資料
def buildTrain(train):
  #pastDay為每步前進30分鐘
  pastDay=1/(24*2)
  #futureDay為每步前進兩天
  futureDay=2
  #pastDay=1/(24*8)
  print(train.shape[0])
  pastMinute=(int)(pastDay*60*24)
  futureMinute=futureDay*60*24
  X_train, Y_train = [], []
  
  x_train=[]
  y_train=[]
  print((int)(train.shape[0]/pastMinute))
  #time_left為訓練資料的Y，代表剩餘分鐘數
  train2=train.drop(['time_left'],axis=1)
  #每步前進30分鐘
  for i in range((int)(train.shape[0]/pastMinute)):
    #如果超過陣列範圍，往前移動(目前為跳過此樣本)
    if i*pastMinute+futureMinute>=train.shape[0]:
        continue
        X_train=np.array(train2.iloc[train.shape[0]-1-futureMinute:train.shape[0]-1][:])
        Y_train=np.array(train.iloc[train.shape[0]-1]["time_left"])
        #print(12345)
        #print(train.shape[0])
        #print(i*pastMinute+futureMinute)
    else:
        #如果超過broken天數，往前移動(目前為跳過此樣本)
        if train.iloc[i*pastMinute+futureMinute]["time_left"]>train.iloc[i*pastMinute]["time_left"]:
            continue
            tempMinute=0
            for j in range(futureMinute):
                if unnormalize([train.iloc[i*pastMinute+j]["time_left"]])[0]==0:
                    tempMinute=j
                    break
            #print(unnormalize([train.iloc[i*pastMinute+tempMinute]["time_left"]])[0])
            #print(train.iloc[i*pastMinute+tempMinute]["time_left"])
            X_train=np.array(train2.iloc[i*pastMinute+tempMinute-futureMinute:i*pastMinute+tempMinute][:])
            Y_train=np.array(train.iloc[i*pastMinute+tempMinute]["time_left"])
            #print(Y_train)
            #將該兩天份的資料放入陣列
        else:
            X_train=np.array(train2.iloc[i*pastMinute:i*pastMinute+futureMinute][:])
            Y_train=np.array(train.iloc[i*pastMinute+futureMinute]["time_left"])
    #print(X_train.shape)
    x_train.append(X_train)
    y_train.append(Y_train)
  return np.stack(np.array(x_train)), np.stack(np.array(y_train))
#洗牌
def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]
#剩餘分鐘數改成剩餘天數
df["time_left"]=df["time_left"]/60/24
#儲存正規化前的參數
save_normalize(df)
#使用上一步驟儲存的參數
valueList= pd.read_csv('./ValueList.csv')
#正規化
train_norm = normalize(df)
#健力訓練集
X_train, Y_train = buildTrain(train_norm)
#洗牌
X_train, Y_train = shuffle(X_train, Y_train)
print(X_train.shape)
#切分為訓練集和測試集
X_train, Y_train, X_val, Y_val = splitData(X_train, Y_train, 0.2)
X_train.shape
#建立模型 (Steve modified the function name!)
def buildOneToOneModel_old(shape):
  model = Sequential()
  #model.add(Conv2D(32, 5, 5, input_shape=(4,38, 38, 50),activation = 'relu'))
  model.add(LSTM(128,input_length=1, input_dim=50*2880,return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(128,return_sequences=True))
  #model.add(Dropout(0.2))
  #model.add(LSTM(128,return_sequences=True))
  #model.add(Dropout(0.2))
  model.add(Dense(1))    # or use model.add(Dense(1))
  model.compile(loss='mse', optimizer="adam") 
  model.summary()
  return model
#建立模型: Steve modified!
def buildOneToOneModel(shape):
  model = Sequential()
  #model.add(Conv2D(32, 5, 5, input_shape=(4,38, 38, 50),activation = 'relu'))
  model.add(LSTM(32,input_length=1, input_dim=50*2880,return_sequences=True))
  model.add(Dropout(0.1))
  model.add(LSTM(16,return_sequences=True))
  model.add(Dropout(0.2))
  #model.add(LSTM(128,return_sequences=True))
  #model.add(Dropout(0.2))
  model.add(Dense(1))    # or use model.add(Dense(1))
  model.compile(loss='mse', optimizer="adam")
  model.summary()
  return model
#儲存loss圖片
def plot1(history):
     N = np.arange(0, len(history['loss']))
     fig=plt.figure()
     fig.set_size_inches(18.5, 10.5)
     plt.plot(N, history['loss'], label = "train_loss")
     plt.plot(N, history['val_loss'], label = "val_loss")
     plt.xlabel("Epoch #")
     plt.ylabel("Loss")
     plt.legend()
     plt.savefig('loss.png', dpi=100)
     plt.close()
#建立模型
model = buildOneToOneModel(X_train.shape)
#model=load_model("LSTM_result_batch256_epoch128_per96_random2.h5")
#連續10步loss不下降時停止訓練
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
#reshape
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1]*X_train.shape[2]))
X_val=np.reshape(X_val,(X_val.shape[0],X_val.shape[1]*X_val.shape[2]))
#開始訓練
batch_size=16 #128 (Steve modified!)
#history=model.fit(X_train[:,np.newaxis], Y_train, epochs=128, batch_size=batch_size, validation_data=(X_val[:,np.newaxis], Y_val),  callbacks=[callback])
history=model.fit(X_train[:,np.newaxis], Y_train, epochs=128, batch_size=batch_size, validation_data=(X_val[:,np.newaxis], Y_val),  callbacks=[callback])
len(X_val)
#驗證測試集，驗證過程會印出預測和實際的label
count=0
trueCount=0
for i in range(0,len(X_val),1):
    if 1+1==2:
        x_val=np.array(X_val[i])
        y_val=np.array(Y_val[i])
        x_val=np.reshape(x_val,(1,x_val.shape[0]))
        y_val=np.reshape(y_val,(1))
        prediction=model.predict(x_val[:,np.newaxis])
        # Steve modified!
        #if unnormalize(prediction[0][0])[0]<=unnormalize(y_val)[0]+0.5 and unnormalize(prediction[0][0])[0]>=unnormalize(y_val)[0]-0.5:
        if unnormalize(prediction[0][0])[0]<=unnormalize(y_val)[0]+1.0 and unnormalize(prediction[0][0])[0]>=unnormalize(y_val)[0]-1.0:
            #print('true')
            trueCount+=1
        count+=1
        print('predict: ', unnormalize(prediction[0][0])[0], 'day')
        print('label:   ', unnormalize(y_val)[0], 'day') 
        print('\n')
print('acc',trueCount/count)
print('accuracy of validation: ',trueCount/count)
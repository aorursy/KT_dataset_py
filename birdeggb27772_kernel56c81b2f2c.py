import pandas as pd
import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten, LSTM, TimeDistributed, RepeatVector
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint
#from tensorflow.python.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import copy
df = pd.read_csv('../input/train-test-data/train_sensor.csv')
df2 = pd.read_csv('../input/train-test-data/test_sensor.csv')
df = df.drop(['timestamp','sensor_15','sensor_50','Unnamed: 0','machine_status'],axis=1)
df2 = df2.drop(['timestamp','sensor_15','sensor_50','Unnamed: 0','machine_status'],axis=1)
df =df.fillna(value=0)
df2 =df2.fillna(value=0)
def normalize(train2):
  for i in train2:
    max11=valueList[i][0]
    min11=valueList[i][1]
    mean11=valueList[i][2]
    train2[i][train2[i]>max11]=max11
    train2[i][train2[i]<min11]=min11
    train2[i]= train2[i].apply(lambda x: (x - mean11) / (max11 - min11))
  #train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
  return train2
def unnormalize(train):
  train2 = copy.deepcopy(train) 
  for i in range(len(train)):
      train2[i]=train[i]*(valueList['time_left'][0]-valueList['time_left'][1])+valueList['time_left'][2]
  return train2
def save_normalize(df):
    ValueList=copy.deepcopy(df[:][:3])
    for i in df:
        max1=np.max(df[i][:])
        min1=np.min(df[i][:])
        mean1=np.mean(df[i][:])
        ValueList[i][0]=max1
        ValueList[i][1]=min1
        ValueList[i][2]=mean1
    ValueList.to_csv('./ValueList.csv')
def splitData(X,Y,rate):
  X_train = X[int(X.shape[0]*rate):]
  Y_train = Y[int(Y.shape[0]*rate):]
  X_val = X[:int(X.shape[0]*rate)]
  Y_val = Y[:int(Y.shape[0]*rate)]
  return X_train, Y_train, X_val, Y_val
def buildTrain(train):
  X_train, Y_train = [], []
  train2=train.drop(['time_left'],axis=1)
  X_train=np.array(train2.iloc[:][:]).tolist()
  Y_train=np.array(train.iloc[:]["time_left"]).tolist()
  return np.array(X_train), np.array(Y_train)
def shuffle(X,Y):
  np.random.seed(10)
  randomList = np.arange(X.shape[0])
  np.random.shuffle(randomList)
  return X[randomList], Y[randomList]
df["time_left"]=df["time_left"]/60/24
df2["time_left"]=df2["time_left"]/60/24
save_normalize(df)
valueList= pd.read_csv('./ValueList.csv')
train_norm = normalize(df)
train_norm2 = normalize(df2)
X_train, Y_train = buildTrain(train_norm)
X_val, Y_val = buildTrain(train_norm2)
X_train, Y_train = shuffle(X_train, Y_train)
X_val, Y_val = shuffle(X_val, Y_val)
X_train = X_train[:,np.newaxis]
X_val = X_val[:,np.newaxis]
def buildOneToOneModel(shape):
  model = Sequential()
  model.add(LSTM(128, input_length=shape[1], input_dim=shape[2],return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(128, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(LSTM(128, return_sequences=True))
  model.add(Dropout(0.2))
  model.add(TimeDistributed(BatchNormalization()))
  #model.add(LSTM(10, input_length=shape[1], input_dim=shape[2],return_sequences=True))
  #model.add(LSTM(10, input_length=shape[1], input_dim=shape[2],return_sequences=True))
  #model.add(Dense(1))
  model.add(TimeDistributed(Dense(1)))    # or use model.add(Dense(1))
  #sgd = optimizers.SGD(lr = 0.1, decay = 1e-6, momentum = 0.9, nesterov = True)
  model.compile(loss='mse', optimizer="adam")
  model.summary()
  return model
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
model = buildOneToOneModel(X_train.shape)
#model=load_model("LSTM_result_batch256_epoch128_per96_random2.h5")
callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
history=model.fit(X_train, Y_train, epochs=100, batch_size=256, validation_data=(X_val, Y_val), callbacks=[callback])
#plot1(history.history)

prediction=model.predict(X_val)
count=0
prediction1=prediction
for i in range(prediction1.shape[0]):
    prediction1[i]=unnormalize(prediction[i])
Y_val1=unnormalize(Y_val)

for i in range(len(Y_val)):
    if prediction1[i][0][0]<=Y_val1[i]+0.5 and prediction1[i][0][0]>=Y_val1[i]-0.5:
        count=count+1
print(count/len(Y_val1)) 




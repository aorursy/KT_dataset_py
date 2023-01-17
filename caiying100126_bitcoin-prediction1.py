import pandas as pd

from sklearn.preprocessing import MinMaxScaler

import numpy as np
start_date = '2016-1-1'

end_date = '2019-8-22'

data = pd.read_csv('Coinbase_BTCUSD_d.csv',skiprows=1)



#Filtering out relevant dates

data['Date']=pd.to_datetime(data['Date'])

mask = (data['Date'] >= start_date) & (data['Date'] <= end_date)

data = data.loc[mask]



#Sort date in inverse order

data =data.sort_values(by='Date')

data.drop(['Symbol'],axis=1,inplace=True)

data.head()
data.isnull().any()
from sklearn.model_selection import train_test_split



#Defining features to training model and normalizing data

features = ["Open","High","Low","Close","Volume BTC","Volume USD"]

train, test = train_test_split(data,train_size=0.8,test_size=0.2,shuffle=False)

scaler = MinMaxScaler(feature_range=(0,1))

trainScaled  = scaler.fit_transform(train.loc[:,features].values)

testScaled = scaler.fit_transform(test.loc[:,features].values)

print(trainScaled.shape, testScaled.shape)
#look ahead time window (historical data used to predict)

timeWindow = 20

batchSize = 32

# days of prices to predict

predictWindow = 7 
from tqdm import tqdm_notebook as tqdm



# To extract the input data and target labels for model

# Parameters: data       - input data 

#             targetCol  - column number for the target label

# Return:     X          - input data into model. Each data point = (timeWindow, number of features)

#             y          - corresponding target labels ('Close' price)

def buildTimeSeries(data, targetCol):

    nData = data.shape[0] - timeWindow - predictWindow

    nFeatures = data.shape[1]

    # Initialize with array of zeros 

    X = np.zeros((nData,timeWindow,nFeatures))

    y = np.zeros((nData,predictWindow))

    

    for i in tqdm(range(nData)):

        X[i]=data[i:timeWindow+i]

        y[i]=data[timeWindow+i:i+timeWindow+predictWindow,targetCol]    #Only extract the "Close" price

    # Reshaping y to (nData, predictWindow, 1) to fit into the model below

    y = np.reshape(y,(y.shape[0],y.shape[1],1))

    print("length of time-series i/o",X.shape,y.shape)

    return X,y



# To drop the extra rows that do not fit into one batch (Optional)

def dropExtraRows(data, batchSize):

    extraRows = data.shape[0]%batchSize

    if extraRows>0:

        return data[:-extraRows]

    else:

        return data
# Generating XTrain and yTrain from the scaled training data

XTrain, yTrain = buildTimeSeries(trainScaled, 3) #"Close" is in column 3 of X_train





# Generating XTest, yTest, XVal, yVal from the scaled testing data

Xtemp, ytemp = buildTimeSeries(testScaled, 3)

XVal, XTest = np.array_split(Xtemp,2)

yVal, yTest = np.array_split(ytemp,2)





# Optional - Dropping of extra rows that do not fit into batch size

# variables = [XTrain, yTrain, XVal, yVal, XTest, yTest]

# for var in variables:

#     var = dropExtraRows(var,batchSize)

#     print(var.shape)
from keras import backend as K

from keras.models import Model, Sequential

from keras.layers import (BatchNormalization, Dense, Input, TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, RepeatVector)



outputDim =predictWindow

nUnits = 50



Input = Input(name ="input", shape = (timeWindow,XTrain.shape[2]))

GRU1 = GRU(name='gru1',units=nUnits,activation='relu')(Input)

repVec = RepeatVector(outputDim)(GRU1)

GRU2 = GRU(name='gru2',units=nUnits,activation='relu',dropout=0.3,return_sequences=True)(repVec)

timeDistributed = TimeDistributed(Dense(1),name='timeDistributedDense')(GRU2)

yPred = Activation('sigmoid',name='sigmoid')(timeDistributed)



model =Model(inputs=Input,outputs=yPred)



print(model.summary())
from keras import backend as K

from keras.models import Model, Sequential

from keras.layers import (BatchNormalization, Dense, Input, TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, RepeatVector)



outputDim =predictWindow

nUnits = 50



Input = Input(shape = (timeWindow,XTrain.shape[2]))

GRU1 = GRU(units=nUnits,activation='relu',return_sequences=True)(Input)

GRU2 = GRU(units=nUnits,activation='relu',return_sequences=False)(GRU1)

dense = Dense(outputDim)(GRU2)

yPred = Activation('sigmoid')(dense)

model2 =Model(inputs=Input,outputs=yPred)



print(model2.summary())
# Defining RMSE loss function

def rmse(y_true,y_pred):

    return K.sqrt(K.mean(K.square(y_pred-y_true)))



model.compile(optimizer='rmsprop',loss=rmse)

model2.compile(optimizer='rmsprop',loss=rmse)
from keras.callbacks import ModelCheckpoint

import os.path



print('TRAINING ENCODER-DECODER ARCHITECTURE')

print('=====================================\n')

epochs=5

checkpointer = ModelCheckpoint(filepath='model1.h5',save_best_only=True)

history = model.fit(XTrain,yTrain, validation_data= (XVal,yVal), epochs=epochs, shuffle=False,callbacks=[checkpointer])
# Different Dimensions for the Simple RNN output. Need to remove the extra axis 

yTest2 = np.squeeze(yTest)

yVal2 = np.squeeze(yVal)

yTrain2 = np.squeeze(yTrain)



print('TRAINING SIMPLE RNN')

print('=====================================\n')



epochs=5

checkpointer = ModelCheckpoint(filepath='model2.h5',save_best_only=True)

history = model2.fit(XTrain,yTrain2, validation_data= (XVal,yVal2), epochs=epochs, shuffle=False,callbacks=[checkpointer])
from sklearn.metrics import mean_squared_error

predictions = model.predict(XTest)

predictions = np.squeeze(predictions)

yTest = np.squeeze(yTest)

rootmse = np.sqrt(mean_squared_error(yTest,predictions))

print('With the Encoder-Decoder Model')

print('Test RMSE: %.3f' % rootmse)
predict2 = model2.predict(XTest)

rootmse = np.sqrt(mean_squared_error(yTest2,predict2))

print('With the Simple RNN')

print('Test RMSE: %.3f' % rootmse)
import pandas as pd

ARIMA_Coinbase_BTCUSD_d = pd.read_csv("../input/ARIMA_Coinbase_BTCUSD_d.csv")

ARIMA_Coinbase_USD_daily = pd.read_csv("../input/ARIMA_Coinbase_USD_daily.csv")

NN_Coinbase_BTCUSD_d = pd.read_csv("../input/NN_Coinbase_BTCUSD_d.csv")

NN_Coinbase_ETHUSD_d = pd.read_csv("../input/NN_Coinbase_ETHUSD_d.csv")
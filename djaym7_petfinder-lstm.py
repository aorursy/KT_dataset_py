import pandas as pd
import numpy as np
import keras
%matplotlib inline
import missingno
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense,Dropout,CuDNNLSTM,LSTM
from keras.callbacks import ModelCheckpoint
pd.options.display.max_columns=500
import numpy as np
import keras
from keras.callbacks import EarlyStopping,ModelCheckpoint
import pickle
from sklearn.preprocessing import StandardScaler

from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, CuDNNLSTM, Multiply,Dropout
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
from keras.callbacks import TensorBoard
from subprocess import check_output
print(check_output(["ls", "../input/"]).decode("utf8"))

data = pd.read_csv('../input/train.csv')
data.describe()
data.hist(figsize=(20,20))
missingno.matrix(data,figsize=(20,8))
def preprocess(data,testdata=False):
    temp = data.drop(columns=['Name','Description','RescuerID','PetID'])
    
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    temp.State=encoder.fit_transform(temp.State)
    
    from dummyPy import OneHotEncoder
    encoder = OneHotEncoder(['Type','Breed1', 'Breed2', 'Gender', 'Color1', 'Color2',
           'Color3', 'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed',
           'Sterilized', 'Health'])
    temp=encoder.fit_transform(temp)
    X = temp.drop(columns=['AdoptionSpeed'])
    y=temp['AdoptionSpeed']
    y = np.array(y)
    y = keras.utils.to_categorical(y)
    print(X.shape,y.shape)
    return X,y
X_train,y_train=preprocess(data,False)
print(X_train.shape)
X_train = np.array(X_train).reshape(14993,1,361)
print(X_train.shape)
def Lstm(X_train):

    lstm  = Sequential()
    lstm.add(LSTM(units=128,input_shape=(X_train.shape[1],X_train.shape[2]),return_sequences=True,activation='tanh'))
    lstm.add(Dropout(0.5))
    lstm.add(LSTM(units=256,return_sequences=True,activation='tanh'))
    lstm.add(Dropout(0.5))
    lstm.add(LSTM(units=96))

    lstm.add(Dense(units=5,activation='softmax'))
    lstm.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
    return lstm
def dnn():
    #DNN
    model = Sequential()
    model.add(Dense(input_shape=(X.shape[1],),units=32,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=32,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=5,activation='softmax'))
    model.compile(optimizer='adam',loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])
    return model

lstm = Lstm(X_train)
i = 'lstm'
#filePath =f'C:\\Users\\djaym7\\Desktop\\Github\\petfinder\\models\\{i}.h5'
#checkpoint = ModelCheckpoint(filepath=filePath,monitor='val_acc',mode='max',save_best_only=True,verbose=1,save_weights_only=False)
lstm.fit(X_train,y_train,epochs=50,validation_split=0.25)
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas import datetime

from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, Flatten, Bidirectional, Conv2D, MaxPooling2D, Lambda, MaxPool2D, BatchNormalization, Input, concatenate, K, Reshape, LSTM, CuDNNLSTM

from keras.utils import np_utils

from keras.optimizers import *

from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard

from keras.utils.np_utils import to_categorical

from keras.losses import *

from sklearn.preprocessing import LabelEncoder, minmax_scale, MinMaxScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from keras import backend as K

import sklearn

import os

import matplotlib.pyplot as plt

%matplotlib inline

print(os.listdir("../input"))



#Import Dataset

df = pd.read_csv('../input/BITS AIC 2019 - Reflexis Raw Dataset.csv')

df['DATE'] = pd.to_datetime(df['DATE']) #Convert date strings to datetime format

df = df.set_index('DATE') #Index by Date

df.head()
#Cleaning Dataset

df = df.dropna() #Drop missing values

df.head()
#Exploratory Data Analysis

print('Max sales = ', df['SALES_ACTUAL'].max()) 

print('Number of records =', len(df))
#Scaling Actual Sales values

df1 = df.groupby('STORE') #Group data by store

_temp = []

scalar = MinMaxScaler(feature_range=(-1,1))

for store_no, store_data in df1:

    data = df1.get_group(store_no).copy()    

    data['SALES_NORM'] = scalar.fit_transform(data['SALES_ACTUAL'].values.reshape(-1,1)) #Scaled sales between (-1,1)

    for i in data['SALES_NORM'].values:

        _temp.append(i)

df['SALES_NORM'] = _temp 

df1 = df.groupby('STORE') #Store grouped dataset separately 

df
# Helper Functions for Graph Generation

class MetricsCheckpoint(Callback):

    # Callback that saves metrics after each epoch

    def __init__(self, savepath):

        super(MetricsCheckpoint, self).__init__()

        self.savepath = savepath

        self.history = {}

        

    def on_epoch_end(self, epoch, logs=None):

        for k, v in logs.items():

            self.history.setdefault(k, []).append(v)

        np.save(self.savepath, self.history)



#Plots RMSE

def plotKerasLearningCurve():

    plt.figure(figsize = (10,5))

    metrics = np.load('logs.npy')[()]

    filt = ['loss'] # try to add 'loss' to see the loss learning curve

    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):

        l = np.sqrt(scalar.inverse_transform(np.array(metrics[k]).reshape(-1,1)).ravel()).tolist() #Restructuring data

        plt.plot(l, c = 'r' if 'val' not in k else 'b', label = 'val' if 'val' in k else 'train') #line plot of of metrics or loss

        x = np.argmin(l)

        y = float(l[x])

        plt.scatter(x, y, lw = 0, alpha = 0.25, s = 100, c = 'r' if 'val' not in k else 'b') #Scatter plot of maxima

        plt.text(x, y, '{} = {:.4f}'.format(x,y), size ='15', color= 'r' if 'val' not in k else 'b')   

    plt.legend(loc = 4)

    plt.axis([0, None, None, None]);

    plt.grid()

    plt.xlabel('Number of epochs')

    plt.ylabel('RMSE')

    plt.savefig('./RMSE_curve.png')



#Plot Loss    

def plot_learning_curve(history):

    plt.figure(figsize = (10,5))

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('model loss')

    plt.ylabel('loss')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('./loss_curve.png')
#RMSE Calculation function

def root_mean_squared_error(y_true, y_pred):

    return K.sqrt(K.mean(K.square(y_pred - y_true), axis = -1)) 



#Neural Network

def f1(dfin, store_no, epochs):

    #predict sales given date, store number, system scheduled hours and manager scheduled hours

    while True:

        data = dfin.get_group(store_no) #Get data for given store number

        batch_size = 8

        scalar.fit(data['SALES_ACTUAL'].values.reshape(-1,1)) #Fit scalar to data for given store number

        split_date = '2018-02-01' 

        #Split data around split_date

        X_train, Y_train, X_test, Y_test = data[data.index < split_date][['STORE','MANAGER_SCHED_HOURS','SYSTEM_SCHED_HOURS']], data[data.index < split_date]['SALES_NORM'], data[data.index>= split_date][['STORE','MANAGER_SCHED_HOURS','SYSTEM_SCHED_HOURS']], data[data.index >= split_date]['SALES_NORM']

        #Reshaping the data to correct shape (Time stamp form)

        X_train = X_train.values.reshape(X_train.shape[0],1,X_train.shape[1])

        X_test = X_test.values.reshape(X_test.shape[0],1,X_test.shape[1])

        input_shape = (1,X_test.shape[2])

        num_classes = 1 # Since there is only one output column with one value in it

        model = Sequential() 

        model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True, input_shape=input_shape)))

        model.add(Dropout(0.25))

        model.add(Bidirectional(CuDNNLSTM(64, return_sequences=True)))

        model.add(Dropout(0.25))

        model.add(Bidirectional(CuDNNLSTM(64)))

        model.add(Dropout(0.25))

        model.add(Dense(num_classes, activation='tanh')) 

        #Set loss function and train model

        model.compile(loss=root_mean_squared_error, optimizer = Adam(lr=0.01), metrics=[root_mean_squared_error])

        history = model.fit(X_train, Y_train.values, epochs = epochs, steps_per_epoch=int(len(X_train) / batch_size), validation_data=(X_test, Y_test.values), validation_steps=max(int(len(X_test)/batch_size),1), callbacks = [MetricsCheckpoint('logs')])

        if np.inf not in history.history['loss']:

            score = model.evaluate(X_test, Y_test.values)

            plotKerasLearningCurve()

            plt.show()  

            plot_learning_curve(history)

            plt.show()

            return model
#Generating model for store 203

lstm_model = f1(df1, 203, 50)
#Iterating over all possible inputs to get variation of sales.

store_no = 203

ls1 = []

ls2 = []

dfstore = df1.get_group(store_no)

scalar.fit(dfstore['SALES_ACTUAL'].values.reshape(-1,1))

wpd = 12

for i in range(wpd*7*4+1):

    x_in = np.array([[store_no, float(i)/4.0, float(i)/4.0]])

    x_in = x_in.reshape(x_in.shape[0], 1, x_in.shape[1])

    ls1.append(scalar.inverse_transform(lstm_model.predict(x_in)).tolist()[0][0])



for j in range(wpd*7*4+1):

    x_in = np.array([[store_no, float(j)/4.0, float(ls1.index(max(ls1)))/4.0]])

    x_in = x_in.reshape(x_in.shape[0], 1, x_in.shape[1])

    ls2.append(scalar.inverse_transform(lstm_model.predict(x_in)).tolist()[0][0])

xrange = np.arange(0,wpd*7+0.25,0.25)        

#Plot predicted values of sales 

plt.figure(figsize = (24,6))

plt.subplot(1,2,1)

plt.plot(xrange, ls1)

plt.xlabel('System Hours')

plt.ylabel('Profits')

x = np.argmax(ls1)

y = ls1[x]

plt.scatter(x/4, y, lw = 0, alpha = 0.25, s = 100, c = 'b')

plt.text( x/4, y, 'Max profit at\n{} = {:.4f}'.format(x/4,y), size = '15', color = 'b')

plt.subplot(1,2,2)

plt.plot(xrange, ls2)

plt.xlabel('Manager Hours')

plt.ylabel('Profits')

x = np.argmax(ls2)

y = ls2[x]

plt.scatter(x/4,y, lw = 0, alpha = 0.25, s = 100, c = 'b')

plt.text(x/4, y, 'Max profit at\n{} = {:.4f}'.format(x/4,y), size = '15', color = 'b')

plt.show()

plt.savefig('./optimum_hours.png')
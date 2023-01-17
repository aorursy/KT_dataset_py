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
import numpy as np # linear algebra import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas as pd


%matplotlib inline 

import matplotlib
import matplotlib.pyplot as plt 



df= pd.read_csv('/kaggle/input/covid-latest-git/owid-covid-data.csv')
df


data_in=df.loc[df['location'] == "India"]


data_in["date"]= pd.to_datetime(data_in["date"]) 

data_in.info()
data_in.tail()
data_in.drop(['new_cases_smoothed', 'new_deaths','new_deaths_smoothed','gdp_per_capita','extreme_poverty','cardiovasc_death_rate','diabetes_prevalence','female_smokers','male_smokers','handwashing_facilities','hospital_beds_per_thousand','life_expectancy','human_development_index'], axis=1, inplace=True)
data_in.drop(['iso_code', 'continent','total_cases_per_million','new_cases_per_million','new_cases_smoothed_per_million','total_deaths_per_million','new_tests_smoothed_per_thousand','tests_per_case','positive_rate','population','population_density','tests_units','new_deaths_smoothed_per_million', 'new_deaths_per_million','total_tests','total_tests_per_thousand','new_tests_per_thousand','new_tests_smoothed','stringency_index','median_age','aged_65_older','aged_70_older','new_tests'], axis=1, inplace=True)
data_in
df  = data_in[data_in['date']>'2020-03-18']
df.drop(['location'],axis = 1,inplace = True)
df
df.reset_index(drop=True, inplace=True)
df.drop(['total_cases','total_deaths'],axis =1,inplace = True)
df
df['new_cases'].plot()
plt.figure(figsize=(100,50))
plt.plot(df['date'],df['new_cases'])
plt.xlabel("date")
plt.ylabel("case")

def preprocessing(data , days=14):
    X, Y = [], []
    for i in range(len(data)-days-1):
        a = data[i:(i+days), 0]
        X.append(a)
        Y.append(data[i + days, 0])
    return np.array(X), np.array(Y)
data = df['new_cases']
data = np.array(data, dtype=np.float)
data = data.reshape(len(data),1)
X,Y =preprocessing(data)
Y = Y.reshape(len(Y),1)
Y.shape
from sklearn.model_selection import train_test_split
X_train,X_val,Y_train,Y_val = train_test_split(X, Y, test_size=0.3)
X_train = X_train.reshape(X_train.shape[0] , 1 ,X_train.shape[1])
X_val = X_val.reshape(X_val.shape[0] , 1 ,X_val.shape[1])
Y_train.shape

import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import Sequential
from keras.layers import Dense , BatchNormalization , Dropout , Activation
from keras.layers import LSTM , GRU
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras import optimizers
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(1,14),return_sequences=True))
model.add(LSTM(150, activation='relu'))
model.add(Dense(1,activation='relu'))

print(model.summary())


import math
import pickle
import os
import pandas as pd
import folium 
import numpy as np
import matplotlib
matplotlib.use('nbagg')
import matplotlib.pylab as plt
import seaborn as sns
from matplotlib import rcParams
import plotly as py
import cufflinks
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm_notebook as tqdm
import warnings
import tensorflow as tf
from numpy import array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Input
from tensorflow.keras.layers import BatchNormalization
from dateutil.relativedelta import relativedelta
import datetime
warnings.filterwarnings("ignore")
model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError(),)
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),
             EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
# fit the model
hist=model.fit(X_train,Y_train, epochs=16, batch_size=64, validation_data=(X_val, Y_val), verbose=2, 
               shuffle=True,callbacks=callbacks)
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Epoch vs Loss for Confirmed Cases')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()


X1 = X.reshape(X.shape[0],1,X.shape[1])
pred = model.predict(X1)
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_squared_error,mean_squared_log_error
print(mean_squared_log_error(Y, pred))

def prediction_30(arr,days,Y_ans,model):
    for i in range(0,days):
        pred = model.predict(arr)
        Y_ans = np.append(Y_ans,pred)
        Y_ans.reshape(len(Y_ans),1)
        arr = arr[-1,0,1:]
        arr = np.append(arr,pred)
        arr  =arr.reshape(1,1,len(arr))
    return Y_ans
        
arr = X[-1,:].reshape(1,1,len(X[-1,:]))
Y_pred = prediction_30(arr,30,Y,model)
Y_pred = Y_pred.reshape(len(Y_pred),1)

print("Red - Predicted,  Blue - Actual")

plt.rcParams["figure.figsize"] = (15,7)

plt.plot(Y_pred, 'b')
plt.plot(Y, 'r')
plt.xlabel('Time in days')
plt.ylabel('Daily new Cases')
plt.legend(['Prediction', 'Actual'], loc='best')
plt.title("stacked LSTM for Daily new cases prediction")
plt.grid(True)
plt.show()

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Bidirectional
model_b = Sequential()
model_b.add(Bidirectional(LSTM(100, activation='relu'), input_shape=(1,14)))
model_b.add(Dense(1,activation='relu'))

print(model_b.summary())
model_b.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredLogarithmicError())
callbacks = [ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1, factor=0.6),
             EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model1.h5', monitor='val_loss', save_best_only=True)]
# fit the model
hist=model_b.fit(X_train,Y_train, epochs=8, batch_size=64, validation_data=(X_val, Y_val), verbose=2, 
               shuffle=True,callbacks=callbacks)

arr = X[-1,:].reshape(1,1,len(X[-1,:]))
Y_pred = prediction_30(arr,30,Y,model_b)
Y_pred = Y_pred.reshape(len(Y_pred),1)

print("Red - Predicted,  Blue - Actual")


plt.plot(Y_pred, 'b',label='First')
plt.plot(Y, 'r',label = 'Predicted')
plt.xlabel('Time in days')
plt.ylabel('New Cases')
plt.legend(['Prediction', 'Actual'], loc='best')
plt.title(" BidirectionalLSTM for Total Deaths")
plt.grid(True)
plt.show()

pred = model_b.predict(X1)
from sklearn.metrics import mean_absolute_error,median_absolute_error,mean_squared_error,mean_squared_log_error
print(mean_squared_log_error(Y, pred))


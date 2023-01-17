%matplotlib inline

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import zipfile
import cv2
import tensorflow as tf
import time

from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge,Lasso

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Flatten, Dropout
from keras.optimizers import RMSprop,Adam,SGD
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation
from keras.layers import Dense, LSTM
from keras import initializers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

sns.set()
url='../input/dataquest2020/energy_train.csv'
df_training = pd.read_csv(url)
df_training.head()
url='../input/dataquest2020/energy_test.csv'
df_testing = pd.read_csv(url)
df_testing.head()
df_training['source']='train'
df_testing['source']='test'
print(df_training.head(3))
print(df_testing.head(3))
df = pd.concat([df_training,df_testing],axis=0,ignore_index=True)
df
def secSinceNoon(datTimStr):
    tt = pd.to_datetime(datTimStr).time()
    return (tt.hour * 3600 + tt.minute * 60 + tt.second)/60.0
df['timestamp'] = pd.to_datetime(df['date'])
df['timestamp']
df['SSM'] = df['timestamp'].apply(secSinceNoon)
df['SSM']
df['SSM'].value_counts()
df.isnull().sum().sort_values(ascending=False)
plt.figure(figsize=(10,8))
sns.heatmap(df.isnull(),cbar=False,yticklabels=False,cmap='viridis')
plt.show()
df.describe()
#luminoisity may tend to be categorical as 75%data is 0
#moisture_1 and degree_c2 are +vely skewed but shoot up in 75% - 100% value
#MISSING=> degree_c1 makes sense to fill up with mean or find a categorical mean-way as normal distributn n not much skewness
#watthour- highly +vely skewed due to also very high std
#moisture_2 is okayish but can be issue as 75% -1005 value shoots up
#MISSING=> degree_3 is also good to go with mean_values or find a categorical mean-way as normal distributn n not much skewness
#moisture_3,degree_c4,moisture_4,deg_c4,deg_c5 - good to go..normal distributn
#moisture_5 is a bit +vely skewed, but good to go as it averages out better
#degree_c6 has -ve values, and its +vely skewed b/w 75-100%
#moisture_6 is okay but min-25% gap is large
#deg_c7 is good to go- normal distrbn
#moisture_c7 is okay
#deg_c8,moisture_c8,deg_c9 is perf- normal dstrbn
#MISSING => moisture_c9 is also good to go with mean_values or find a categorical mean-way as normal distributn n not much skewness
#deg_cout has -ve values n check for skewness , mostly +vely skewed b/w 75-100%
#pressure,moisture_out - is god but might be a bit -vely skewed b/w 0-25% -> hihghly dependent asinp
#wind has 0 values , whihc may need to be given equal to min values, but distribn is okay
#clarity- 50% and 75% val is same, high std, might hv issues
#dew_index is +ve skewed b/w 75%-100% , mean is okay so mostly  okay
#random_vals1,2 have the same variable values and distrubn. need to look closely
df.apply(lambda x: len(x.unique()))
#as said luminousity may be categorial
plt.figure(figsize=(10,8))
sns.boxplot(x=df['luminousity'],y=df['WattHour'])
plt.show()
corr = df.corr().round(2)
corr
#cleaning deg_c1,deg_c3,moisture_c9

mean_deg_c1 = df.groupby('SSM').mean()['degree_C1']
mean_deg_c1

miss_bool = df['degree_C1'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
df.loc[miss_bool,'degree_C1'] = df.loc[miss_bool,'SSM'].apply(lambda x: mean_deg_c1[x])
df['degree_C1'].isnull().sum()

mean_deg_c3 = df.groupby('SSM').mean()['degree_C3']
mean_deg_c3

miss_bool = df['degree_C3'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
df.loc[miss_bool,'degree_C3'] = df.loc[miss_bool,'SSM'].apply(lambda x: mean_deg_c3[x])
df['degree_C3'].isnull().sum()

mean_moisture_9 = df.groupby('SSM').mean()['moisture_9']
mean_moisture_9
miss_bool = df['moisture_9'].isnull() 

#Impute data and check #missing values before and after imputation to confirm
df.loc[miss_bool,'moisture_9'] = df.loc[miss_bool,'SSM'].apply(lambda x: mean_moisture_9[x])
df['moisture_9'].isnull().sum()
df.isnull().sum().sort_values(ascending=False)
min_wind_value = df[df['Wind']!=0]['Wind'].min()
min_wind_value
filt = df['Wind'] == 0
df.loc[filt,'Wind'] = min_wind_value
df[df['Wind']==0]
plt.figure(figsize=(10,8))
sns.boxplot(x=df['luminousity'],y=df['WattHour'])
plt.show()
plt.figure(figsize=(20,10))
sns.barplot(x=df['SSM'] , y=df['WattHour'].dropna())
plt.show()
plt.figure(figsize=(20,10))
sns.boxplot(x=df['SSM'] , y=df['WattHour'].dropna())
plt.show()
plt.figure(figsize=(15,12))
sns.heatmap(df.corr().round(2),cmap='coolwarm')
columns= ['degree_C1','degree_C2','degree_C3','degree_C4','degree_C5','degree_C6','degree_C7','degree_C8','degree_C9']

df['mean_degree'] = df.loc[:,columns].sum(axis=1)
df['mean_degree'] = df['mean_degree'] / 9.0
df['mean_degree']
columns= ['moisture_1','moisture_2','moisture_3','moisture_4','moisture_5','moisture_6','moisture_7','moisture_8','moisture_9']

df['mean_moisture'] = df.loc[:,columns].sum(axis=1)
df['mean_moisture'] = df['mean_moisture'] / 9.0
df['mean_moisture']
df.drop(['random_variable_1','random_variable_2','degree_C1','moisture_1','moisture_2','degree_C3','moisture_3',
         'moisture_4','degree_C4','moisture_5','degree_C5','moisture_6','moisture_7','degree_C7','moisture_8','degree_C8'
         ,'moisture_9','degree_C9','moisture_out']
     ,axis=1,inplace=True)
df.drop(['date','timestamp','id'],axis=1,inplace=True)
df
train = df.loc[df['source']=='train']
test = df.loc[df['source']=='test']
train.head()
test.head()
#train.to_csv('train_modified_final.csv', index=False)
#test.to_csv('test_modified_final.csv', index=False)
#df_train = pd.read_csv("/content/train_modified_final.csv")
#df_test = pd.read_csv("/content/test_modified_final.csv")
#df_train
#df_test
test = test.drop(["WattHour", "source"], axis = 1)
train = train.drop(["source"], axis = 1)
Y_train = train.loc[:, "WattHour"]
train = train.drop(["WattHour"], axis = 1)
train
Y_train
test
train = tf.keras.utils.normalize(train)
test = tf.keras.utils.normalize(test)
X_train, X_test, y_train, y_test = train_test_split(train, Y_train, test_size = 0.2)
X_train = X_train.to_numpy()
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
def build_regressor():
    regressor = Sequential()
    regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (X_train.shape[1], 1), kernel_initializer = 'glorot_uniform', activation = 'linear'))
    regressor.add(LSTM(units = 100, kernel_initializer = 'glorot_uniform', return_sequences = True, activation = 'linear'))
    regressor.add(LSTM(units = 100, kernel_initializer = 'glorot_uniform', activation = 'linear'))
    regressor.add(Dense(units = 128, kernel_initializer = 'glorot_uniform', activation = 'linear'))
    regressor.add(Dense(units = 1, activation = 'linear'))
    regressor.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])
    return regressor
regressor = build_regressor()

regressor.fit(X_train, y_train, batch_size = 32, epochs = 20)
X_test = X_test.to_numpy()
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
y_pred = regressor.predict(X_test)
y_pred
test = test.to_numpy()
test = np.reshape(test, (test.shape[0], test.shape[1], 1))
actual_pred = regressor.predict(test)
actual_pred
actual_pred = np.ceil(actual_pred / 10.95) * 10
actual_pred
actual_pred = actual_pred[:, 0].astype('int32')
actual_pred
ID = pd.read_csv("../input/dataquest2020/sample_submission.csv").iloc[:, 0].values
ID
df_final = pd.DataFrame({'id': ID, 'WattHour': actual_pred})
df_final
#df_final.to_csv('submission_lstm_11.csv', index=False)
#from google.colab import files
#files.download("submission_lstm_11.csv")

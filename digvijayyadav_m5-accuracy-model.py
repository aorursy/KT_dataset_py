# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf

import keras

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

filepath= '/kaggle/input/m5-forecasting-accuracy/'

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
subs_df=pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')

sales_train_df=pd.read_csv(filepath + '/sales_train_validation.csv')

sell_df=pd.read_csv(filepath + '/sell_prices.csv')

calender_df=pd.read_csv(filepath + '/calendar.csv')
sales_train_df.head(5)
sales_train_df.isnull().sum().sum()
sales_train_df.describe()
sales_train = pd.get_dummies(sales_train_df)

sales_train.head()
window_len = 10

value = sales_train_df.loc[sales_train_df['id'] == 'HOBBIES_1_001_CA_1_validation'].iloc[:,1919-window_len:1919].head()

#print(value.iloc[0])

value=list(value.iloc[0])

print('---',value)
window_len = 10

pred_values = {}

pred_values1 = {}

import statistics 

for i in sales_train_df['id']:

    value = sales_train_df.loc[sales_train_df['id'] == i].iloc[:,1919-window_len:1919]

    value = list(value.iloc[0])

    #print(i[0])

    for j in range(0,28):

        ans = statistics.mean(value[j:])

        value.append(ans)

        if i in pred_values:

            pred_values[i].append(int(round(ans)))

        else:

            pred_values[i] = [i,int(round(ans))]

    

    for j in range(0,28):

        ans = statistics.mean(value[j+28:])

        value.append(ans)

        ii = i.replace('validation','evaluation')

        if ii in pred_values1:

            pred_values1[ii].append(int(round(ans)))

        else:

            pred_values1[ii] = [ii,int(round(ans))]

            

        

        
#print(pred_values1)
#print(write_csv1)
from statsmodels.tsa.arima_model import ARIMA

from pandas.plotting import autocorrelation_plot

from matplotlib import pyplot



window_len = 50

value = sales_train_df.loc[sales_train_df['id'] == 'HOBBIES_1_001_CA_1_validation'].iloc[:,1919-window_len:1919].head()

#print(value.iloc[0])

value=list(value.iloc[0])



autocorrelation_plot(value)

pyplot.show()
window_len = 50

pred_values = {}

pred_values1 = {}

import statistics 

for i in sales_train_df['id']:

    value = sales_train_df.loc[sales_train_df['id'] == i].iloc[:,1919-window_len:1919]

    value = list(value.iloc[0])

    #print(i[0])

    for j in range(0,28):

        model = ARIMA(value, order=(5,0,7))

        model_fit = model.fit(disp=0)

        ans = model_fit.forecast()

        print(ans,'-sadasdasd-')

        value.append(ans)

        if i in pred_values:

            pred_values[i].append(int(round(ans)))

        else:

            pred_values[i] = [i,int(round(ans))]

    

    for j in range(0,28):

        ans = statistics.mean(value[j+28:])

        value.append(ans)

        ii = i.replace('validation','evaluation')

        if ii in pred_values1:

            pred_values1[ii].append(int(round(ans)))

        else:

            pred_values1[ii] = [ii,int(round(ans))]

            
from sklearn.preprocessing import MinMaxScaler

mms= MinMaxScaler()

mms.fit_transform(categorical_feat)

mms
from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM

from keras.layers import Dropout



# Initialising the RNN

regressor = Sequential()



# Adding the first LSTM layer and some Dropout regularisation

layer_1_units=40

regressor.add(LSTM(units = layer_1_units, return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))

regressor.add(Dropout(0.2))



# Adding a second LSTM layer and some Dropout regularisation

layer_2_units=300

regressor.add(LSTM(units = layer_2_units, return_sequences = True))

regressor.add(Dropout(0.2))



# Adding a third LSTM layer and some Dropout regularisation

layer_3_units=300

regressor.add(LSTM(units = layer_3_units))

regressor.add(Dropout(0.2))



# Adding the output layer

regressor.add(Dense(units = 30490))



# Compiling the RNN

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')



# Fitting the RNN to the Training set

epoch_no=32

batch_size_RNN=44

regressor.fit(X_train, y_train, epochs = epoch_no, batch_size = batch_size_RNN)
write_csv = []

write_csv1 = []

for key, value in pred_values.items():

        write_csv.append(value)

for key, value in pred_values1.items():

        write_csv1.append(value)
import csv

filename = "submission.csv"

with open(filename, 'w') as csvfile:

    csvwriter = csv.writer(csvfile)

    csvwriter.writerows(write_csv)

    csvwriter.writerows(write_csv1)

    
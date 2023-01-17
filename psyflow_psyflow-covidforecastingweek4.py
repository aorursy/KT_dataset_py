# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#import os
#for dirname, _, filenames in os.walk('/kaggle/input'):
#    for filename in filenames:
#        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#importing Libraries
import re
import string
import pandas as pd
from pickle import dump
from pickle import load
from numpy import array
from datetime import datetime, timedelta
from unicodedata import normalize
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from math import sqrt
#Initiallizing RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.optimizers import Adam
#importing dataset
dataset = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/train.csv")
testset = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/test.csv")
sample = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-4/submission.csv")
#replacing null State will ""
index = dataset[dataset["Province_State"].isnull()==True].index
dataset.loc[index, "Province_State"] = ""
dataset["location"] = dataset["Country_Region"] + " | " + dataset["Province_State"]


index = testset[testset["Province_State"].isnull()==True].index
testset.loc[index, "Province_State"] = ""
testset["location"] = testset["Country_Region"] + " | " + testset["Province_State"]
#dataset Year, Month and day
dataset["year"] = dataset["Date"].apply(lambda x:x.split("-")[0])
dataset["month"] = dataset["Date"].apply(lambda x:x.split("-")[1])
dataset["day"] = dataset["Date"].apply(lambda x:x.split("-")[2])


testset["year"] = testset["Date"].apply(lambda x:x.split("-")[0])
testset["month"] = testset["Date"].apply(lambda x:x.split("-")[1])
testset["day"] = testset["Date"].apply(lambda x:x.split("-")[2])
#location which had more than 0 Confirmed Cases on 2020-01-22
nonzero_location = dataset[(dataset["ConfirmedCases"]>0) & (dataset["Date"]=="2020-01-22")]["location"].unique()


dataset["days_from_start"] = 0
index = dataset[dataset["location"].isin(nonzero_location)].index
dataset.loc[index, "days_from_start"] = dataset.loc[index, "Date"].apply(lambda x:(datetime.strptime(x, "%Y-%m-%d") - datetime.strptime("2020-01-22", "%Y-%m-%d")).days)


location_startdate = {}
for location in dataset[dataset["location"].isin(nonzero_location)==False]["location"].unique():
    location_startdate[location] = (dataset[(dataset["location"]==location)&(dataset["ConfirmedCases"]==0)]["Date"].iloc[-1])

for location in dataset[dataset["location"].isin(nonzero_location)==False]["location"].unique():
    index = (dataset[(dataset["location"]==location) & (dataset["Date"]>location_startdate[location])].index)
    dataset.loc[index, "days_from_start"] = dataset.loc[index, "Date"].apply(lambda x:(datetime.strptime(x, "%Y-%m-%d") - datetime.strptime(location_startdate[location], "%Y-%m-%d")).days)

toit = dataset.copy()
location_map = {}
i = 0
for location in dataset["location"].unique():
    location_map[location] = i
    i = i + 1
dataset["location"] = dataset["location"].apply(lambda x:location_map[x])
window = 8
X = []
y = []
for location in dataset["location"].unique():
    temp = dataset[dataset["location"] == location].reset_index()
    for row in temp.loc[0:len(temp)-window-2, :].index:
        X.append(temp.loc[row:row+window-1, ["location", "days_from_start", "ConfirmedCases"]].values)
        y.append(temp.loc[row+window,"ConfirmedCases"])
        
y = np.array(y)
X = np.array(X)
window = 8
P = []
q = []
for location in dataset["location"].unique():
    temp = dataset[dataset["location"] == location].reset_index()
    for row in temp.loc[0:len(temp)-window-2, :].index:
        P.append(temp.loc[row:row+window-1, ["location", "days_from_start", "Fatalities"]].values)
        q.append(temp.loc[row+window,"Fatalities"])
        
q = np.array(q)
P = np.array(P)
n = int(0.7 * len(X))
X_train = X[:n]
X_test = X[n:]
y_train = y[:n]
y_test = y[n:]

X_train .shape, X_test .shape, y_train .shape, y_test .shape

n = int(0.7 * len(P))
P_train = P[:n]
P_test = P[n:]
q_train = q[:n]
q_test = q[n:]

P_train.shape, P_test.shape, q_train.shape, q_test.shape
regressor1 = Sequential()

#Adding first LSTM layer and some Dropout regularization to avoid Overfitting
regressor1.add(LSTM(units=200,return_sequences=True,input_shape=(X_train.shape[1],3)))
regressor1.add(Dropout(0.2))

#Adding second and third LSTM layer
regressor1.add(LSTM(units=200,return_sequences=True))
regressor1.add(Dropout(0.2))
#regressor.add(LSTM(units=200,return_sequences=True))
#regressor.add(Dropout(0.2))
#Adding 1RNN Layer
regressor1.add(LSTM(units=200,return_sequences=False))
regressor1.add(Dropout(0.2))


#Adding 1NN Layer
regressor1.add(Dense(units=100))
#Adding output layer
regressor1.add(Dense(units=1))

#Compiling RNN
regressor1.compile(optimizer='rmsprop',loss='mae')#Adam(lr=0.003) #

#Fitting RNN to the training set
regressor1.fit(X_train,y_train,epochs=10,batch_size=10)
regressor2 = Sequential()

#Adding first LSTM layer and some Dropout regularization to avoid Overfitting
regressor2.add(LSTM(units=200,return_sequences=True,input_shape=(X_train.shape[1],3)))
regressor2.add(Dropout(0.2))

#Adding second and third LSTM layer
regressor2.add(LSTM(units=200,return_sequences=True))
regressor2.add(Dropout(0.2))
#regressor.add(LSTM(units=200,return_sequences=True))
#regressor.add(Dropout(0.2))
#Adding 1RNN Layer
regressor2.add(LSTM(units=200,return_sequences=False))
regressor2.add(Dropout(0.2))


#Adding 1NN Layer
regressor2.add(Dense(units=100))
#Adding output layer
regressor2.add(Dense(units=1))

#Compiling RNN
regressor2.compile(optimizer='rmsprop',loss='mae')#Adam(lr=0.003) #

#Fitting RNN to the training set
regressor2.fit(P_train,q_train,epochs=10,batch_size=10)
result1 = []
for location in dataset["location"].unique():
    temp = (dataset[dataset["location"] == location].tail(8)).reset_index()
    temp = temp[["location", "days_from_start", "ConfirmedCases"]].values
    result1.extend(temp.tolist())
    for days in range(60):
        prediction = regressor1.predict(np.reshape(temp,(1,temp.shape[0], temp.shape[1])))
        last_record = np.resize(np.array([location, temp[7][1]+1, int(prediction[0][0])]), (1,temp.shape[1]))
        temp = (np.append(temp,  last_record, axis=0))[1:]
        result1.extend([[location, temp[7][1], int(prediction[0][0])]])
result1 = pd.DataFrame(result1)
result1.columns = (["location", "days_from_start", "ConfirmedCases"])
result2 = []
for location in dataset["location"].unique():
    temp = (dataset[dataset["location"] == location].tail(8)).reset_index()
    temp = temp[["location", "days_from_start", "Fatalities"]].values
    result2.extend(temp.tolist())
    for days in range(60):
        prediction = regressor2.predict(np.reshape(temp,(1,temp.shape[0], temp.shape[1])))
        last_record = np.resize(np.array([location, temp[7][1]+1, int(prediction[0][0])]), (1,temp.shape[1]))
        temp = (np.append(temp,  last_record, axis=0))[1:]
        result2.extend([[location, temp[7][1], int(prediction[0][0])]])
result2 = pd.DataFrame(result2)
result2.columns = (["location", "days_from_start", "Fatalities"])
result1["Country_Region"] = result1["location"].apply(lambda x:list(location_map.keys())[list(location_map.values()).index(x)].split(" | ")[0])
result1["Province_State"] = result1["location"].apply(lambda x:list(location_map.keys())[list(location_map.values()).index(x)].split(" | ")[1])
result1["location"] = result1["location"].apply(lambda x:list(location_map.keys())[list(location_map.values()).index(x)])
result1["Date"] = ""
for location in result1["location"].unique():
    index  = result1[result1["location"]==location].index
    result1.loc[index, "start_date"] = result1.loc[index, "location"].apply(lambda x:location_startdate[x] if x in list(location_startdate.keys()) else "2020-01-22")

    
result2["Country_Region"] = result2["location"].apply(lambda x:list(location_map.keys())[list(location_map.values()).index(x)].split(" | ")[0])
result2["Province_State"] = result2["location"].apply(lambda x:list(location_map.keys())[list(location_map.values()).index(x)].split(" | ")[1])
result2["location"] = result2["location"].apply(lambda x:list(location_map.keys())[list(location_map.values()).index(x)])
result2["Date"] = ""
for location in result2["location"].unique():
    index  = result2[result2["location"]==location].index
    result2.loc[index, "start_date"] = result2.loc[index, "location"].apply(lambda x:location_startdate[x] if x in list(location_startdate.keys()) else "2020-01-22")

for row in result1.index:
    days_from_start = result1.loc[row, "days_from_start"]
    start_date = result1.loc[row, "start_date"]
    result1.loc[row, "Date"] = datetime.strftime((datetime.strptime(start_date, "%Y-%m-%d").date()+timedelta(days=days_from_start)), "%Y-%m-%d")


for row in result2.index:
    days_from_start = result2.loc[row, "days_from_start"]
    start_date = result2.loc[row, "start_date"]
    result2.loc[row, "Date"] = datetime.strftime((datetime.strptime(start_date, "%Y-%m-%d").date()+timedelta(days=days_from_start)), "%Y-%m-%d")

final1 = pd.merge(result1, testset, right_on = ["location", "Date"], left_on = ["location", "Date"])
final2 = pd.merge(result2, testset, right_on = ["location", "Date"], left_on = ["location", "Date"])
missing_ids = list(set(testset.ForecastId) - set(final1.ForecastId))
sample["ConfirmedCases"] = sample["ForecastId"].apply(lambda x:final1[final1["ForecastId"] == x]["ConfirmedCases"].values[0] if x not in missing_ids else 0.0)
sample["Fatalities"] = sample["ForecastId"].apply(lambda x:final2[final2["ForecastId"] == x]["Fatalities"].values[0] if x not in missing_ids else 0.0)


sample.to_csv("submission.csv", index=False)
sample.head(10)


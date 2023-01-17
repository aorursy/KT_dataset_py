#%matplotlib notebook

import pandas as pd

pd.set_option('display.max_rows', 500)

pd.set_option('display.max_columns', 500)

pd.set_option('display.width', 1000)

import numpy as np

import matplotlib.pyplot as plt

import datetime as dt

import statistics



train_df = pd.read_csv("../input/bike-sharing-demand/train.csv", parse_dates=True)



train_df.head(2)
train_df['Day'] = pd.DatetimeIndex(train_df['datetime']).day

train_df['Month'] = pd.DatetimeIndex(train_df['datetime']).month

train_df['Year'] = pd.DatetimeIndex(train_df['datetime']).year

train_df['HH'] = pd.DatetimeIndex(train_df['datetime']).hour



train_df.head(1)
train_df = train_df.drop(['datetime', 'casual', 'registered'], axis=1)



train_df.head(1)
from sklearn.preprocessing import LabelEncoder

enc = LabelEncoder()



features_to_convert = ["Year"]



for i in features_to_convert:

    train_df[i] = enc.fit_transform(train_df[i].astype('str'))



train_df.head(1)
cols = train_df.columns.tolist()

cols
cols = ['season',

 'holiday',

 'workingday',

 'weather',

 'temp',

 'atemp',

 'humidity',

 'windspeed',

 'HH',

 'Day',

 'Month',

 'Year',

 'count']



train_df = train_df[cols]

train_df.head(2)
from sklearn import preprocessing



# Get column names first

names = train_df.columns



# Create the Scaler object

scaler = preprocessing.MinMaxScaler()



# Fit your data on the scaler object

scaled_df = scaler.fit_transform(train_df)

scaled_df = pd.DataFrame(scaled_df, columns=names)



train_df = scaled_df

train_df.head(2)
X = train_df.iloc[:,0:12]  #independent columns

y = train_df.iloc[:,12]    #target column



from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.15, random_state=101)



print(X_train.shape, X_test.shape)
#Best n estimator search

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

import statistics



for i in range(1,18):

    model = RandomForestRegressor(n_estimators=i, random_state=101)

    model.fit(X_train, Y_train)



    y_predict = model.predict(X_test)



    kfoldscore = cross_val_score(model, X_train, Y_train, cv=5)



    #print("kFold Scores: {}".format(kfoldscore.round(4)))

    print("Estimator: " + str(i) + "\tkFold Score Mean: " + str(statistics.mean(kfoldscore).round(4)))
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_val_score

import statistics





model = RandomForestRegressor(n_estimators=7)

model.fit(X_train, Y_train)



y_predict = model.predict(X_test)



kfoldscore = cross_val_score(model, X_train, Y_train, cv=5)



print("kFold Scores: {}".format(kfoldscore.round(4)))

print("kFold Score Mean: ", statistics.mean(kfoldscore).round(4))

Y = Y_test.values

A = np.resize(Y,(1633,1)).round(6)



A[100:105]
B = np.resize(y_predict,(1633,1)).round(6)

B[100:105]
ind = []

for i in range(A.size):

    ind.append(i)



accuracy = pd.DataFrame({'True': A.flatten(), 'Predict': B.flatten()}, index=ind)

accuracy.head()
acc = []



for i in range(accuracy['True'].size):

    if accuracy['True'][i] < accuracy['Predict'][i]:

        result = accuracy['True'][i] / accuracy['Predict'][i]

        acc.append(result.round(2))

    else: 

        result = accuracy['Predict'][i] / accuracy['True'][i] 

        acc.append(result.round(2))



accuracy['Accuracy'] = acc



accuracy
plt.figure(figsize=(10, 7))

plt.xlabel('True Value')

plt.ylabel('Predictions')

plt.scatter(accuracy['True'], accuracy['Predict'], alpha=0.5, s=55, c='r')
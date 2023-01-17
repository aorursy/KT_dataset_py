import numpy as np 

import pandas as pd 

import os

from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR

from catboost import CatBoostRegressor, Pool

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt



#print(os.listdir("../input"))
train=pd.read_csv("../input/train.csv", nrows=6000000,dtype={'acoustic_data':np.int16,'time_to_failure':np.float64})

train.head(10)
#Lets plot the data to see and understand the data columns and our problem .

#We will use a small subset of dataset for understanding the pattern ,since the data is large



train_acoustic_df = train['acoustic_data'].values[::100]

train_time_to_failure_df = train['time_to_failure'].values[::100]



fig, ax1 = plt.subplots(figsize=(10,10))

plt.title('Acoustic data and Time to Failure')

plt.plot(train_acoustic_df, color='r')

ax1.set_ylabel('acoustic data', color='r')

plt.legend(['acoustic data'], loc=(0.01, 0.9))



ax2 = ax1.twinx()

plt.plot(train_time_to_failure_df, color='b')

ax2.set_ylabel('time to failure', color='b')

plt.legend(['time to failure'], loc=(0.01, 0.8))



plt.grid(True)



    



def gen_features(X):

    fe = []

    fe.append(X.mean())

    fe.append(X.std())

    fe.append(X.min())

    fe.append(X.max())

    fe.append(X.kurtosis())

    fe.append(X.skew())

    fe.append(np.quantile(X,0.01))

    fe.append(np.quantile(X,0.05))

    fe.append(np.quantile(X,0.95))

    fe.append(np.quantile(X,0.99))

    fe.append(np.abs(X).max())

    fe.append(np.abs(X).mean())

    fe.append(np.abs(X).std())

    return pd.Series(fe)
#Lets read the training set again now in chunks and append features 

train = pd.read_csv('../input/train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})



X_train = pd.DataFrame()

y_train = pd.Series()

for df in train:

    ch = gen_features(df['acoustic_data'])

    X_train = X_train.append(ch, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))
X_train.head(10) #Let's check the training dataframe
submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id') #Taking the segment id from sample_submission file
#Applying Feature Engineering on test data files

X_test = pd.DataFrame()

for seg_id in submission.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    ch = gen_features(seg['acoustic_data'])

    X_test = X_test.append(ch, ignore_index=True)
X_test.head(10) #Lets check the testing dataframe
#Catboost regressor model 

"""       

#Catboost Regressor model



train_pool = Pool(X_train, y_train)



m = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')

m.fit(X_train, y_train, silent=True)

m.best_score_

"""
#Scale Train Data

scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = pd.DataFrame(scaler.transform(X_train))

X_train_scaled.head(10)
#We will also scale the train data

X_test_scaled = pd.DataFrame(scaler.transform(X_test))

X_test_scaled.head(10)
parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],

               'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]



reg1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_absolute_error')

reg1.fit(X_train_scaled, y_train.values.flatten())

submission.time_to_failure = reg1.predict(X_test_scaled) 

submission
submission.to_csv('submission.csv',index=True)
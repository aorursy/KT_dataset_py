# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('/kaggle/input/train.csv',nrows=6000000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})

train.head()
#visualize 1% of samples data, first 100 datapoints

train_ad_sample_df = train['acoustic_data'].values[::100]

train_ttf_sample_df = train['time_to_failure'].values[::100]
import matplotlib.pyplot as plt

#function for plotting based on both features

def plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df, title="Acoustic data and time to failure: 1% sampled data"):

    fig, ax1 = plt.subplots(figsize=(12, 8))

    plt.title(title)

    plt.plot(train_ad_sample_df, color='r')

    ax1.set_ylabel('acoustic data', color='r')

    plt.legend(['acoustic data'], loc=(0.01, 0.95))

    ax2 = ax1.twinx()

    plt.plot(train_ttf_sample_df, color='b')

    ax2.set_ylabel('time to failure', color='b')

    plt.legend(['time to failure'], loc=(0.01, 0.9))

    plt.grid(True)



plot_acc_ttf_data(train_ad_sample_df, train_ttf_sample_df)

del train_ad_sample_df

del train_ttf_sample_df
# Step 4 - Feature Engineering and signifiance of these statistical features



#lets create a function to generate some statistical features based on the training data

def gen_features(X):

    strain = []

    strain.append(X.mean())

    strain.append(X.std())

    strain.append(X.min())

    strain.append(X.max())

    strain.append(X.kurtosis())

    strain.append(X.skew())

    strain.append(np.quantile(X,0.01))

    strain.append(np.quantile(X,0.05))

    strain.append(np.quantile(X,0.95))

    strain.append(np.quantile(X,0.99))

    strain.append(np.abs(X).max())

    strain.append(np.abs(X).mean())

    strain.append(np.abs(X).std())

    return pd.Series(strain)
train = pd.read_csv('/kaggle/input/train.csv', iterator=True, chunksize=150_000, dtype={'acoustic_data': np.int16, 'time_to_failure': np.float64})



X_train = pd.DataFrame()

y_train = pd.Series()

for df in train:

    ch = gen_features(df['acoustic_data'])

    X_train = X_train.append(ch, ignore_index=True)

    y_train = y_train.append(pd.Series(df['time_to_failure'].values[-1]))
X_train.describe()
#Model #1 - Catboost

from catboost import CatBoostRegressor, Pool

train_pool = Pool(X_train, y_train)

m = CatBoostRegressor(iterations=10000, loss_function='MAE', boosting_type='Ordered')

m.fit(X_train, y_train, silent=True)

m.best_score_
#Model #2 - Support Vector Machine w/ RBF + Grid Search



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import GridSearchCV

from sklearn.svm import NuSVR, SVR





scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)



parameters = [{'gamma': [0.001, 0.005, 0.01, 0.02, 0.05, 0.1],

               'C': [0.1, 0.2, 0.25, 0.5, 1, 1.5, 2]}]

               #'nu': [0.75, 0.8, 0.85, 0.9, 0.95, 0.97]}]



reg1 = GridSearchCV(SVR(kernel='rbf', tol=0.01), parameters, cv=5, scoring='neg_mean_absolute_error')

reg1.fit(X_train_scaled, y_train.values.flatten())

y_pred1 = reg1.predict(X_train_scaled)



print("Best CV score: {:.4f}".format(reg1.best_score_))

print(reg1.best_params_)
# LSTM Model

from sklearn.preprocessing import StandardScaler

from keras.models import Sequential

from keras.layers import Dense

from keras.layers import LSTM



scaler = StandardScaler()

scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)



# Reshape to correct dimensions

X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0],X_train_scaled.shape[1],1)



# Model

model = Sequential()

model.add(LSTM(50,  input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))

model.add(Dense(1))

model.compile(loss='mae', optimizer='adam')



# Fit network

history = model.fit(X_train_scaled, 

                    y_train, 

                    epochs=15,

                    batch_size=64,

                    verbose=0)



model.summary()
# Evaluate model

from sklearn.metrics import mean_absolute_error

    

y_pred = model.predict(X_train_scaled)

mae = mean_absolute_error(y_train, y_pred)

print('%.5f' % mae)
# submission format

submission = pd.read_csv('../input/sample_submission.csv', index_col='seg_id')

X_test = pd.DataFrame()



# prepare test data

for seg_id in submission.index:

    seg = pd.read_csv('../input/test/' + seg_id + '.csv')

    ch = gen_features(seg['acoustic_data'])

    X_test = X_test.append(ch, ignore_index=True)



X_test = scaler.transform(X_test)
# model of choice here

X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

y_hat = model.predict(X_test)
# write submission file

submission['time_to_failure'] = y_hat

submission.to_csv('submission.csv')
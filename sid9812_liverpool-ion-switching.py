import sys 

import numpy as np # linear algebra

from scipy.stats import randint

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv), data manipulation as in SQL

import matplotlib.pyplot as plt # this is used for the plot the graph 

import seaborn as sns # used for plot interactive graph. 

from sklearn.model_selection import train_test_split # to split the data into two parts

# from sklearn.cross_validation import KFold # use for cross validation

from sklearn.preprocessing import StandardScaler # for normalization

from sklearn.preprocessing import MinMaxScaler

from sklearn.pipeline import Pipeline # pipeline making

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn import metrics # for the check the error and accuracy of the model

from sklearn.metrics import mean_squared_error,r2_score

# from sklearn.linear_model import LinearRegression

from sklearn.linear_model import ARDRegression, LinearRegression

## for Deep-learing:

import keras

from keras.layers import Dense

from keras.models import Sequential

from keras.utils import to_categorical

from keras.optimizers import SGD 

from keras.callbacks import EarlyStopping

from keras.utils import np_utils

import itertools

from keras.layers import LSTM

from keras.layers.convolutional import Conv1D

from keras.layers.convolutional import MaxPooling1D

from keras.layers import Dropout


sns.set(style='whitegrid', palette='muted', font_scale=1.5)



# rcParams['figure.figsize'] = 16, 10



RANDOM_SEED = 42



np.random.seed(RANDOM_SEED)

# tf.random.set_seed(RANDOM_SEED)





train = pd.read_csv("../input/liverpool-ion-switching/train.csv")

test = pd.read_csv("../input/liverpool-ion-switching/test.csv")



print(train.tail())

test.head()
test = test.values

test_X = test
values = train.values



n_train_time = 5000000

train = values[:n_train_time, :]

# val = values[n_train_time:, :]



train_X, train_y = train[:, 1:-1], train[:, -1]

# val_X, val_y = val[:, :-1], val[:, -1]

print(train_X.shape, train_y.shape, val_X.shape, val_y.shape) 

# clf = ARDRegression(compute_score=True)

clf = ARDRegression(n_iter=300, tol=0.001, alpha_1=1e-06, 

                    alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=True, 

                    threshold_lambda=10000.0, fit_intercept=True, normalize=False, 

                    copy_X=True, verbose=False)

train_X
batch_size = 10000

#train and prediction



# for i in range(5000000//batch_size)-1:

#     train_X = train_X[i*batch_size:(i+1)*batch_size]

#     train_y = train_y[i*batch_size:(i+1)*batch_size]

#     clf.fit(train_X,train_y)

test_X.shape

# test_X = test_X[:,1]

test_X = test_X.reshape(-1,1)
y_pred = clf.predict(test_X)

print(y_pred)

for i in range(len(y_pred)):

    y_pred[i] = round(y_pred[i])
y_pred
output_file = "submission.csv"

with open(output_file, 'w') as f :

    f.write('time,open_channels\n')

    for i in range(len(y_pred)) :

        f.write("".join([str(test['time'][i]),',',str(y_pred[i]),'\n']))
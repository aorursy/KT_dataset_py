## Work In Progress

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn import metrics

from sklearn.model_selection import train_test_split



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X_train = pd.read_csv('../input/X_train.csv')

Y_train=pd.read_csv('../input/y_train.csv')

X_train.head()

print(X_train.shape,Y_train.shape)

plt.figure(figsize=(10,4))

Y_train.surface.value_counts().plot(kind='bar');

plt.show()
## group features based on series_id, picked this from one of the Kernels , added new features

columns=['orientation_X','orientation_Y','orientation_Z','orientation_W','angular_velocity_X','angular_velocity_Y','angular_velocity_Z','linear_acceleration_X','linear_acceleration_Y','linear_acceleration_Z']

def feature_data(X):

    data_train=pd.DataFrame()

    for col in columns:

        data_train[col+'_mean'] = X.groupby(['series_id'])[col].mean()

        data_train[col+'_median'] = X.groupby(['series_id'])[col].median()

        data_train[col+'_max'] = X.groupby(['series_id'])[col].max()

        data_train[col+'_min'] = X.groupby(['series_id'])[col].min()

        data_train[col+'_std'] = X.groupby(['series_id'])[col].std()

        data_train[col+'_var'] = X.groupby(['series_id'])[col].var()

        data_train[col + '_range'] = data_train[col + '_max'] - data_train[col + '_min']

        data_train[col + '_maxtoMin'] = data_train[col + '_max'] / data_train[col + '_min']

        data_train[col + '_mean_abs_chg'] = X.groupby(['series_id'])[col].apply(lambda x: np.mean(np.abs(np.diff(x))))

        data_train[col + '_abs_median_chg'] = X.groupby(['series_id'])[col].apply(lambda x: np.median(np.abs(np.diff(x))))

        data_train[col + '_abs_std_chg'] = X.groupby(['series_id'])[col].apply(lambda x: np.std(np.abs(x)))

        data_train[col + '_abs_max'] = X.groupby(['series_id'])[col].apply(lambda x: np.max(np.abs(x)))

        data_train[col + '_abs_min'] = X.groupby(['series_id'])[col].apply(lambda x: np.min(np.abs(x)))

        data_train[col + '_abs_avg'] = (data_train[col + '_abs_min'] + data_train[col + '_abs_max'])/2

       

    return data_train



data_train=feature_data(X_train)

data_train.head()



data_train, data_valid, Y_train, y_valid = train_test_split(data_train, Y_train, test_size=0.3) # 70% training and 30% test

## First attempt with Logistic Regression

model = LogisticRegression(solver = 'lbfgs',max_iter=500,multi_class='multinomial')

model.fit(data_train,Y_train['surface'])


Y_pred_train=model.predict(data_train)

accuracy = metrics.accuracy_score(Y_train['surface'], Y_pred_train)

print('Accuracy training: {:.2f}'.format(accuracy))



Y_pred_valid=model.predict(data_valid)

accuracy = metrics.accuracy_score(y_valid['surface'], Y_pred_valid)

print('Accuracy validation: {:.2f}'.format(accuracy))
## Simple Logistic Regression has an accuracy of 61% on training set.

from sklearn.neural_network import MLPClassifier

## Next let's try using a neural network.



nn_model = MLPClassifier(hidden_layer_sizes=(200, 100,50,), activation='tanh', solver='adam', alpha=0.01,learning_rate_init=0.001, max_iter=800)

nn_model.fit(data_train,Y_train['surface'])

Y_pred_train=nn_model.predict(data_train)

accuracy = metrics.accuracy_score(Y_train['surface'], Y_pred_train)

print('Accuracy: {:.2f}'.format(accuracy))



Y_pred_valid=nn_model.predict(data_valid)

accuracy = metrics.accuracy_score(y_valid['surface'], Y_pred_valid)

print('Accuracy validation: {:.2f}'.format(accuracy))
## using neural netowrks is leading to overfitting



## Using Random Forest

from sklearn.ensemble import RandomForestClassifier



forest = RandomForestClassifier(n_estimators=600)



forest.fit(data_train,Y_train['surface'])

Y_pred_train=forest.predict(data_train)

accuracy = metrics.accuracy_score(Y_train['surface'], Y_pred_train)

print('Accuracy training: {:.2f}'.format(accuracy))



Y_pred_valid=forest.predict(data_valid)

accuracy = metrics.accuracy_score(y_valid['surface'], Y_pred_valid)

print('Accuracy validation: {:.2f}'.format(accuracy))
## predictions for test data



X_test=pd.read_csv('../input/X_test.csv')

data_test=feature_data(X_test)

Y_pred_test=forest.predict(data_test)

result = pd.DataFrame()

result['series_id'] = data_test.index.values

result['surface'] = Y_pred_test

result.head()

result.to_csv('submission.csv',sep=',',index=False)
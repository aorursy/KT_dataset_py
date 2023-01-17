# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import pandas as pd,matplotlib.pyplot as plt,seaborn as sns,numpy as np

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler



data = pd.read_excel('train_ml.xlsx')

data.shape

data.head()

prediction_var = list(data.columns.drop(['cnt','casual']))

print(prediction_var)

from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size = 0.05,random_state = 0)

print(train.shape)

print(test.shape)

X_train = train[prediction_var]

Y_train = train['cnt']

X_test = test[prediction_var] 

Y_test = test['cnt'] 

from sklearn import ensemble

clf = ensemble.GradientBoostingRegressor(alpha=0.99,n_estimators = 1500, max_depth = 5, min_samples_split = 3,learning_rate = 0.055)

clf.fit(X_train,Y_train)

print(clf.score(X_test,Y_test))

predics = clf.predict(X_test)

mse = mean_squared_error(Y_test,predics)

print('MSE is {}'.format(mse))

test = pd.read_excel('test_ml.xlsx')

instant = test['instant']

test = test.drop('casual',axis=1)

predic = clf.predict(test)

predic = pd.DataFrame(abs(predic.astype('int')))

instant = pd.DataFrame(instant)

final = pd.concat([instant,predic],axis=1)

final.columns = ['instant','cnt']

final.to_csv('finalavi_kaggle.csv')

# Any results you write to the current directory are saved as output.
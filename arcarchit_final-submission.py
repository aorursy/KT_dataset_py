# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import math

from sklearn.metrics import r2_score

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
trainDF = pd.read_csv('../input/train.csv');

testDF = pd.read_csv('../input/test.csv');
all_data = pd.concat((trainDF.loc[:,'MSSubClass':'SaleCondition'],

                      testDF.loc[:,'MSSubClass':'SaleCondition']))

all_data = pd.get_dummies(all_data)

#filling NA's with the mean of the column:

all_data = all_data.fillna(all_data.mean())

#creating matrices for sklearn:

X_train = all_data[:trainDF.shape[0]]

X_test = all_data[trainDF.shape[0]:]

y = trainDF.SalePrice
from math import log
i = [1 for j in range(X_train.shape[0])]

i = np.array(i)

i = np.expand_dims(i, axis=1)

np_X_train = np.hstack((i, X_train.as_matrix()))



i = [1 for j in range(X_test.shape[0])]

i = np.array(i)

i = np.expand_dims(i, axis=1)

np_X_test = np.hstack((i, X_test.as_matrix()))
temp = np.log(y)

np_y = temp.values
from numpy import linalg
I = np.identity(np_X_train.shape[1])

temp0 = np_X_train.T.dot(np_X_train) + 10*I

temp1 = linalg.inv(temp0)

parameter = temp1.dot(np_X_train.T).dot(np_y)

Ypred = np_X_test.dot(parameter)
y2 = np.exp(Ypred)
#Sample submission

submission = pd.DataFrame({ 'Id': testDF['Id'],

                           'SalePrice': y2 })

submission.to_csv("f2.csv", index=False)
y2
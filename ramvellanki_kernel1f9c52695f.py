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
from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import mean_squared_error, r2_score
def gender_converter(s):

    s = s.lower()

    if s == "male":

        return 1

    if s == "female":

        return 0

    return -1
traindata = pd.read_csv('/kaggle/input/titanic/train.csv',converters={"Sex":gender_converter})

testdata = pd.read_csv('/kaggle/input/titanic/test.csv',converters={"Sex":gender_converter})
traindata["Age"].fillna(0, inplace=True)

testdata["Age"].fillna(0, inplace=True)

testdata["Fare"].fillna(0, inplace=True)
traindata.head()
testdata.head()
#split data into x & y for train

y_train = traindata['Survived']

x_train = traindata.drop(columns=['Survived','Name','Ticket','Cabin','Embarked'])



x_test = testdata.drop(columns=['Name','Ticket','Cabin','Embarked'])
x_train.head()
x_test[x_test['Fare'].isnull()]
y_pred = regressor.predict(x_test)
print('Coefficients: \n', regressor.coef_)
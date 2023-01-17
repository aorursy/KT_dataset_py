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
import pandas as pd

sample_submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
null_list = train.isnull().sum()
for i, j in null_list.items():

    if j>0:

        print(i,j)
data = train

encoding_li = []

remove_column = []

idx=0

for i, j in null_list.items():

    if j>600:

        remove_column.append(i)

    if data[i].dtype in ['int64', 'float64']:

        data[i] = data[i].fillna(data[i].mean())

    elif data[i].dtype == 'object':

        data[i] = data[i].fillna(data[i].mode()[0])

        encoding_li.append([*zip(data[i].unique(), range(len(data[i].unique())))])

        for k in encoding_li[idx]:

            data[i] = data[i].replace(k[0], k[1])

        idx+=1

    else:

        print(data[i].dtype)

for i in remove_column:

    data = data.drop(i, axis=1)

#     train[i] = (train[i]-min(train[i]))/(max(train[i])-min(train[i]))
len(encoding_li)
train = train.drop('Id', axis=1)
Y = train['SalePrice']

X = train.drop('SalePrice' ,axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.01, random_state=41)
from sklearn import linear_model

reg = linear_model.LinearRegression()

reg.fit(X_train, Y_train)
predicted_Y = reg.predict(test)
Y_test
predicted_Y
from sklearn.metrics import mean_squared_error

mean_squared_error(Y_test, predicted_Y)
fil=open('submission.csv','w')

fil.write('Id,SalePrice\n')

i=1

for each in predicted_Y:

   fil.write('%d,%d\n'%(i+1460,each))

   i=i+1

fil.close()
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.cross_validation import cross_val_score

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#store the train data

train = pd.read_csv("../input/train.csv")



#store the test data

test = pd.read_csv("../input/test.csv")
#sneak peek at correlation matix

train.corr()
train.shape

#train.columns.values

#train[:].describe()



train.columns.values




#data types of individual columns

train.dtypes
train.loc[10:20]
##desscribe your dataframe

train.info()



train.count()
#train.columns.values

train.dtypes == 'int64'
train['SalePrice'].plot.hist(bins = 35)
#are there null values present?

train.isnull().values.any()



#null values for every column

np.round(((np.array([train.iloc[:,i].isnull().sum() for i in range(0,train.shape[1])])/ train.shape[0]))*100)

train.MSSubClass = train.MSSubClass.astype("object")



train.dtypes
train.corr()
[x for x in train.columns]

cols = ['OverallQual','GrLivArea','GarageCars','TotalBsmtSF']

X = train[cols]

lm = LinearRegression()

lm.fit(X,train['SalePrice'])
lm.coef_

lm.intercept_



scores = cross_val_score(LinearRegression(), X, train['SalePrice'], cv=10, scoring='r2')

np.mean(scores)
rf = RandomForestRegressor()

rf.fit(X,train['SalePrice'])
scores = cross_val_score(RandomForestRegressor(), X, train['SalePrice'], cv=10, scoring='r2')
np.mean(scores)
test.TotalBsmtSF.plot.hist(bins = 30)

test.TotalBsmtSF.median()

a = test[(test.GarageCars.isnull())].index

test.loc[a,'GarageCars'] =2

a = test[(test.TotalBsmtSF.isnull())].index

test.TotalBsmtSF.mean()

test.loc[a,'TotalBsmtSF'] = test.TotalBsmtSF.mean()
predicted = rf.predict(test[cols])

submissions = pd.DataFrame([test.Id,predicted])

#submissions.to_csv("rf.csv", index = False)
a = submissions.T

a.columns = ['Id','SalePrice']

a.Id = a.Id.astype('Int64')

a.to_csv("rf.csv", index = False)

a.Id.dtype = 'Int64'
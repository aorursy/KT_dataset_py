# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV , KFold , cross_val_score

from sklearn.metrics import accuracy_score

from sklearn.metrics import mean_squared_error

from sklearn import linear_model

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsRegressor

from sklearn.ensemble import RandomForestRegressor



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/diamonds.csv')
df.head(10)

df = df.drop('Unnamed: 0',axis=1)
df.shape
df.notnull().count()
df.info()
df.describe()
sns.scatterplot(x="price", y="z", data=df)
sns.scatterplot(x="price", y="y", data=df)
sns.scatterplot(x="price", y="x", data=df)
sns.scatterplot(x="price", y="table", data=df)
sns.scatterplot(x="price", y="depth", data=df)
sns.scatterplot(x="price", y="carat", data=df)
sns.catplot(x="clarity", y="price", kind="box", data=df)
sns.catplot(x="cut", y="price", kind="box", data=df)
sns.catplot(x="color", y="price", kind="box", data=df)
df = df[(df.x!=0)]

df = df[(df.y!=0)]

df = df[(df.z!=0)]
df.shape
df.describe()
df.isnull().sum()
p = df.hist(figsize = (25,25),bins=100)
df_encoded =  pd.get_dummies(df)

df_encoded.head()
df_standard_scale = StandardScaler()

temp =  pd.DataFrame(df_standard_scale.fit_transform(df_encoded[['carat','depth','x','y','z','table']]),columns=['carat','depth','x','y','z','table'],index=df_encoded.index)
df_encoded[['carat','depth','x','y','z','table']] = temp[['carat','depth','x','y','z','table']]
df_encoded.head()
df_encoded.corr()
plt.figure(figsize=(25,25))

sns.heatmap(df_encoded.corr(),cmap='coolwarm',annot=True)
y = df_encoded['price']

x = df_encoded.drop('price',axis=1)
print('shape of x and y is respectively {} and {}'.format(x.shape,y.shape))
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.2, random_state=0) 
print('shape of xtrain is {}'.format(xtrain.shape))

print('shape of xtest is {}'.format(xtest.shape))

print('shape of ytrain is {}'.format(ytrain.shape))

print('shape of ytest is {}'.format(ytest.shape))
lr = linear_model.LinearRegression()

lr.fit(xtrain,ytrain)

crossval = cross_val_score(estimator = lr, X = xtrain, y = ytrain, cv = 5,verbose = 1)

y_pred = lr.predict(xtest)

print('Linear Regression')

print('cross validation score')

print(crossval)

print('mean absolute error = {}'.format(mean_absolute_error(ytest, y_pred)))

print('mean squared error = {}'.format(mean_squared_error(ytest, y_pred)))

print('root mean squared error = {}'.format(np.sqrt(mean_squared_error(ytest, y_pred))))

print('r2 score = {}'.format(r2_score(ytest, y_pred)))

print('accuracy = {}'.format(lr.score(xtest,ytest)))

print('intercept = {}'.format(lr.intercept_))

coeff_df = pd.DataFrame(lr.coef_, x.columns, columns=['Coefficient'])  

coeff_df

pred_df = pd.DataFrame({'Actual':ytest, 'Predicted':y_pred})

pred_df.head(30)
rf = RandomForestRegressor()

rf.fit(xtrain,ytrain)

crossval = cross_val_score(estimator = rf, X = xtrain, y = ytrain, cv = 5,verbose = 1)

y_pred = rf.predict(xtest)

print('Random Forest Regressor')

print('cross validation score')

print(crossval)

print('mean absolute error = {}'.format(mean_absolute_error(ytest, y_pred)))

print('mean squared error = {}'.format(mean_squared_error(ytest, y_pred)))

print('root mean squared error = {}'.format(np.sqrt(mean_squared_error(ytest, y_pred))))

print('r2 score = {}'.format(r2_score(ytest, y_pred)))

print('accuracy = {}'.format(rf.score(xtest,ytest)))

kn = KNeighborsRegressor()

kn.fit(xtrain,ytrain)

crossval = cross_val_score(estimator = kn, X = xtrain, y = ytrain, cv = 5,verbose = 1)

y_pred = kn.predict(xtest)

print('KNeighbors Regressor')

print('cross validation score')

print(crossval)

print('mean absolute error = {}'.format(mean_absolute_error(ytest, y_pred)))

print('mean squared error = {}'.format(mean_squared_error(ytest, y_pred)))

print('root mean squared error = {}'.format(np.sqrt(mean_squared_error(ytest, y_pred))))

print('r2 score = {}'.format(r2_score(ytest, y_pred)))

print('accuracy = {}'.format(kn.score(xtest,ytest)))
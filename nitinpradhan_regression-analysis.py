# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

## data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/diamonds/diamonds.csv")
data = data.drop(["Unnamed: 0"],axis=1)
data.head()
data.describe()
data.keys()
data.head()
data.info()
data.isnull().sum()
sns.set(palette="Spectral", style = 'darkgrid' ,font_scale = 2.0, color_codes=True)

sns.pairplot(data)
plt.figure(figsize=(16,16))

sns.heatmap(data.corr(), annot=True,cmap='RdYlGn',square=True ,linewidths=.5)
print("rows with x == 0: {} ".format((data.x==0).sum()))

print("rows with y == 0: {} ".format((data.y==0).sum()))

print("rows with z == 0: {} ".format((data.z==0).sum()))

print("rows with depth == 0: {} ".format((data.depth==0).sum()))
data[['x','y','z']] = data[['x','y','z']].replace(0,np.NaN)
data.isnull().sum()
data.dropna(inplace=True)
data.isnull().sum()
data.head()
cut_count = sns.factorplot(x='cut', data=data , kind='count',aspect=2.5 )
cut_price = sns.factorplot(x='cut', y = 'price', data=data ,aspect=2.5 )
clarity_count = sns.factorplot(x='clarity', data=data , kind='count',aspect=2.5 )
clarity_price = sns.factorplot(x='clarity', y = 'price', data=data ,aspect=2.5 )
data.head()
data_hist = data.hist(figsize = (20,20),bins=150)
p = sns.factorplot(x='cut', y='price', data=data, kind='box' ,aspect=2.5 )
data_clarity = sns.factorplot(x='clarity', y='price', data=data, kind='box' ,aspect=2.5)
data_encode = pd.get_dummies(data)

data_encode.head()
col= data_encode.columns

final_data = pd.DataFrame(data_encode,columns= col)

final_data.head()
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

diamond =  pd.DataFrame(sc_X.fit_transform(final_data[['carat','depth','x','y','z','table']]),

                           columns=['carat','depth','x','y','z','table'],

                           index=final_data.index)
diamond.head()
real_diamond = final_data.copy(deep=True)

real_diamond[['carat','depth','x','y','z','table']] = diamond[['carat','depth','x','y','z','table']]
real_diamond.head()
plt.figure(figsize=(35,24))

p=sns.heatmap(real_diamond.corr(), annot=True,cmap='coolwarm',linewidths=.5,annot_kws={'size':16},fmt=".1f")

plt.tight_layout
X = real_diamond.drop(["price"],axis=1)

Y = real_diamond.price
print(X)
from sklearn.model_selection import train_test_split

train_X, test_X, train_Y, test_Y = train_test_split(X, Y,random_state = 2,test_size=0.3)
from sklearn import linear_model

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score





reg = linear_model.LinearRegression()

reg.fit(train_X,train_Y)

y_pred = reg.predict(test_X)

Accuracy = reg.score(test_X,test_Y)*100

print("Accuracy : {}%".format(Accuracy))

MAE = mean_absolute_error(test_Y,y_pred)

print("Mean absolute error: {}".format(MAE))

MSE = mean_squared_error(test_Y,y_pred)

print("Mean squared error: {}".format(MSE))

Score = r2_score(test_Y,y_pred)

print('Score: {}'.format(Score))

n=test_X.shape[0]

p=test_X.shape[1] - 1



adj_rsquared = 1 - (1 - Score) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))
rig_reg = linear_model.Ridge()

rig_reg.fit(train_X,train_Y)

y_pred = rig_reg.predict(test_X)

Accuracy = reg.score(test_X,test_Y)*100

print("Accuracy : {}%".format(Accuracy))

MAE = mean_absolute_error(test_Y,y_pred)

print("Mean absolute error: {}".format(MAE))

MSE = mean_squared_error(test_Y,y_pred)

print("Mean squared error: {}".format(MSE))

Score = r2_score(test_Y,y_pred)

print('Score: {}'.format(Score))

n=test_X.shape[0]

p=test_X.shape[1] - 1



adj_rsquared = 1 - (1 - Score) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))
las_reg = linear_model.Lasso()

las_reg.fit(train_X,train_Y)

y_pred = las_reg.predict(test_X)

Accuracy = reg.score(test_X,test_Y)*100

print("Accuracy : {}%".format(Accuracy))

MAE = mean_absolute_error(test_Y,y_pred)

print("Mean absolute error: {}".format(MAE))

MSE = mean_squared_error(test_Y,y_pred)

print("Mean squared error: {}".format(MSE))

Score = r2_score(test_Y,y_pred)

print('Score: {}'.format(Score))

n=test_X.shape[0]

p=test_X.shape[1] - 1



adj_rsquared = 1 - (1 - Score) * ((n - 1)/(n-p-1))

print('Adjusted R Squared: {}'.format(adj_rsquared))
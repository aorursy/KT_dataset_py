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
boston=pd.read_csv('/kaggle/input/bostoncsv/Boston.csv')
from matplotlib import pyplot as plt

import seaborn as sns
boston.head()
boston.isna().sum()
boston.isnull().sum()
boston.duplicated().sum()
boston=boston.drop(['Unnamed: 0'],axis=1)
boston.hist(figsize=(16, 20),xlabelsize=12, ylabelsize=12)

cor=boston.corr()

plt.figure(figsize=(20,10))

sns.heatmap(cor[(cor >= 0.5) | (cor <= -0.4)],annot=True,cmap='Blues')
boston.describe()
sns.distplot(boston['medv'], color='g', bins=100, hist_kws={'alpha': 0.4});
boston.corr()['medv'][:-1]
for i in range(0, len(boston.columns), 5):

    sns.pairplot(data=boston,

                x_vars=boston.columns[i:i+5],

                y_vars=['medv'])
boston.columns
y=boston['medv']
fig, ax = plt.subplots(round(len(boston.columns) / 3), 3, figsize = (18, 12))

for i, ax in enumerate(fig.axes):

    if i < len(boston.columns) - 1:

        sns.regplot(x=boston.columns[i],y='medv',data=boston[boston.columns], ax=ax)
boston=boston.drop('medv',axis=1)
boston.info()
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.metrics import r2_score

from sklearn.model_selection import cross_val_score, cross_val_predict

from sklearn import metrics

x_train,x_test,y_train,y_test=train_test_split(boston,y,test_size=0.3)
y_train.shape
reg=LinearRegression()

reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
plt.scatter(y_test, y_pred, c = 'red') 
from sklearn.metrics import mean_squared_error 

mse = mean_squared_error(y_test, y_pred) 

print("Mean Square Error : ", mse) 
df1 = pd.DataFrame({'Actual': y_test, 'Predicted':y_pred})

df2 = df1.head(10)

df2
from sklearn import metrics

from sklearn.metrics import r2_score

print('MAE', metrics.mean_absolute_error(y_test, y_pred))

print('MSE', metrics.mean_squared_error(y_test, y_pred))

print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

print('R squared error', r2_score(y_test, y_pred))
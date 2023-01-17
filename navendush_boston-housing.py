# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
boston_housing = pd.read_csv('/kaggle/input/boston-housing/boston_housing.csv')
boston_housing.head()
boston_housing.info()
boston_housing.describe()
boston_housing.columns
sns.pairplot(boston_housing,palette='coolwarm')
sns.distplot(boston_housing['medv'],kde = False, bins = 30)
boston_housing.corr()
sns.heatmap(boston_housing,cbar='coolwarm')
sns.heatmap(boston_housing.isnull(),cbar='coolwarm',yticklabels=False)
boston_housing.columns
X = boston_housing[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax','ptratio', 'black', 'lstat']]

y = boston_housing['medv']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
print(lm.intercept_)
lm.coef_
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])

coeff_df
plt.scatter(y_test,predictions)
sns.distplot((y_test-predictions),kde = False, bins = 40)
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))

print('MSE:', metrics.mean_squared_error(y_test, predictions))

print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))
metrics.explained_variance_score(y_test, predictions)
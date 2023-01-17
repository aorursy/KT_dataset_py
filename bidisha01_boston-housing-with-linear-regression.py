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
boston_data = pd.read_csv('/kaggle/input/boston-dataset/boston_data.csv')
boston_data.head()
boston_data.describe()
boston_data.info()
boston_data.isnull().sum()
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(50,50))

sns.pairplot(boston_data)
sns.distplot(boston_data['medv'])
plt.figure(figsize=(50,50))

sns.heatmap(boston_data.corr(), annot=True)
boston_data.columns
X=boston_data[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax','ptratio', 'black', 'lstat']]

y=boston_data[['medv']]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.10)
print(X_train.shape)

print(X_test.shape)
from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train,y_train)
lm.intercept_
coeff_df = pd.DataFrame(lm.coef_.reshape(13,1),X.columns,columns=['Coefficient'])

coeff_df
y_pred = lm.predict(X_test)

plt.scatter(y_test,y_pred)
sns.distplot(y_pred-y_test)
from sklearn import metrics
metrics.mean_absolute_error(y_test,y_pred)
metrics.mean_squared_error(y_test,y_pred)
np.sqrt(metrics.mean_squared_error(y_test,y_pred))
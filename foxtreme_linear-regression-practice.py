# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

%matplotlib inline

import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/USA_Housing.csv')

df.head()
df.info()
df.describe()
df.columns
sns.pairplot(df)
sns.distplot(df['Price'])
sns.heatmap(df.corr(),annot=True)
df.columns
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

       'Avg. Area Number of Bedrooms', 'Area Population']]
y= df['Price']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, random_state=101)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X_train,  y_train)
print(lm.intercept_)
lm.coef_
X_train.columns
pd.DataFrame(lm.coef_, X.columns, columns=['Coeff'])
from sklearn.datasets import load_boston
boston = load_boston()
boston.keys()
print(boston['DESCR'])
predictions = lm.predict(X_test)
predictions
y_test
plt.scatter(y_test,predictions)
sns.distplot(y_test-predictions)
from sklearn import metrics

metrics.mean_absolute_error(y_test, predictions)
metrics.mean_squared_error(y_test, predictions)
np.sqrt(metrics.mean_squared_error(y_test, predictions))
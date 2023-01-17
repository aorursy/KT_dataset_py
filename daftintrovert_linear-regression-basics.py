# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
file = '../input/USA_Housing.csv'

df = pd.read_csv(file)

df.head()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

import seaborn as sns
%matplotlib inline
df.info()
df.describe()
sns.pairplot(df)
sns.distplot(df['Price'])
df.corr()
sns.heatmap(df.corr(),annot = True)
df.columns
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',

       'Avg. Area Number of Bedrooms', 'Area Population']]

X.head()
y = df[['Price']]

y.head()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.4,random_state = 101)
lm = LinearRegression()

lm
lm.fit(X_train,y_train)
lm.intercept_
lm.coef_
X.columns
predictions = lm.predict(X_test)
predictions
y_test.head()
plt.scatter(y_test,predictions) # it should be straight line 
sns.distplot((y_test-predictions)) #hist of residuals
from sklearn import metrics
metrics.mean_absolute_error(y_test,predictions)
metrics.mean_squared_error(y_test,predictions)
np.sqrt(metrics.mean_squared_error(y_test,predictions))
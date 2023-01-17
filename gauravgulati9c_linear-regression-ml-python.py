# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from matplotlib import style

style.use('fivethirtyeight')

import sklearn

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn import metrics



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")



data.describe()
print(data.shape)

print(test.shape)
data = data.dropna()

test = test.dropna()

print(data.shape)

print(test.shape)

data.describe()
clf = LinearRegression()
X = data[['x']]

y = data['y']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=101)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

print(predictions)
plt.scatter(y_test, predictions)
sns.distplot((y_test-predictions))
print(metrics.mean_absolute_error(y_test,predictions))
print(metrics.mean_squared_error(y_test,predictions))
print(np.sqrt(metrics.mean_squared_error(y_test,predictions)))
# Simple example using Linear Regression



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import sklearn

from sklearn.linear_model import LinearRegression



from sklearn import metrics

import pandas_profiling as pp



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")





train.describe()
train = train.dropna()

test = test.dropna()

print(train.shape)

print(test.shape)
train.describe()
train.corr()
train.plot.scatter(x='x',y='y',title="train")

test.plot.scatter(x='x',y='y',title="test")
train.hist(bins=40)
clf = LinearRegression()
type(train.x)
clf.fit(train.x.values.reshape(-1, 1), train.y.values.reshape(-1, 1))
predictions = clf.predict(test.x.values.reshape(-1, 1))

print(predictions)
plt.scatter(test.x, predictions)
y_test=test.y.values.reshape(-1,1)

plt.plot(y_test-predictions)
print(metrics.r2_score(y_test,predictions))
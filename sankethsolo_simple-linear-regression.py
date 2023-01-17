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
import pandas as pd

test=pd.read_csv("/kaggle/input/random-linear-regression/test.csv")

train=pd.read_csv("/kaggle/input/random-linear-regression/train.csv")


train.describe()

train.info()

train=train.dropna()

test=test.dropna()

train.corr()

#x and y are highly correlated

import numpy as np

xtrain=train.values.reshape(-1,1)

ytrain=train.values.reshape(-1,1)

xtest=test.values.reshape(-1,1)

ytest=test.values.reshape(-1,1)

import matplotlib.pyplot as plt

plt.scatter(xtrain,ytrain)

plt.title("relation between x and y")

plt.xlabel("x train")

plt.ylabel("y train")

plt.show()

plt.subplot(1,2,1)

plt.hist(xtrain)

plt.title("histogram of x")

plt.subplot(1,2,2)

plt.hist(ytrain)

plt.title("histogram of y")

plt.subplot(1,2,1)

plt.boxplot(xtrain)

plt.title("box plot of x")

plt.subplot(1,2,2)

plt.boxplot(ytrain)

plt.title("box plot of y")

from sklearn import linear_model

model=linear_model.LinearRegression()
model.fit(xtrain,ytrain)
print(model.coef_)

print(model.intercept_)
pr=model.predict(xtest)

print("the accuracy of model is",(model.score(ytest,pr))*100)

from sklearn.metrics import r2_score

print("the r2 score is ",r2_score(ytest,pr))
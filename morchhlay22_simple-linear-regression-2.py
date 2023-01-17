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

import numpy as np

import matplotlib.pyplot as plt

import math 

from sklearn import linear_model

import warnings

warnings.filterwarnings('ignore')
first_data = pd.read_csv("../input/random-linear-regression/train.csv")

second_data =pd.read_csv("../input/random-linear-regression/test.csv")
train_set = first_data.dropna()

test_set = second_data.dropna()
print(train_set.head())

print(test_set.head())
X = train_set[['x']].as_matrix()

y= train_set[['y']].as_matrix()
Xtest  = test_set[['x']].as_matrix()

ytest = test_set[['y']].as_matrix()
plt.figure(figsize=([8,6]))

plt.title("let see the realation b/w x and y of traning set")

plt.scatter(X,y,s=5,c="black",marker="*")

plt.xlabel("traning_set_x")

plt.ylabel("traning_set_y")

plt.show()
regression = linear_model.LinearRegression()
regression.fit(X,y)
regression.score(X,y)
math.sqrt(regression.score(X,y))
predic = regression.predict(X)
plt.figure(figsize=([8,6]))

plt.title("scatter b/w predicted and actual values")

plt.scatter(predic,y,s=5,c="red")

plt.xlabel("actual")

plt.ylabel("predicted")

plt.show()
predict = regression.predict(Xtest)
plt.figure(figsize=([8,6]))

plt.title("scatter b/w predicted and actual values in test set")

plt.scatter(ytest,predict,s=5,c="cyan")

plt.xlabel("test values")

plt.ylabel("predicted values")

plt.show()
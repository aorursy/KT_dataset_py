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

from matplotlib import style



style.use('fivethirtyeight')



df = pd.read_csv('/kaggle/input/headbrain/headbrain.csv')

df.head()
X = df['Head Size(cm^3)'].values

y = df['Brain Weight(grams)'].values
mean_X = np.mean(X)

mean_y = np.mean(y)

numi = 0

deno = 0

for i in range(len(X)):

    numi += (X[i] - mean_X)*(y[i]-mean_y)

    deno += (X[i]-mean_X)**2



m = numi/deno



# print(numi, deno)

print('Slope = ', m)

# print(mean_X, mean_y)
intercept = mean_y - (m*mean_X)

y_line = m*X + intercept

print('y-intercept:',intercept)
import seaborn as sns

plt.plot(X, y_line, label = 'Regression Line')

sns.scatterplot(X, y, color = 'r', label = 'Actual points')

plt.xlabel('Head Size')

plt.ylabel('Brain wt')
# now predict the value of y for given value of x

y_predicted = []

for i in range(len(X)):

    y_pred = m*X[i] + intercept

    y_predicted.append(y_pred)
# check how good our model is by using R Squared Method

num_s = 0

deno_s = 0

for i in range(len(X)):

    num_s += (y_predicted[i]-y[i])**2

    deno_s += (y[i] - mean_y)**2

r_sq = 1 - num_s/deno_s

print(r_sq)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.model_selection import train_test_split



X = X.reshape(-1, 1)

y = y.reshape(-1, 1)

# on partial data

X_train, X_valid, y_train, y_valid = train_test_split(X,y,train_size = 0.8, random_state = 42)



clf = LinearRegression()

clf.fit(X_train,y_train)

prediction = clf.predict(X_valid)



r2_score = clf.score(X_train, y_train)

print(r2_score)

# On full data

clf1 = LinearRegression()

clf.fit(X,y)

prediction = clf.predict(X_valid)



r2_score = clf.score(X, y)

print(r2_score)
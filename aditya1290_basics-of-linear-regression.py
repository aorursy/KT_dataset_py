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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')
train_data.info()
train_data.dropna(axis=0,how='any',inplace = True)
train_data.describe()
plt.scatter(x='x',y='y',data = train_data,color = 'green')

plt.show()
x = train_data.iloc[:,:1]

y = train_data.iloc[:,:-1]
from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(x,y)
test_data.info()
x_test = test_data.iloc[:,:1]

y_test = test_data.iloc[:,:-1]
y_predict = lr.predict(x_test)
from sklearn.metrics import mean_squared_error,r2_score

mse = mean_squared_error(y_test, y_predict)

r2 = r2_score(y_test, y_predict)
print(mse)

print(r2)
plt.figure(figsize=(10,7))

plt.scatter(x = x_test, y = y_predict, color = 'green',label='Predicted')

plt.scatter(x = x_test, y = y_test, color = 'red',label = 'Original')

plt.legend()

plt.show()
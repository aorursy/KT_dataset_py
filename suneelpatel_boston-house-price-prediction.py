# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Import libraries necessary for this project

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']

bos1 = pd.read_csv('../input/housing.csv', header=None, delimiter=r"\s+", names=column_names)
bos1.shape
x = bos1.iloc[:,0:13]

y = bos1["MEDV"]
#code to plot correlation



#librarry to establish correlation

import seaborn as sns

names = []

#creating a correlation matrix

correlations = bos1.corr()

sns.heatmap(correlations,square = True, cmap = "YlGnBu")

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.show()
from sklearn.model_selection import train_test_split

#testing data size is of 33% of entire data

x_train, x_test, y_train, y_test =train_test_split(x,y, test_size = 0.33, random_state =5)
from sklearn.linear_model import LinearRegression

#fitting our model to train and test

lm = LinearRegression()

model = lm.fit(x_train,y_train)
pred_y = lm.predict(x_test)
pd.DataFrame({"Actual": y_test, "Predict": pred_y}).head()
plt.scatter(y_test,pred_y)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
import sklearn

mse = sklearn.metrics.mean_squared_error(y_test, pred_y)

print(mse)
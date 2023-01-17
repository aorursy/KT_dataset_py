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
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/Boston.csv')
dataset.head()
dataset.shape #the shape return as 506 rows and 15 columns from this we need identify dependent and independent variable
X = dataset.drop(['Unnamed: 0','medv'],axis=1)

y = dataset['medv']
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3)

#the above two lines are added import train_test functionality from sklearn, the second line is added to split

#(contd) the dataset to train and test with size =0.3, i.e 70% for train and 30% for test
#import the linear model

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train,y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test,y_pred)
plt.scatter(y_test,y_pred)

plt.xlabel("Prices: $Y_i$")

plt.ylabel("Predicted prices: $\hat{Y}_i$")

plt.title("Predicted vs Prices: $Y_i$ vs $\hat{Y}_i$")
print(mse)
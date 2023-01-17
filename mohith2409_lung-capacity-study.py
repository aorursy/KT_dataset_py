# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
df = pd.read_excel("../input/dataset/Processed_Lungcap.xls")
df.head()
df.isnull().any()
df.info()
sns.pairplot(df)
from IPython.display import Image

Image("../input/dashboard/Capture1.PNG")
df.corr()
X = df[['Height']]

y = df['Lungcap']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, random_state = 2)

X_train.head()
regression_model = LinearRegression()

regression_model.fit(X_train,y_train)
intercept = regression_model.intercept_

print("The intercept for our model is {}".format(intercept))
regression_model.coef_
regression_model.score(X_test,y_test)
rmse = np.sqrt(np.mean((regression_model.predict(X_test)-y_test)**2))

rmse
y_pred = regression_model.predict(X_test)
R_square = 1 - (np.sum((y_test-y_pred)**2)/(np.sum((y_test-np.mean(y_test))**2)))

R_square
np.corrcoef(y_test,y_pred)
plt.scatter(y_test,y_pred)
model2 = LinearRegression()

model2.fit(X,y)
intercept = model2.intercept_

print("The intercept for our model is {}".format(intercept))
model2.coef_
model2.score(X_test,y_test)
y_pred2 = model2.predict(X)
rmse = np.sqrt(np.mean((model2.predict(X)-y)**2))

rmse
R_square = 1 - (np.sum((y-y_pred2)**2)/(np.sum((y-np.mean(y))**2)))

R_square
np.corrcoef(y,y_pred2)
df.plot.scatter(x='Height',y='Lungcap')

plt.plot(X,y_pred2,c='red')
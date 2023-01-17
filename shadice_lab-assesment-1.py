# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import random

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
X1 = [random.randrange(1,101, 1) for _ in range(10)]

X2= [random.randrange(1,201, 1) for _ in range(10)]

Y= [random.randrange(1,6, 1) for _ in range(10)]
print(X1)

print(X2)

print(Y)


df = pd.DataFrame({'X1': X1,'X2': X2,'Y': Y})             

print (df)
#Correlation between X1 and Y

df['X1'].corr(df['Y'])
#Correlation between X2 and Y

df['X2'].corr(df['Y'])
# scatter plot illustrating the relationship between X1 and Y and X2 and Y

sp = df.plot.scatter(x='X1', y='Y', c='Red')

sp2 = df.plot.scatter(x='X2', y='Y', c='Blue')
import matplotlib.pyplot as plt 

import seaborn as sns

%matplotlib inline

import sklearn

import scipy.stats as stats
X_data = df[['X1','X2']]

Y_data = df['Y']

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test= train_test_split(X_data,Y_data, test_size=0.50)
from sklearn import linear_model
reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
reg.coef_
X_train.columns
print('Regression Coefficient')

pd.DataFrame(reg.coef_, index=X_train.columns, columns=['Coefficient'])
reg.intercept_
test_predicted = reg.predict(X_test)
test_predicted
#plot illustrating the Residuals

from sklearn.decomposition import PCA

#pca is the principle amonent analysis
pca = PCA(n_components=1)
pca.fit(df[X_train.columns])
b= reg.intercept_
X_reduced = pca.transform(X_test)
X_test



X_reduced
plt.scatter(X_reduced, y_test, color='orange')

plt.ylabel("Y")

plt.xlabel("Reduced_X")
plt.scatter(X_reduced, y_test, color='purple')

plt.scatter(X_reduced, test_predicted, color='yellow')

plt.plot(X_reduced, test_predicted, 'b')

#plt.plot(X_reduced, y_test, color='red')



plt.xticks(())

plt.yticks(())

plt.ylabel("Y")

plt.xlabel("Reduced_X")



plt.show()
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from math import sqrt
print('R^2 = %.2f' %r2_score(y_test, test_predicted) )
# Mean Absolute Error of your model refers to the mean of the absolute values of each prediction error

print('Mean Absolute Error: %.2f' % mean_absolute_error(y_test, test_predicted))
rmse= sqrt(mean_squared_error(y_test, test_predicted))

print('Root Mean Absolute Error: %.2f' %rmse)
print('Mean Squared Error: %.2f' % mean_squared_error(y_test, test_predicted))
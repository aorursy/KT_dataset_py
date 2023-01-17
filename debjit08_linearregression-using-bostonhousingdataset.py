import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
from sklearn.datasets import load_boston

boston = load_boston()

bos = pd.DataFrame(boston.data)

bos.head()
bos.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']

bos.head()
bos['MEDV'] = boston.target
bos.head()



#Fetching more information about the dataset using the info() function.

#bos.info()



#Fetching more information about the dataset using the info() function.

#bos.describe();



#check for null values if any present in the dataset.

#bos.isnull().sum()
sns.distplot(bos['MEDV'])

plt.show()
#this shows the relationships between all the features present in the dataset

sns.pairplot(bos)
#Corelation Matrix

corelation = bos.corr();

sns.heatmap(corelation,square= True)

plt.yticks(rotation=0)

plt.xticks(rotation=90)

plt.show()
sns.lmplot(x = 'RM', y = 'MEDV', data = bos)
#X is the independent variable and y is the dependent variable.

X = bos[['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX','PTRATIO', 'B', 'LSTAT']]

y = bos['MEDV']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 10)
#import LinearRegression from sklearn library

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)
prediction = lm.predict(X_test)

plt.scatter(y_test, prediction)
df1 = pd.DataFrame({'Actual': y_test, 'Predicted':prediction})

df1.head(10)
df1.head(10).plot(kind='bar')
from sklearn import metrics

from sklearn.metrics import r2_score

print('MAE', metrics.mean_absolute_error(y_test, prediction))

print('MSE', metrics.mean_squared_error(y_test, prediction))

print('RMSE', np.sqrt(metrics.mean_squared_error(y_test, prediction)))

print('R squared error', r2_score(y_test, prediction))
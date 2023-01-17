# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd



# Any results you write to the current directory are saved as output.

train = pd.read_csv("../input/train.csv")

train.info()

train.head()

test = pd.read_csv('../input/test.csv')

test.head()
train.describe(include='all')
import matplotlib.pyplot as plt

Y = train["revenue"].values

X = train["budget"].values

plt.scatter(X, Y, color = 'red')

plt.title('Budget vs Revenud)')

plt.xlabel('Budget')

plt.ylabel('Revenue')

plt.show()
Y = train["revenue"].values

X = train["popularity"].values

plt.scatter(X, Y, color = 'red')

plt.title('Popularity vs Revenue')

plt.xlabel('Popularity')

plt.ylabel('Revenue')

plt.show()
Y = train["revenue"].values

X = train["runtime"].values

plt.scatter(X, Y, color = 'red')

plt.title('Run Time vs Revenue')

plt.xlabel('Run Time')

plt.ylabel('Revenue')

plt.show()


Y = train["revenue"].values

X = train["original_language"].values

plt.scatter(X, Y, color = 'red')

plt.title('Original Language vs Revenue')

plt.xlabel('Original Language')

plt.ylabel('Revenue')

plt.show()
import pandas as pd

train=train.dropna(how='any')

X = train.iloc[:, [2,9,14]].values

Y = train["revenue"].values



import statsmodels.formula.api as sm

regressor_OLS = sm.OLS(endog = Y, exog = X).fit()

regressor_OLS.summary()

# taking care of missing values

train['runtime'] = train['runtime'].fillna(method='ffill')

X = train[['runtime', 'budget','popularity']]

y = train.revenue



#splitting the data into training and validation to check validity of the model



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=1)
#Linear Model

from sklearn.linear_model import LinearRegression, Ridge, Lasso

def rmsle(y,y0): return np.sqrt(np.mean(np.square(np.log1p(y)-np.log1p(y0)))) 

reg = LinearRegression()

lin_model = reg.fit(X_train, y_train)

y_pred = reg.predict(X_val)

print('RMSLE score for linear model is {}'.format(rmsle(y_val, y_pred)))



#Applyting the model on test data and submission

test = pd.read_csv('../input/test.csv')

test['runtime'] = test.runtime.fillna(method='ffill')

test.status = pd.get_dummies(test.status)

X_test = test[['runtime','popularity','budget']]

pred1 = reg.predict(X_test)



#Submission

sub1 = pd.read_csv('../input/sample_submission.csv')

sub1['revenue'] = pred1

sub1.to_csv('lin_model_sub.csv',index=False)
from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors = 5)

knn_model = knn.fit(X_train, y_train)

knn_y_pred = knn.predict(X_val)

print('RMSLE score for k-NN model is {}'.format(rmsle(y_val, knn_y_pred)))

pred2 = knn.predict(X_test)



#Submission

sub2 = pd.read_csv('../input/sample_submission.csv')

sub2['revenue'] = pred2

sub2.to_csv('knn_model_sub.csv',index=False)
X = train[['runtime', 'budget','popularity','original_language']]

Y = train.iloc[:, 22].values

X = X.iloc[:, [0, 1, 2, 3]].values

# Encoding Categorical Data

# Encoding the Independent Variable 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X = LabelEncoder()

X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])

X = onehotencoder.fit_transform(X).toarray()

#splitting the data into training and validation to check validity of the model



from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.1, random_state=1)

#Applyting the model on test data and submission

test = pd.read_csv('../input/test.csv')

test['runtime'] = test.runtime.fillna(method='ffill')

test.status = pd.get_dummies(test.status)

X_test = test[['runtime', 'budget','popularity','original_language']]

X_test = X_test.iloc[:, [0, 1, 2, 3]].values

X_test[:, 3] = labelencoder_X.fit_transform(X_test[:, 3])

onehotencoder = OneHotEncoder(categorical_features = [3])

X_test = onehotencoder.fit_transform(X_test).toarray()



# Fitting Multiple Linear Regression to the Training Set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)



#predecting the Test set Results

Y_pred = regressor.predict(X_val)

print('RMSLE score for Multi linear model is {}'.format(rmsle(y_val, y_pred)))

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn import metrics

from sklearn.metrics import mean_squared_error

from math import sqrt

import matplotlib.pyplot as plt

import numpy as np 

import seaborn as sns




df= pd.read_csv("../input/winequality-red.csv")

correlation=df.corr()

fig=plt.subplots(figsize=(10,10))

sns.heatmap(correlation,vmax=1,square=True, annot=True,cmap='Reds')
def get_features(correlation_threshold):

    abs_corrs = correlation.abs()

    high_correlations= abs_corrs[abs_corrs > correlation_threshold].index.values.tolist()

    return high_correlations
features=get_features(0.05)

print(features)

x=df[features]

y=df['quality']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3)
regressor=LinearRegression()

regressor.fit(x_train,y_train)

print(regressor.coef_)

train_pred=regressor.predict(x_train)

print(train_pred)

test_pred =regressor.predict(x_test)

print(test_pred)


train_rmse = mean_squared_error(train_pred, y_train) ** 0.5

train_rmse

test_rmse=(mean_squared_error(test_pred, y_test) ** 0.5)

test_rmse
train_rmse=(mean_squared_error(train_pred, y_train) ** 0.5)

train_rmse
predicted_data=np.round(test_pred)

print(predicted_data)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, test_pred))

print('Mean Squared Error:', metrics.mean_squared_error(y_test, test_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, test_pred)))
coeffecients = pd.DataFrame(regressor.coef_,features) 

coeffecients.columns = ['Coeffecient'] 

print(coeffecients)


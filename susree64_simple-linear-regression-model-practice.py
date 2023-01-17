# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/train.csv')
test = pd.read_csv("../input/test.csv")

sns.regplot(x = test['x'], y = test['y'], data = test)
sns.regplot(x = train['x'], y = train['y'], data = train)
train.isna().any()

X = train[['x']]
y = train['y']
# replacing the na value with the mean of the column 
y.fillna(y.mean(), inplace = True)
# Import linear regression from sklearn
from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(X,y)
print(lm.intercept_)
print(lm.coef_)
predictions = lm.predict(test[['x']])
#check to see the predicted values and the actual values if they are matching
y_test = test['y']
plt.scatter(y_test, predictions)
from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

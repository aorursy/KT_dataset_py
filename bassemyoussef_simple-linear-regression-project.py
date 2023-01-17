import numpy as np
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df=pd.read_csv('../input/simplelinearregression/train.csv')
df.head()
df.isnull().sum()
df[df['x'].isnull()]

df[df['y'].isnull()]

df['y'].fillna(df['y'].mean(),inplace=True)
df.isnull().sum()
df.nlargest(10,'x')
df = df.drop([213])
df.nlargest(10,'x')
df.head()
X = df.iloc[:, :-1].values
y = df.iloc[:,-1:].values

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X,y)
#plotting data points for dataset
plt.scatter(X, y, color = 'red')

#plotting prediction line on Training dataset
plt.plot(X, regressor.predict(X), color = 'blue')

plt.title('X vs y (Training set)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
df_test=pd.read_csv('../input/simplelinearregression/test.csv')
df_test
df_test.isnull().sum()
X_test = df_test.iloc[:, :-1].values
y_test = df_test.iloc[:,-1:].values

y_predict=regressor.predict(X_test)
test_df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_predict.flatten()})
test_df
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('X vs y (Test set)')
plt.xlabel('X')
plt.ylabel('y')
plt.show()
from sklearn import metrics

print('Mean Absolute Error:\t\t', metrics.mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:\t\t', metrics.mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:\t', np.sqrt(metrics.mean_squared_error(y_test, y_predict)))

accuracy = regressor.score(X_test,y_test)
print("Accuracy:\t\t",accuracy*100,'%')

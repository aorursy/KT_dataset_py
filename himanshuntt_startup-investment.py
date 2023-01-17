%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as seabornInstance
dataset=pd.read_csv('../input/startup-data-investment/Start_data.csv')
x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values
plt.figure(figsize=(6,4))
plt.tight_layout()
seabornInstance.distplot(y)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder 
ct= ColumnTransformer(transformers=[('encoder ',OneHotEncoder(),[3])],remainder='passthrough')
x=np.array (ct.fit_transform(x))
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state = 0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_predict=regressor.predict(x_test)
print(regressor.intercept_)
print(regressor.coef_)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_predict.flatten()})
df
df1 = df.head(80)
df1.plot(kind='bar',figsize=(8,3))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='red')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
import sklearn
from sklearn import metrics
print('Mean Absolute Error:', sklearn.metrics.mean_absolute_error(y_test, y_predict))  
print('Mean Squared Error:', sklearn.metrics.mean_squared_error(y_test, y_predict))  
print('Root Mean Squared Error:', np.sqrt(sklearn.metrics.mean_squared_error(y_test, y_predict)))
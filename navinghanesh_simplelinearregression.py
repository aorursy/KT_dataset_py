import numpy as np

import matplotlib.pyplot as plt 

import pandas as pd 

import seaborn as sns 

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score,confusion_matrix

from sklearn.model_selection import train_test_split
data=pd.read_csv('../input/101-simple-linear-regressioncsv/1.01. Simple linear regression.csv')
data.head()
X=data['SAT']

y=data['GPA']
sns.regplot(data['SAT'],y)
X.head()
y.head()
X_reshaped=X.values.reshape(-1,1)
X.shape


X_reshaped.shape
regressor=LinearRegression()

regressor.fit(X_reshaped,y)
regressor.score(X_reshaped,y)
regressor.coef_
regressor.intercept_
new_data=pd.DataFrame([1117,1000])

new_data
X_train,X_test,y_train,y_test=train_test_split(X_reshaped,y,test_size=0.25,random_state=0)
y_pred=regressor.predict(X_test)
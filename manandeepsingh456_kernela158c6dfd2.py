import pandas as pd

import numpy as np

bike=pd.read_csv("../input/bike_share.csv")

bike.info()

bike.describe()

bike.nunique()

bike.head()
import matplotlib.pyplot as plt

import seaborn as sns 

#%matplotlib inline

sns.pairplot(bike) #histogram and scatter plot

sns.pairplot(bike,x_vars='count', y_vars='season', size=7, aspect=0.7, kind='scatter')

X=bike[['season','holiday','workingday','weather','temp','atemp','humidity','windspeed','casual','registered']]



y=bike['count']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test= train_test_split(X,y,train_size=0.7, random_state=100)



from sklearn.linear_model import LinearRegression

lm= LinearRegression()

lm.fit(X_train, y_train)



print(lm.intercept_)

coeff_df=pd.DataFrame(lm.coef_, X_test.columns,columns=['Coefficient'])

coeff_df







y_pred=lm.predict(X_test)

y_pred
from sklearn.metrics import mean_squared_error, r2_score

mse=mean_squared_error(y_test, y_pred)

r_squared=r2_score(y_test, y_pred)

print('Mean_squared_error :', mse)

print('r_square_value :', r_squared)
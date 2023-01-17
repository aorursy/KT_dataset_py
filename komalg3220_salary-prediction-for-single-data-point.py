



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split

from sklearn import linear_model

from sklearn import metrics



import os





print(os.listdir("../input"))



dataframe=pd.read_csv('../input/Salary_Data.csv')



dataframe.head(6)

dataframe.isnull().any()

x=dataframe['YearsExperience'].values

y=dataframe['Salary'].values

print(x)

print(y)
#train and test dataset creation 

x_train, x_test, y_train, y_test = train_test_split(x.reshape(-1,1), y, test_size=0.20, random_state=0) 

regression = linear_model.LinearRegression() 

regression.fit(x_train,y_train) 

predicted_Values = regression.predict(x_test) 



#checking accuracy of matrix 

print('score',regression.score(x_test,y_test)) 

mean_squared_error = metrics.mean_squared_error(y_test, predicted_Values) 

print('Root Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2))  

print('R-squared (training) ', round(regression.score(x_train, y_train), 3))  

print('R-squared (testing) ', round(regression.score(x_test, y_test), 3)) 

print('Intercept: ', regression.intercept_)

print('Coefficient:', regression.coef_)
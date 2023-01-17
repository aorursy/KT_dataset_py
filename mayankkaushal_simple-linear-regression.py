#Importing the necessary libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
#importing the dataset as dataframe 

df_test=pd.read_csv("../input/random-linear-regression/test.csv")

df_train=pd.read_csv("../input/random-linear-regression/train.csv")
df_train.info()
#To know the index of corresponding row 

df_train[df_train['y'].isnull()].index.tolist()
#Dropping the row

df_train=df_train.drop([213])

df_train.info()
df_test.info()
test_data=df_test.values

train_data=df_train.values
#Since we have a single column thus we will be reshaping it

X_test,y_test=test_data[:,0].reshape(-1,1),test_data[:,1].reshape(-1,1)

X_train,y_train=train_data[:,0].reshape(-1,1),train_data[:,1].reshape(-1,1)
#Scatter Plot

plt.scatter(X_train,y_train)

plt.xlabel('X_train')

plt.ylabel('y_train')

plt.title('Relationship between X and Y')
#Heatmap

import seaborn as sns

sns.heatmap(df_train.corr(),annot=True,cmap='Blues')
#Fitting of linear regression model

X_train=X_train.reshape(-1,1)

y_train=y_train.reshape(-1,1)

from sklearn.linear_model import LinearRegression

reg=LinearRegression()

reg.fit(X_train, y_train)

y_predict=reg.predict(X_test)
#To calculate R-square

reg.score(X_test,y_test)
#To find the coefficient and the intercept

coefficient=reg.coef_

intercept=reg.intercept_
print(f"The values of coefficient and intercept are {coefficient} and { intercept} .")
#To find the Root mean squared error

from sklearn.metrics import mean_squared_error

rmse=np.sqrt(mean_squared_error(y_test,y_predict))

rmse
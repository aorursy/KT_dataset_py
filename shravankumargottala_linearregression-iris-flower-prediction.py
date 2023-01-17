import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics

from sklearn.metrics import confusion_matrix
location = r"../input/Iris.csv"
df = pd.read_csv(location)
df.shape
df.isnull().sum()
df.head()
df.describe()
df.hist()
df_train_data = df.iloc[:,1:5]

sns.heatmap(df_train_data.corr(),annot = True)
#check for any object data types in the dataset

df.dtypes
#converting object data type into int data type using labelEncoder

enc = LabelEncoder()

df.iloc[:,4] = enc.fit_transform(df.iloc[:,4])
#Check the data types

df.dtypes
#Extract dependent and independent variables. Spliting the dataset into Training set and Test set.

X = df.iloc[:,0:4]

y = df.iloc[:,4]

X_train,x_test,y_train,y_test =train_test_split(X,y,test_size=.25,random_state=0)
#loading the model constructor

lin_Reg = LinearRegression()
#training or fitting the train data into the model

lin_Reg.fit(X_train,y_train)
y_pred = lin_Reg.predict(x_test)

y_pred
#calculating the residuals

print('y-intercept             :' , lin_Reg.intercept_)

print('beta coefficients       :' , lin_Reg.coef_)

print('Mean Abs Error MAE      :' ,metrics.mean_absolute_error(y_test,y_pred))

print('Mean Sqrt Error MSE     :' ,metrics.mean_squared_error(y_test,y_pred))

print('Root Mean Sqrt Error RMSE:' ,np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

print('r2 value                :' ,metrics.r2_score(y_test,y_pred))
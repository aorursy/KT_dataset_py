#importing the libraries



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



#importing the dataset

dataset=pd.read_csv('../input/salary-data-simple-linear-regression/Salary_Data.csv')

x= dataset.iloc[:,:-1].values

y=dataset.iloc[:,-1].values

"""

#Taing care of missing data

from sklearn.preprocessing import Imputer

Imputer = Imputer(missing_values="NaN", strategy="mean")

Imputer.fit(x[:,:-1])

x[:,:-1]=Imputer.transform(x[:,:-1])

"""

#Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)



"""

Feature scalling is not required here



#Feature scalling

from sklearn.preprocessing import StandardScaler

sc_x=StandardScaler()

x_train=sc_x.fit_transform(x_train)

x_test=sc_x.transform(x_test)



"""



from sklearn.linear_model import LinearRegression

regressor=LinearRegression()

regressor.fit(x_train, y_train)





#predicting the test set results

y_pred=regressor.predict(x_test)



#visualising the training set results

plt.scatter(x_train, y_train, color='red')

plt.plot(x_train, regressor.predict(x_train), color='blue')

plt.title('salary vs expereince result set')

plt.xlabel('years of expereince')

plt.ylabel('salary')

plt.show
#importing Necessary libraries

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
#Reading data from the datsheet which is in csv format



data=pd.read_csv('../input/salary/Salary.csv')

data
#look for any non-integer values ex:?,nan values

data.info()
#Lets see the mean and median and how the data in each column ranges and look for any values equal to 0

data.describe()
x=data['YearsExperience']
x
#converting series object to n-dimensional array which will be useful in reshaping 



x=np.array(x)
#reshape

x=x.reshape(-1,1)
y=data['Salary']
y=np.array(y)
y=y.reshape(-1,1)
from sklearn.model_selection import train_test_split
#Considering only 20 percent of data as test size



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#Applying linear Regression model



from sklearn.linear_model import LinearRegression

lr=LinearRegression()
#Fitting the training data for the model

lr.fit(x_train,y_train)
#predicting the output for the given inputs

pred=lr.predict(x_test)
#Measuring the accuracy of regressor model using r2_score



from sklearn.metrics import r2_score
print("The accuracy with linear regression model is:", r2_score(y_test,pred))
#Lets see the same data how it performs with Randomforestregressor
from sklearn.ensemble import RandomForestRegressor
rr=RandomForestRegressor(n_estimators=1000)
rr.fit(x_train,y_train)
predict=rr.predict(x_test)
print("The accuracy with Random forest regressor model is:" ,r2_score(y_test,predict))
#we have better accuracy with randomforest when compared with Linearregression 
#Lets use scatterplot to see how our testing datset Experience(input) is linearly related with Salary(output)



plt.scatter(x_test,y_test,color="red")    #gives scatter points in the dataset

plt.plot(x_test,lr.predict(x_test),color="yellow")    #draws/plots a line

plt.show()
#Lets use scatterplot to see how our training datset Experience(input) is linearly related with Salary(output)



plt.scatter(x_train,y_train,color="red")    #gives scatter points in the dataset

plt.plot(x_train,lr.predict(x_train),color="yellow")    #draws/plots a line

plt.show()
#As seen above the datapoints(Experience) are almost having strong linear relationship with Salary which is the reason for having high accuracy in the model
print("The acccuracy for the given input",rr.predict([[10]]))
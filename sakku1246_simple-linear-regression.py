# importing the dataset



import pandas as pd

Data = pd.read_csv("../input/regressioncsv/Salary_Data.csv")



Data.head()
# check if any null values

Data.isnull().sum()
# inependent and dependent variable

x= Data.iloc[:,:-1].values

y= Data.iloc[:, -1].values

from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest= train_test_split(x,y, test_size=0.2, random_state=0)

                                           
# Model building

from sklearn.linear_model import LinearRegression

reg=LinearRegression()

model= reg.fit(xtrain,ytrain)
ypred= reg.predict(xtest)
comparision= pd.DataFrame({"Actual ":ytest, "Predicted": ypred})

comparision
# Visualizing the training set 

import matplotlib.pyplot as plt

plt.scatter(xtrain,ytrain, color= "red")

plt.plot(xtrain, reg.predict(xtrain), color= "blue")

plt.title("Salary vs Years of Experience")

plt.xlabel("Salary")

plt.ylabel("Years of Experience")

# Visualizing the test set

plt.scatter(xtest,ytest, color= "red")

plt.plot(xtest, reg.predict(xtest), color= "blue")

plt.title("Salary vs Years of Experience")

plt.xlabel("Salary")

plt.ylabel("Years of Experience")

from sklearn.metrics import accuracy_score, mean_squared_error

from math import *

error= mean_squared_error(ytest, reg.predict(xtest))

print(error)



reg.score(xtest,ytest)*100
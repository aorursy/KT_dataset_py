import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
#supress warning

pd.set_option('mode.chained_assignment', None)



#Importing dataset

df = pd.read_csv("../input/Salary_Data.csv")

X = df.iloc[:,:-1]

Y = df.iloc[:,1]
#split data into train and test dataset

from sklearn.model_selection import train_test_split  #(for python2)

#from sklearn.model_selection import train_test_split  (for python3)

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=1/3, random_state=0)
#Fitting simple linear regression model to the training set

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()

regressor.fit(X_train, y_train)
#predicting the test dataset

y_pred = regressor.predict(X_test)
#visualize the training set result

plt.scatter(X_train, y_train, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title("Salary vs Experience (Training set)")

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.show()


#visualize the testing set result

plt.scatter(X_test, y_test, color='red')

plt.plot(X_train, regressor.predict(X_train), color='blue')

plt.title("Salary vs Experience (Testing set)")

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.show()
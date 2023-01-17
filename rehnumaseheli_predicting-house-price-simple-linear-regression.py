

# This model is using Linear Regression to predict house price based on the accessibility to the nearest MRT station.



#installing python libraries.



import pandas as pd

import numpy as np

import seaborn as sns

%matplotlib inline



#Reading the input data file.

Housing_data = pd.read_csv("../input/housing-data/Real_Estate_data.csv")
#Showing first few rows of data.

Housing_data.head()
#showing last few rows of data.

Housing_data.tail()
#Checking any null values.

Housing_data.isnull()
#Showing data types of all columns.

Housing_data.info()
#Showing the total number of rows and columns in the dataset.

Housing_data.shape
#Showing the scatter plot of "distance to the nearest MRT station" vs "house price".



sns.pairplot(Housing_data, x_vars = ['X3 distance to the nearest MRT station'], y_vars = ['Y house price of unit area'], height = 10, aspect = 1.0, kind = 'scatter')
#Declaring the independent variable.



X = Housing_data['X3 distance to the nearest MRT station']

X.head()
#Declaring the dependent variable.



Y = Housing_data['Y house price of unit area']

Y.head()
#Splitting the dataset into train and test.



from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, train_size = 0.7, random_state = 100)
#Showing the train data.



print(X_train.shape)
#Fixing the column issues in the train and test dataset as previously they showed only the number of rows.





import numpy as np



X_train = X_train[:, np.newaxis]

X_test = X_test[:, np.newaxis]
print(X_train.shape)
#Fixing linear regression model.



from sklearn.linear_model import LinearRegression 



lr = LinearRegression()



lr.fit (X_train, Y_train)
#Printing the intercept and slope of the linear regression model.



print (lr.intercept_)

print(lr.coef_)
#Declaring a prediction variable to test the model.



Y_pred = lr.predict(X_test)
#Showing prediction and test variable size.



print(Y_pred.shape)

print(Y_test.shape)
#Showing comparison between actual and predicted values.



import matplotlib.pyplot as plt

c = [i for i in range (0,125,1)]

fig=plt.figure()

plt.plot(c,Y_test, color="blue", linewidth=2.5, linestyle="-")

plt.plot(c,Y_pred, color="red", linewidth=2.5, linestyle="-")

fig.suptitle('Actual and Predicted', fontsize=20)

plt.xlabel('Index', fontsize=18)

plt.ylabel('Unit price',fontsize=16)
#Showing the residual plot.



c = [i for i in range (0,125,1)]

fig=plt.figure()

plt.plot(c,Y_test-Y_pred, color="blue", linewidth=2.5, linestyle="-")

fig.suptitle('Error Terms', fontsize=20)

plt.xlabel('Index', fontsize=18)

plt.ylabel('Y_test-Y_pred',fontsize=16)
#Calculating mean squared error.



from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error (Y_test, Y_pred)
#Calculating the R-squared value.



r_squared=r2_score (Y_test, Y_pred)
print('Mean_Squared_Error :', mse)

print('r_squared_value :',r_squared)
#Showing scatter plot of actual and predicted values.



import matplotlib.pyplot as plt

plt.scatter(Y_test, Y_pred)

plt.xlabel('Y Test')

plt.ylabel('Predicted Y')
# The Simple Linear Regression exercise has an R-squared value of 0.51 and Mean Squared Error of 68.61, which indicate that the replationship between

# the two variables do not have a very strong Linear relationship. That is why the actual and prdicted house unit price differed so

# much from each other. A non-linear regression model could be tried in this case to come up with a better model.
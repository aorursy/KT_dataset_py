#importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error, r2_score
df = pd.read_csv("../input/salary-data-simple-linear-regression/Salary_Data.csv")
#showing the dataframe information

df.info()
df.head()
#show the main statistics informations of the dataframe

df.describe()
  

# Creating dataset 



data = df.iloc[:, -1]

  

fig = plt.figure(figsize =(10, 7)) 

  

# Creating plot 

plt.boxplot(data) 

  

# show plot 

plt.show() 
#Showing the distribution

plt.rcParams['figure.figsize'] = [7, 7]

sns.distplot(df['YearsExperience'])
#Showing the values and how they are scattered

sns.scatterplot(x = "YearsExperience", y = "Salary", data = df)
df.corr()
#Show the correlation matrix

plt.title("Correlation Matrix")

plt.rcParams['figure.figsize'] = [10, 10]

sns.heatmap(df.corr(), annot = True)
#Splitting the dataset into the dependent and independent variables

x = df.iloc[:, :-1].values

y = df.iloc[:, -1].values
x
y
#splitting the data into the training set and the test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
regressor = LinearRegression()

regressor.fit(x_train, y_train)
#getting the score of the regression

regressor.score(x_train, y_train)
#Getting the coeficients of the linear regression

print("Salary = " + str(regressor.intercept_) + " + YearsExperience*" + 

     str(regressor.coef_[0]))
#Visualizing the training set results

plt.scatter(x_train, y_train, color="purple")

plt.plot(x_train, regressor.predict(x_train), color = "green")

plt.title("Salary vs Experience (Training set)")

plt.xlabel("Years of Experience")

plt.ylabel("Salary")

plt.show()
#Visualizing the test set results

plt.scatter(x_test, y_test, color = "purple")

plt.plot(x_train, regressor.predict(x_train), color = "green")

plt.title("Salary vs Experience (Test Set)")

plt.xlabel("Years of experience")

plt.ylabel("Salary")

plt.show()
#Show some relevant informations about the linear regression

y_pred = regressor.predict(x_test)

print("R2 value is " + str(r2_score(y_test, y_pred)))

print("The Mean Squared Error is " + str(mean_squared_error(y_test, y_pred)))
#Numpy is used so that we can deal with array's, which are necessary for any linear algebra

# that takes place "under-the-hood" for any of these algorithms.



import numpy as np





#Pandas is used so that we can create dataframes, which is particularly useful when

# reading or writing from a CSV.



import pandas as pd





#Matplotlib is used to generate graphs in just a few lines of code.



import matplotlib.pyplot as plt





#Sklearn is a very common library that allows you to implement most basic ML algorithms.

#Train_test_split will allow us to quickly split our dataset into a training set and a test set.



from sklearn.model_selection import train_test_split





#LinearRegression is the class of the algorithm we will be using.



from sklearn.linear_model import LinearRegression





#This will allow us to evaluate our fit using the R^2 score. 



from sklearn.metrics import r2_score

#read dataset from csv

dataset = pd.read_csv('/kaggle/input/sample-salary-data/Salary_Data.csv')



#set independent variable using all rows, and all columns except for the last one.

X = dataset.iloc[:, :-1].values



#set the dependent variable using all rows, but ony the last column.

y = dataset.iloc[:, 1].values



#Lets look at our data

dataset
#This will create x and y variables for training and test sets.

#Here we are using 25% of our examples for the test set.



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#this sets the object regressor to the class of LinearRegression from the Sklearn library.

regressor = LinearRegression()



#this fits the model to our training data.

regressor.fit(X_train, y_train)
#Predict on our test set.

y_pred = regressor.predict(X_test)
#here is the function, we simply pass in the x and y we want to plot.



def plot_results(x,y):

    plt.scatter(x, y, color = 'red')

    plt.plot(x, regressor.predict(x), color = 'blue')

    plt.title('Salary vs Experience')

    plt.xlabel('Years of Experience')

    plt.ylabel('Salary')

    plt.show()

    
#Visualize the training set



plot_results(X_train, y_train)
#Visualize the test set



plot_results(X_test, y_test)
#calculate the R^2 score

score = r2_score(y_test, y_pred)



#print out our score properly formatted as a percent.

print("R^2 score:", "{:.0%}".format(score))
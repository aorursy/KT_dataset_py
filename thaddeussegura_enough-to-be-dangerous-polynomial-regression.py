#Numpy is used so that we can deal with array's, which are necessary for any linear algebra

# that takes place "under-the-hood" for any of these algorithms.



import numpy as np





#Pandas is used so that we can create dataframes, which is particularly useful when

# reading or writing from a CSV.



import pandas as pd





#Matplotlib is used to generate graphs in just a few lines of code.



import matplotlib.pyplot as plt





#LinearRegression is the class of the algorithm we will be using.



from sklearn.linear_model import LinearRegression



#Polynomial Features will allow us to fit a polynomial model to the data. 



from sklearn.preprocessing import PolynomialFeatures



#read the data from csv

dataset = pd.read_csv('../input/position-salaries/Position_Salaries.csv')



#set independent variable by using all rows, but just column 1.

X = dataset.iloc[:, 1:2].values



#set the dependent variable using all rows but only the last column. 

y = dataset.iloc[:, 2].values



#take a look at our dataset

dataset
#I am going to wrap this all in a function.

def define_model(degree):



    #create an object of the class PolynomialFeatures

    poly_reg = PolynomialFeatures(degree)



    #call fit_transform on the x variables.

    X_poly = poly_reg.fit_transform(X)



    #now fit the transformed x's to the y's

    poly_reg.fit(X_poly, y)



    #create an object of the class LinearRegression

    lin_reg = LinearRegression()



    #fit the model to our transformed X

    lin_reg.fit(X_poly, y)

    

    #return these so we can pass them into the visualization

    return lin_reg, poly_reg



#call our function with the desired degrees (4). 

lin_reg, poly_reg = define_model(4)
# create a function so we can reuse the code.

def show_regression(lin_reg, poly_reg):

    #create a scatter plot with x and y.

    plt.scatter(X, y, color = 'red')

    #plot the predictions as a blue line.

    plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color = 'blue')

    #axes and title labels.

    plt.title('Salary Model')

    plt.xlabel('Position level')

    plt.ylabel('Salary')

    #show the completed plot

    plt.show()

    

#call our function.     

show_regression(lin_reg, poly_reg)
#Predicting a new result with Polynomial Regression

#call fit transform on the position level using the poly_reg object.

#feed this into the linear regression object to predict it.

#convert it to an int so it is easier to read (by default its a floating point in an array)

#assign to a variable so we can print it.



salary = int(lin_reg.predict(poly_reg.fit_transform([[7.5]])))



print("Estimated Salary: $", salary)
# degree = int(input("Please input the degree."))

# lin_reg, poly_reg = define_model(degree)

# show_regression(lin_reg, poly_reg)
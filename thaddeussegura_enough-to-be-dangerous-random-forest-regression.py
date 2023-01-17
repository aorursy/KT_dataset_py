#Numpy is used so that we can deal with array's, which are necessary for any linear algebra
# that takes place "under-the-hood" for any of these algorithms.

import numpy as np


#Pandas is used so that we can create dataframes, which is particularly useful when
# reading or writing from a CSV.

import pandas as pd


#Matplotlib is used to generate graphs in just a few lines of code.

import matplotlib.pyplot as plt


#DecisionTreeRegressor is the class of the algorithm we will be using.

from sklearn.tree import DecisionTreeRegressor

#Random forest regressor
from sklearn.ensemble import RandomForestRegressor
#read the data from csv
dataset = pd.read_csv('../input/position-salaries/Position_Salaries.csv')

#set independent variable by using all rows, but just column 1.
X = dataset.iloc[:, 1:2].values

#set the dependent variable using all rows but only the last column. 
y = dataset.iloc[:, 2].values

#take a look at our dataset
dataset
#create an object of the DecisionTree class.
rf_regressor = RandomForestRegressor(random_state = 0)

#fit it on the data, do not need to fit_transform her
rf_regressor.fit(X, y) 

#create an object of the DecisionTree class.
dt_regressor = DecisionTreeRegressor(random_state = 0)

#fit it on the data, do not need to fit_transform her
dt_regressor.fit(X, y) 
#wrap in a function so I can use it for both. 
def show_fit(regressor, title):    
    #Create a grid, necessary because of the veritical jumps.
    X_grid = np.arange(min(X), max(X), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))

    #create a scatter plot
    plt.scatter(X, y, color = 'red')

    #plot the X values and the predictions 
    plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

    #Titles and labels.
    plt.title(title)
    plt.xlabel('Position level')
    plt.ylabel('Salary')

    #Show the plot.
    plt.show()
show_fit(dt_regressor, 'Decision Tree Regression')
show_fit(rf_regressor, 'Random Forest Regression')
# Predicting a new result
y_pred = dt_regressor.predict([[6.5]])
print("Decision Tree Predicted Salary: $"+str(y_pred[0]))

y_pred2 = rf_regressor.predict([[6.5]])
print("Random Forest Predicted Salary: $"+str(y_pred2[0]))
#read the data from csv
dataset = pd.read_csv('../input/50-startups/50_Startups.csv')

#take a look at our dataset.  head() gives the first 5 lines. 
dataset.head()
#drop the columns.
dataset = dataset.drop(columns = ['Administration', 'State'])

#look at the changes
dataset.head()
#set independent variable by using all rows, but just column 1.
X = dataset.iloc[:, :-1].values

#set the dependent variable using all rows but only the last column. 
y = dataset.iloc[:, -1].values
#create an object of the DecisionTree class.
rf_regressor = RandomForestRegressor(random_state = 0)

#fit it on the data, do not need to fit_transform her
rf_regressor.fit(X, y) 

#create an object of the DecisionTree class.
dt_regressor = DecisionTreeRegressor(random_state = 0)

#fit it on the data, do not need to fit_transform her
dt_regressor.fit(X, y) 
budgets = [[300000, 200000], [200000,300000], [100000,400000]]

for budget in budgets:
    y_pred = dt_regressor.predict([budget])
    y_pred2 = rf_regressor.predict([budget])
    print("BUDGET:", budget,
          "\nDecision Tree Prediction:", y_pred,
          "\n Random Forest Prediction:", y_pred2,
          "\n--------------------------------------")
    
#wrap it in a function so I can use it for both. 
def show_3d(regressor, title):   
    # First well create the 3d figure.
    fig = plt.figure(figsize = (10,10))
    ax = plt.axes(projection='3d')

    #Next well pull out the datapoints for each axis
    zdata = y
    xdata = X[:, 0]
    ydata = X[:, 1]
    #Now we plot the points 
    ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Reds', s = 50);

    #Next we need to make x and y dimensions for the data.
    xline = np.linspace(min(X[:, 0]), max(X[:, 0]), 50)
    yline = np.linspace(min(X[:, 1]), max(X[:, 1]), 50)
    #combine those back into a dataset to apply the prediction on 
    z = np.concatenate((xline.reshape(-1,1),yline.reshape(-1,1)), axis = 1)
    #call the predictions 
    zline = regressor.predict(z)
    #plot the resulting line. 
    plt.title(title)
    ax.plot3D(xline, yline, zline, 'black')
show_3d(dt_regressor, 'Decision Tree Model')
show_3d(rf_regressor, 'Random Forest Model')

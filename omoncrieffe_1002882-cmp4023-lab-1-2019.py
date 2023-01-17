# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd  # assists with managing data frames and data series, useful data structures and methods

import numpy as np

import os

import random

import matplotlib.pyplot as plt

# indicates that we want our plots to be shown in our notebook and not in a sesparate viewer

# %matplotlib inline

rng= np.random.RandomState(1)

rng
#1 a.b.

cnt = 50



x1 = [rng.randint(500, 2000) for i in range(cnt)]

x2 = [rng.randint(100, 500) for x in range(cnt)]

x3 = []

y = []
#1 c.d.

for i in range(0, cnt):

    # Generate x3 Value

    x3Value = x2[i] * 3 + rng.randint(0, 50)

    x3.append(x3Value)



    # Generate y Value

    yValue = x3[i] + x2[i]

    y.append(yValue)
#defining the column names for dataframe

df = pd.DataFrame({

    'X1': x1,

    'X2': x2,

    'X3': x3,

    'Y': y

})
print(df)
#2

#Determine the correlation between: 

#a.X1 and Y



print("The correlation between X1 and Y:")

df[['X1','Y']].corr()
#b.X2 and Y, 



print("The correlation between X2 and Y:")

df[['X2','Y']].corr()
#c.X3 and Y



print("The correlation between X3 and Y:")

df[['X3','Y']].corr()
#3

#Plot a scatter plot illustrating the relationship between 

#a.X1 and Y



df.plot(kind="scatter",

        x='X1',

        y='Y',

        title="Graph displaying the Relationship Between X1 and Y",

        figsize=(12,8)

       )

plt.title("Graph displaying the Relationship Between X1 and Y",color='g')

plt.xlabel("X1", color='b',size=15)

plt.ylabel("Y", color='b',size=15)
#b. X2 and Y



df.plot(kind="scatter",

        x='X2',

        y='Y',

        title="Graph displaying the Relationship Between X2 and Y",

        figsize=(12,8)

       )

plt.title("Graph displaying the Relationship Between X2 and Y",color='g')

plt.xlabel("X2", color='b',size=15)

plt.ylabel("Y", color='b',size=15)
#4

#Perform Regression Analysis on your data set to determine the impact that X1

#and X2 (Independent) has on Y. Ensure that you perform cross validation on the data
# this_X = df['X1'].values

# this_Y = df['Y'].values
# #Mean of X and Y

# mean_x = np.mean(this_X)

# mean_y = np.mean(this_Y)



# # Total Number of Values

# m = len(this_X)



# #using the formula to calculate b1 and b0

# numer = 0

# denom = 0

# for i in range (m):

#     numer += (this_X[i] - mean_x) * (this_Y[i] - mean_y)

#     denom += (this_X[i] - mean_x) ** 2

# this_b1 = numer / denom

# this_b0 = mean_y - (this_b1 * mean_x)



# #print coefficients

# print(this_b1, this_b0)



# #The value for b1 is m and the value of b0 is C

# # in y = mx + c
# #plotting values and regression line

# max_x = np.max(X) + 100

# min_x = np.min(X) - 100



# #calculating line values x and y

# x = np.linspace(min_x, max_x, 1000)

# y = b0 + b1 * x



# #Plotting Line

# plt.plot(x, y, color='#58b970', label= 'Regression Line')



# #Plotting Scatter Points

# plt.scatter(this_X, this_Y, color='#ef5423', label= 'Scatter Plot')



# plt.xlabel('X1')

# plt.ylabel('Y')

# plt.gcf().set_size_inches((12, 8)) 

# plt.legend()

# plt.show()

# #X2 and Y

# X = df['X2'].values

# Y = df['Y'].values
# #Mean of X and Y

# mean_x = np.mean(X)

# mean_y = np.mean(Y)



# # Total Number of Values

# m = len(X)



# #using the formula to calculate b1 and b0

# numer = 0

# denom = 0

# for i in range (m):

#     numer += (X[i] - mean_x) * (Y[i] - mean_y)

#     denom += (X[i] - mean_x) ** 2

# b1 = numer / denom

# b0 = mean_y - (b1 * mean_x)



# #print coefficients

# print(b1, b0)



# #The value for b1 is m and the value of b0 is C

# # in y = mx + c
# #plotting values and regression line

# max_x = np.max(X) + 100

# min_x = np.min(X) - 100



# #calculating line values x and y

# x = np.linspace(min_x, max_x, 1000)

# y = b0 + b1 * x



# #Plotting Line

# plt.plot(x, y, color='#58b970', label= 'Regression Line')



# #Plotting Scatter Points

# plt.scatter(X, Y, color='#ef5423', label= 'Scatter Plot')



# plt.xlabel('X2')

# plt.ylabel('Y')

# plt.gcf().set_size_inches((12, 8)) 

# plt.legend()

# plt.show()
#Residual Analysis

# The difference between the observed value of the dependent variable and what is predicted by the regression model.

#observed - predicted

# The variance left unexpalined is due to model error. This error is the residual



# ss_r = []

# for i in range(m):

#     y_pred = b0 + b1 * X[i]

#     ss_rValues = Y[i] - y_pred

#     ss_r.append(ss_rValues)

# print (ss_r)
#5b. X2 and Y

# plt.scatter(df.X2, ss_r, color='#ef5423', label= 'Scatter Plot')

# this_ss_r = []

# for i in range(m):

#     y_pred = this_b0 + this_b1 * this_X[i]

#     ss_rValues = this_Y[i] - y_pred

#     this_ss_r.append(ss_rValues)

# print (this_ss_r)
#5a. X1 and Y

# plt.scatter(df.X1, this_ss_r, color='#ef5423', label= 'Scatter Plot')

# separate data into dependent (Y) and independent(X1 and X2) variables

#4

#Perform Regression Analysis on your data set to determine the impact that X1

#and X2 (Independent) has on Y. Ensure that you perform cross validation on the data



X_data = df[['X1','X2']]

Y_data = df['Y']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
# Import linear model package

from sklearn import linear_model
reg= linear_model.LinearRegression()
reg.fit(X_train,y_train)
X_train.columns
reg.coef_
print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
# perform cross validation

from sklearn import datasets

from sklearn import svm

from sklearn.model_selection import cross_val_score



scores = cross_val_score(reg,X_data, Y_data, cv=5)

scores  
test_predicted = reg.predict(X_test)

test_predicted
#5.Residual Analysis

# The difference between the observed value of the dependent variable and what is predicted by the regression model.

#observed - predicted

# The variance left unexpalined is due to model error. This error is the residual



import seaborn

w = 12

h = 10

d = 70

plt.figure(figsize=(w, h), dpi=d)

seaborn.residplot(y_test, test_predicted, lowess=True)

plt.savefig("out.png")
#6.Determine the Coefficient of Determination (R^2 ) of your model. Explain what this means



from sklearn.metrics import r2_score



coefficient_of_dermination = r2_score(y_test, test_predicted)



print(coefficient_of_dermination)

#7 What is the impact of X1 and X2  on Y?



# Coefficient

#X1 has little impact on Y base on the coefficient, while X2 has a greater impact on y given its coefficient calculated.

reg.coef_
#8 From your model, deduce the regression formula.



reg.intercept_
#9 Use a new value for X1 and X2 (not previously in your data set) and predict the associated outcome



newX = [[700, 250]]

newPredict = reg.predict(newX)

print("X=%s, Predicted=%s" % (newX, newPredict))
#10a MAE



from sklearn.metrics import mean_absolute_error



print("Mean absolute error: %.2f" % mean_absolute_error(y_test, test_predicted))

#10b RMSE

from math import sqrt

from sklearn.metrics import mean_squared_error



print("Root of Mean Squared Error: %.2f" % sqrt(mean_squared_error(y_test, test_predicted)))
#10c MSE



from sklearn.metrics import mean_squared_error



print("Mean Squared Error: %.2f" % mean_squared_error(y_test, test_predicted))

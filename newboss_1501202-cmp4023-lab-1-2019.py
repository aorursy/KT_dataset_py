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
import random

#Generate random number from X1

X1 = [random.randint(500, 2000) for iter in range(50)]



print (X1)
X2 = [random.randint(100, 500) for iter in range(50)]



print (X2)
X3=[]

for i in range(50):

    k = X1[i]* 3 + random.randint(2,5)

    X3.append(k) #Add each value to the list after calculation

print(X3)

#len(X3)

  
y=[]

for i in range(50):

    l = X3[i] + X2[i]

    y.append(l)

print(y)

#Turned list into dictionary and created dataframe from dictionary

dict = {'X1': X1, 'X2': X2, 'X3': X3, 'y': y}  

    

df = pd.DataFrame(dict) 

    

df 



df.corr() # Finds Pearson correlation on the dataframe
df['X1'].corr(df['y'])
df['X2'].corr(df['y'])
df['X3'].corr(df['y'])
import matplotlib.pyplot as plt

df.plot(kind= 'scatter',x='X1',y='y',

           title=" Scatter Plot X1 vs Y",

           figsize=(12,8))

plt.title("From %d to %d" %(

     df['X1'].min(),

     df['X1'].max()

    

 ))



plt.suptitle("X1 vs Y", size=12)

plt.ylabel("Y")

plt.xlabel("X1 Axis")
df.plot(kind= 'scatter',x='X2',y='y',

           title=" Scatter Plot X2 vs Y",

           figsize=(12,8))

plt.title("From %d to %d" %(

     df['X2'].min(),

     df['X2'].max()

    

 ))



plt.suptitle("X2 vs Y", size=12)

plt.ylabel("Y Axis")

plt.xlabel("X2 Axis")
# separate our data into dependent (Y) and independent(X) variables

X_data = df[['X1','X2']]

Y_data = df['y']
from sklearn.model_selection import train_test_split 

X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.30)
from sklearn import linear_model


# Create an instance of linear regression

reg = linear_model.LinearRegression()


reg.fit(X_train,y_train)


X_train.columns


print("Regression Coefficients")

pd.DataFrame(reg.coef_,index=X_train.columns,columns=["Coefficient"])
reg.score(X_test,y_test)

# R2 of 0.99 means that 99 percent of the variance in Y is predictable from X
#Cross validation

from sklearn import datasets

from sklearn import svm

from sklearn.model_selection import cross_val_score



scores = cross_val_score(reg,X_data, Y_data, cv=5)

scores                                              

#Residual Plot

plt.scatter(reg.predict(X_train), reg.predict(X_train)-y_train,c='b',s=40,alpha=0.5)

plt.scatter(reg.predict(X_test),reg.predict(X_test)-y_test,c='g',s=40)

plt.hlines(y=0,xmin=np.min(reg.predict(X_test)),xmax=np.max(reg.predict(X_test)),color='red',linewidth=3)

plt.title('Residual Plot using Training (blue) and test (green) data ')

plt.ylabel('Residuals')


# Make predictions using new data

Xnew = [[99, 3000]]

test_predicteds = reg.predict(Xnew)

print("X=%s, Predicted=%s" % (Xnew, test_predicteds))
#Intercept

reg.intercept_

# Coeffiecnt

#postitive impact on y because the values are positive

reg.coef_
from sklearn.metrics import mean_squared_error, mean_absolute_error

from math import sqrt
test_predicted = reg.predict(X_test)

print("Mean squared error: %.2f" % mean_squared_error(y_test, test_predicted))

print("Mean Absolute error: %.2f" % mean_absolute_error(y_test, test_predicted))

print("Root Mean squared error: %.2f" % sqrt(mean_squared_error(y_test, test_predicted)))



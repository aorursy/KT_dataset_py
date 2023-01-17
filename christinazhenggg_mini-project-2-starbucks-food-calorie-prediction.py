# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

import math

import pylab

from sklearn import metrics





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





#import dataset

data = pd.read_csv("../input/starbucks/starbucks-menu-nutrition-food.csv",encoding='ISO-8859-1')



# drop the column with the food names cuz we don't need it

data.drop(columns=['Food'], inplace=True)



data.head(10)
# check if any of the columns contains NaN

data.isnull().any()
# check correlation between calories and fat

corr = data.corr()



# plot the correlation heatmap

sns.heatmap(corr, xticklabels=corr.columns, yticklabels=corr.columns, cmap='Oranges')
# assign independent and dependent variables

# convert columns into array

X = data.iloc[:,1].values.reshape(-1,1)

Y = data.iloc[:,0].values.reshape(-1,1)



# split X and y into X_

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split (X, Y, test_size = 0.2, random_state=1)
# retrieve the intercept:

print(model.intercept_)

# retrieving the slope:

print(model.coef_)
# create a linear regression model and fit the data into the model

model = LinearRegression()

model.fit(X_train,y_train)



# create a prediction chart

Y_pred = model.predict(X_test)



plt.figure(figsize=(8, 8))

plt.scatter(X_test,y_test,s=20,c='#ff9900')

plt.plot(X_test,Y_pred,color='yellow')

plt.title('Positive Relationship Between Calories and Fat Content')

plt.xlabel('Fat Content')

plt.ylabel('Calorie Count')
# compare the first 25 actual output values with predicted values

df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})

df1 = df.head(25)



# visualize the dataframe

df1.plot(kind='bar',figsize=(12,6))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')

plt.show()
# make a single prediction

prediction = model.predict([[26]])

predicted_value = prediction[0][0]

print("When there are 26g of fat, the predicted calorie count of the food is {:.4}". format(predicted_value))
# check the average value of the 'Calories' column

plt.figure(figsize=(10,6))

plt.tight_layout()

sns.distplot(data.iloc[:,0])
# define input and output variables

X = data.iloc[:,1:].values

Y = data.iloc[:,0].values



# Split X and y into X_

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)



# create and train the model

mlmodel = LinearRegression()

mlmodel.fit(X_train, y_train)
# get multiple predictions

y_pred = mlmodel.predict(X_test)



# show the first 5 predictions

y_pred[:5]
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})



df1 = df.head(25)



df1.plot(kind='bar',figsize=(12,6))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='grey')

plt.show()
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
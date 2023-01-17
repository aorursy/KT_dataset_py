# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as pp
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
%matplotlib inline
prices = pd.read_csv("../input/avocado.csv").drop(['Unnamed: 0'],axis = 1)
prices['Date'] = pd.to_datetime(prices['Date'])
prices.head(5)
prices = prices.drop(['4046','4225','4770','Large Bags','Small Bags','XLarge Bags','Total Volume'],axis=1)
US = prices.loc[(prices['region']) == 'TotalUS']  
US.head(5)
sns.regplot(US['Total Bags'],US['AveragePrice'])

#from the graph, we can see that as the total bas is low, average price is higher. demand and supply  
#creating a simple regression model for total bags and Average price

training_data,testing_data = train_test_split(US,train_size = 0.80,random_state = 35)
reg = linear_model.LinearRegression()

X_train,Y_train = pd.DataFrame(training_data['Total Bags']),training_data['AveragePrice']

model = reg.fit(X_train,Y_train)
print("cocoefficient Value: ",(np.float(model.coef_)))
print("Intercept Value: ",(model.intercept_))

X_test,Y_test = pd.DataFrame(testing_data['Total Bags']),testing_data['AveragePrice']

pred = model.predict(X_test)
mean_squared_error(Y_test,pred)
r2_score(Y_test,pred)
pp.figure(figsize=(12,4))

pp.scatter(X_test,Y_test)
pp.plot(X_test,pred,color = 'black')
pp.ylabel("Average Price")
pp.xlabel("Total Bags")
pp.show()

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.metrics import r2_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")

df.head()
df.info()
df.isnull().sum()
df = df.fillna(0)
df.sort_values("Global_Sales", axis = 0, ascending = False, inplace = True) 

df
plt.figure(figsize=(10,5))

sns.countplot(df["Genre"])

plt.show()
plt.figure(figsize=(20,5))

sns.countplot(df["Platform"])

plt.show()
plt.figure(figsize=(10,6))

sns.heatmap(df.corr(), annot=True, fmt=".2f")

plt.show()
x = np.array(df.loc[:,"NA_Sales"]).reshape(-1,1)

y = np.array(df.loc[:,"Global_Sales"]).reshape(-1,1)

plt.figure(figsize=(10,5))

plt.xlabel("NA Sales")

plt.ylabel("Global Sales")

plt.scatter(x,y,c="red")
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

#predict range

predict_space = np.linspace(min(x), max(x)).reshape(-1,1)

#fitting

lin_reg.fit(x,y)

#predict

y_head = lin_reg.predict(predict_space)



# R square score

print("R square score: ",lin_reg.score(x,y))

# Plot regression line and scatter

plt.figure(figsize=(15,6))

plt.scatter(x=x,y=y,c="red")

plt.plot(predict_space, y_head, color='blue', linewidth=3)

plt.xlabel('NA Sales')

plt.ylabel('Global Sales')

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures





#polynomial regression fitting

poly_reg = PolynomialFeatures(degree=5)

x_polynomial = poly_reg.fit_transform(x) # I determined that X array as polynomial fit value and I'll use it for linear regression model's fit.





lin_reg = LinearRegression()





#fitting

lin_reg.fit(x_polynomial,y) 



#predict

y_head = lin_reg.predict(x_polynomial)



# Visualize



print("R Square Score: ",r2_score(y,y_head))





plt.figure(figsize=(15,6))

plt.scatter(x=x,y=y,c="red")

plt.plot(x, y_head, color='blue', linewidth=3)

plt.xlabel('NA Sales')

plt.ylabel('Global Sales')

plt.show()
from sklearn.tree import DecisionTreeRegressor





tree_reg = DecisionTreeRegressor()



#fitting and prediction

tree_reg.fit(x,y)



x_ = np.arange(min(x),max(x),0.01).reshape(-1,1) # I increased leaf of tree actually 

y_head = tree_reg.predict(x) # machine will do prediction according to x_ 



#visualize



print("R Square Score: ",r2_score(y,y_head))





plt.figure(figsize=(15,6))

plt.scatter(x,y,c="red")

plt.plot(x,y_head,c="green")

plt.xlabel("NA Sales")

plt.ylabel("Global Sales")

plt.show()
from sklearn.ensemble import RandomForestRegressor



random_forest = RandomForestRegressor(n_estimators=100,random_state=42)

#fitting

random_forest.fit(x,y)



#prediction

y_head = random_forest.predict(x)



# visualize and R-Square Score



print("R Square Score: ",r2_score(y,y_head))



plt.figure(figsize=(15,6))

plt.scatter(x,y,c="red")

plt.plot(x,y_head,c="blue")

plt.xlabel("NA Sales")

plt.ylabel("Global Sales")

plt.show()
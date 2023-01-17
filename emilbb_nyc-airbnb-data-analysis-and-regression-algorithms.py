import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

import plotly_express as px



#matplotlib inline

sns.set_style("whitegrid")

color_pallete = ['#FF1744', '#666666']

sns.set_palette(color_pallete, 2)

sns.set(style="ticks", color_codes=True)
# Importing the dataset

dataset = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 1].values

#print(dataset)





print(dataset.info())

print(dataset.describe())
dataset.head()


num_cols = []

cat_cols = []



for i in dataset.columns[:]:

    if (dataset[i].dtype == 'object'):

        cat_cols.append(i)

    else:

        num_cols.append(i)



# print(num_cols)

# print(cat_cols)



#Looking for missing values¶



for i in cat_cols:

    if((dataset[i] == '?').sum()>0):

        print(i, (dataset[i] == '?').sum())



#Droping rows with missing values

# first replace '?' with np.nan

dataset.replace('?', np.nan, inplace=True)

for i in cat_cols:

    if ((dataset[i] == '?').sum() > 0):

        print(i, (dataset[i] == '?').sum())



# then drop rows with na

dataset.dropna(axis=0, inplace=True)

plt.figure(figsize=(28, 18))

sns.heatmap(dataset.corr(), annot=True, cmap='RdBu', vmin=-1, vmax=1)

plt.plot()
plt.figure(figsize=(10, 10))

sns.barplot(x='room_type' , y='price', data=dataset)

plt.plot()
plt.figure(figsize=(10, 10))

sns.barplot(x='neighbourhood_group' , y='price', data=dataset)

plt.plot()
plt.figure(figsize=(10, 10))

sns.barplot(x='price' , y='minimum_nights', data=dataset)

plt.plot()
plt.figure(figsize=(9, 9))

sns.barplot(x='price' , y='number_of_reviews', data=dataset)

plt.plot()
from sklearn.tree import DecisionTreeRegressor





X = dataset.iloc[:, 15:16].values

y = dataset.iloc[:, 9].values



# X = dataset['number_of_reviews'].values

# y = dataset['price'].values

regressor = DecisionTreeRegressor(random_state = 0, max_depth=3)

regressor.fit(X, y)

# Predicting a new result

lvl = 6.5

y_pred = regressor.predict(X)

lvl_pred = regressor.predict([[lvl]])

print(f"The predicted price at level {lvl} is {lvl_pred}")



# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Availability 365 and price (Decision Tree Regression)')

plt.xlabel('Availability 365')

plt.ylabel('Price')

plt.show()
regressor = DecisionTreeRegressor(random_state = 0, max_depth=5)

regressor.fit(X, y)

lvl = 6.5

y_pred = regressor.predict(X)

lvl_pred = regressor.predict([[lvl]])

print(f"The predicted price at level {lvl} is {lvl_pred}")



# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Availability 365 and price (Decision Tree Regression)')

plt.xlabel('Availability 365')

plt.ylabel('Price')

plt.show()
from sklearn.tree import DecisionTreeRegressor





X = dataset.iloc[:, 15:16].values

y = dataset.iloc[:, 9].values



# X = dataset['number_of_reviews'].values

# y = dataset['price'].values

regressor = DecisionTreeRegressor(random_state = 0)

regressor.fit(X, y)

# Predicting a new result

lvl = 10

y_pred = regressor.predict(X)

lvl_pred = regressor.predict([[lvl]])

print(f"The predicted price at level {lvl} is {lvl_pred}")



# Visualising the Decision Tree Regression results (higher resolution)

X_grid = np.arange(min(X), max(X), 0.01)

X_grid = X_grid.reshape((len(X_grid), 1))

plt.scatter(X, y, color = 'red')

plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')

plt.title('Availability 365 and price (Decision Tree Regression)')

plt.xlabel('Availability 365')

plt.ylabel('Price')

plt.show()
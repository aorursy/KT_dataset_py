import pandas as pd



bikes = pd.read_csv("../input/nyc-east-river-bicycle-crossings/nyc-east-river-bicycle-counts.csv")



bikes.head()
# Unnamed: 0 appears to be index column. Easy enough to get rid of it

# Could add parameter to pd.read_csv to import first column as index,  but not needed

bikes.drop('Unnamed: 0', axis=1, inplace=True)

bikes.head()
# Any NaN values in the data?

len(bikes[bikes.isnull().any(axis=1)])
# Call describe to get an overview - maybe see columns with data issues

bikes.describe()
# Wish to see the relationship between low temp and number of cyclists

# import libraries

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

%matplotlib inline
# Separate the data

# low temp (4th col) and total bikers (10th). Python index starts at 0

# using .values to convert df to numpy array

data = bikes.iloc[:, [3,9]].values 
# scale the data so the relationships can make sense

# the number of people being in the thousands versus temps in the tens makes comparison clumsy

scaler = MinMaxScaler()

data = scaler.fit_transform(data)

data # Now each entry is between 0 and 1 in their respective columns
X = data[:, 0] # Low temps. Feature

y = data[:, 1] # Total bikers. Label

plt.scatter(X, y) # Produce scatter plot

plt.xlim((-.1, 1.1)) # Set x and y limits on graph

plt.ylim((-.1, 1.1))
#Use numpy polyfit to plot a quick linear regression line

import numpy as np



a, b = np.polyfit(X, y, deg=1) # deg=1 for linear

f = lambda x: a*x + b

plt.scatter(X, y)

plt.plot(X, f(X), lw=2.5, c="orange")

plt.xlim((-.1, 1.1))



plt.ylim((-.1, 1.1))
bikes.dtypes
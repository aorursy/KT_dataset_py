# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!pip3 install vpython 
# Load libraries
import numpy as np
import pylab as pl
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
# Import libraries necessary for this project
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
import seaborn as sns

# Import supplementary visualizations code visuals.py
#import visuals as vs
from vpython import *
# Pretty display for notebooks
%matplotlib inline
"""visual module has been renamed to vpython lately.

So to run this now, you first install vpython like:
sudo pip3 install vpython 
then replace the line:

from visual import *
with

from vpython import *
That worked for me.
credits to : https://stackoverflow.com/questions/28592211/importerror-no-module-named-visual
"""

data = pd.read_csv('../input/bostonhoustingmlnd/housing.csv')
data
# Load the Boston housing dataset

prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
print("Boston housing dataset has {} data points with {} variables each.".format(*data.shape))
# TODO: Minimum price of the data
minimum_price = np.min(prices)

# TODO: Maximum price of the data
maximum_price = np.max(prices)

# TODO: Mean price of the data
mean_price = np.mean(prices)

# TODO: Median price of the data
median_price = np.median(prices)

# TODO: Standard deviation of prices of the data
std_price = np.std(prices)

# Show the calculated statistics
print("Statistics for Boston housing dataset:\n")
print("Minimum price: ${:,.2f}".format(minimum_price))
print("Maximum price: ${:,.2f}".format(maximum_price))
print("Mean price: ${:,.2f}".format(mean_price))
print("Median price: ${:,.2f}".format(median_price))
print("Standard deviation of prices: ${:,.2f}".format(std_price))
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
ax.hist(data['RM'], bins = 15)  
plt.title("Average number of rooms Distribution ")
plt.xlabel("RM")
plt.ylabel("frequency")
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
ax.hist(data['LSTAT'], bins = 15)  
plt.title("Homeowners distribution with low class")
plt.xlabel("LSTAT")
plt.ylabel("frequency")
plt.show()

fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
ax.hist(data['PTRATIO'], bins = 15)  
plt.title("Students to Teachers ratio distribution")
plt.xlabel("PTRATIO")
plt.ylabel("frequency")
plt.show()
fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
ax.scatter(data['RM'], data['MEDV']) 
#Lables & Title
plt.title("Average selling Prices and Average number of rooms")
plt.xlabel("RM")
plt.ylabel("Prices")
plt.show()

#                       LSTAT VS PRICES
fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
ax.scatter(data['LSTAT'], data['MEDV'])  
plt.title("Average selling Prices VS % of low class Homeowners")
plt.xlabel("LSTAT")
plt.ylabel("Prices")
plt.show()

#                       PTRATIO VS PRICES
fig=plt.figure()
ax=fig.add_subplot(1, 1, 1)
ax.scatter(data['PTRATIO'], data['MEDV'])  
plt.title("Average selling Prices and Ratio of Students to Teachers")
plt.xlabel("PTRATIO")
plt.ylabel("Prices")
plt.show()

def df_to_plotly(df):
    return {'z': data.values.tolist(),
            'x': data.columns.tolist(),
            'y': data.index.tolist()}
#plot scatter between prices and number of rooms.
sns.regplot(x=features['RM'],y=prices,color='c')
sns.set_style("whitegrid")
#plot scatter between prices and percent of lower class workers.
sns.regplot(x=features['LSTAT'], y=prices ,color='c')
sns.set_style("whitegrid")
#plot scatter between prices and ratio of student and teacher.
sns.regplot(x=features['PTRATIO'], y=prices, color='c')
sns.set_style("whitegrid")

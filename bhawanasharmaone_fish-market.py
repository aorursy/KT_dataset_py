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
# Supress Warnings

import warnings
warnings.filterwarnings('ignore')
# Import required libraries

# import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import data from github

# url = 'https://raw.githubusercontent.com/bhavna9719/fish_market/master/Fish.csv'
# fish = pd.read_csv(url)
fish = pd.read_csv("/kaggle/input/fish-market/Fish.csv")
# First 5 rows of df

fish.head()
# Rows and columns count

fish.shape
# Datatypes, null value check

fish.info()
# Statistical data of df

fish.describe()
# Species count graph

sns.countplot(fish.Species)
plt.show()
# Correlation check

sns.heatmap(fish.corr(), annot = True, cmap = "Greens")
plt.show()
# Can conclude that every column has a good correlation with target variable
# Create dummies and merge

fish_dum = pd.get_dummies(fish['Species'], drop_first = True)
fish = pd.concat([fish, fish_dum], axis = 1)
fish.drop( "Species", axis = 1, inplace = True)
fish.head()
# Import required library
from sklearn.model_selection import train_test_split

# Train and test df split
np.random.seed(0)
fish_train, fish_test = train_test_split( fish, train_size = 0.7, test_size = 0.3, random_state = 100)
# Row and column count of train df

fish_train.shape
# Row and column count of test df

fish_test.shape
# Import required library

from sklearn.preprocessing import StandardScaler
# Standard scaler 

scaler = StandardScaler()
# scaler() to all the columns except the 'dummy' variables

num_vars = ['Weight', 'Length1', 'Length2', 'Length3', 'Height', 'Width']

fish_train[num_vars] = scaler.fit_transform(fish_train[num_vars])
# Statistical data of train df

fish_train.describe()
# Correlation recheck

plt.figure( figsize = [20,10])
sns.heatmap( fish_train.corr(), annot = True, cmap = "YlGnBu")
plt.show()
# We can see a detailed view of correlations with low correlation too previously it was high with all
# Prepare dataframe for model building

y_train = fish_train.pop('Weight')
X_train = fish_train
# Import required library and create model one

import statsmodels.api as sm

X_train_lm = sm.add_constant(X_train)
lr_1 = sm.OLS(y_train, X_train_lm).fit()
lr_1.summary()
# Drop high p-value variable

X_train.drop( "Height", axis = 1, inplace = True)
# create model two after drop

X_train_lm = sm.add_constant(X_train)
lr_2 = sm.OLS(y_train, X_train_lm).fit()
lr_2.summary()
# Drop high p-value variable

X_train.drop( "Whitefish", axis = 1, inplace = True)
# create model three after drop

X_train_lm = sm.add_constant(X_train)
lr_3 = sm.OLS(y_train, X_train_lm).fit()
lr_3.summary()
# Drop high p-value variable

X_train.drop( "Length3", axis = 1, inplace = True)
# create model four after drop

X_train_lm = sm.add_constant(X_train)
lr_4 = sm.OLS(y_train, X_train_lm).fit()
lr_4.summary()
# Drop high p-value variable

X_train.drop( "Width", axis = 1, inplace = True)
# create model five after drop

X_train_lm = sm.add_constant(X_train)
lr_5 = sm.OLS(y_train, X_train_lm).fit()
lr_5.summary()
# Drop high p-value variable

X_train.drop( "Perch", axis = 1, inplace = True)
# create model six after drop

X_train_lm = sm.add_constant(X_train)
lr_6 = sm.OLS(y_train, X_train_lm).fit()
lr_6.summary()
# Drop high p-value variable

X_train.drop( "Roach", axis = 1, inplace = True)
# create model seven after drop

X_train_lm = sm.add_constant(X_train)
lr_7 = sm.OLS(y_train, X_train_lm).fit()
lr_7.summary()
# Drop high p-value variable

X_train.drop( "Length1", axis = 1, inplace = True)
# create model eight after drop

X_train_lm = sm.add_constant(X_train)
lr_8 = sm.OLS(y_train, X_train_lm).fit()
lr_8.summary()
# Drop high p-value variable

X_train.drop( "Parkki", axis = 1, inplace = True)
# create model nine after drop

X_train_lm = sm.add_constant(X_train)
lr_9 = sm.OLS(y_train, X_train_lm).fit()
lr_9.summary()
# Import required libraries
from statsmodels.stats.outliers_influence import variance_inflation_factor
# Create a dataframe that will contain the names of all the feature variables and their respective VIFs

vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif
# Predict the target variable using final model

y_train_weight = lr_9.predict(X_train_lm)
# Plot the histogram of the error terms

fig = plt.figure()
sns.distplot((y_train - y_train_weight), bins = 20)
fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 
plt.xlabel('Errors', fontsize = 18)                         # X-label
plt.show()
# Prepare test dataframe
# Scale previously scaled variables
fish_test[num_vars] = scaler.transform(fish_test[num_vars])
# Statistical values check
fish_test.describe()
# Prepare test dataset

y_test = fish_test.pop('Weight')
X_test = fish_test
# Adding constant variable to test dataframe

X_test_m1 = sm.add_constant(X_test)
# Creating X_test_m1 dataframe by dropping variables from X_test_m1

X_test_m1 = X_test_m1.drop(["Height", "Whitefish", "Length3", "Width", "Perch", "Roach", "Length1", "Parkki"], axis = 1)
# Making predictions using the ninth model

y_pred_m1 = lr_9.predict(X_test_m1)
# Plotting y_test and y_pred to understand the spread

fig = plt.figure()
plt.scatter(y_test, y_pred_m1)
fig.suptitle('y_test vs y_pred', fontsize = 20)              # Plot heading 
plt.xlabel('y_test', fontsize = 18)                          # X-label
plt.ylabel('y_pred', fontsize = 16)  
plt.show()
# Check predicted vs actual target variable

plt.figure( figsize = [10,8])
n = range(1,len(y_test)+1)
plt.plot(n, y_test, label = "Actual")
plt.plot(n, y_pred_m1, label = "Predicted")
plt.legend()
plt.show()
# Checking R-square value of test dataset

from sklearn.metrics import r2_score
r2_score(y_test, y_pred_m1)
# The equation of best fitted line is:

# Weight = const * 0.0480 + Length2 *	1.2001 + Smelt * 0.7073 + Pike * -0.9741

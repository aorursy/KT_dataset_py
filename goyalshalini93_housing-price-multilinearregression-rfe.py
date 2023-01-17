# Supress Warnings



import warnings

warnings.filterwarnings('ignore')
import pandas as pd

import numpy as np
# Importing Housing.csv

housing = pd.read_csv('../input/Housing.csv')
# Looking at the first five rows

housing.head()
# List of variables to map



varlist =  ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']



# Defining the map function

def binary_map(x):

    return x.map({'yes': 1, "no": 0})



# Applying the function to the housing list

housing[varlist] = housing[varlist].apply(binary_map)
# Check the housing dataframe now



housing.head()
# Get the dummy variables for the feature 'furnishingstatus' and store it in a new variable - 'status'



status = pd.get_dummies(housing['furnishingstatus'])



# Check what the dataset 'status' looks like

status.head()
# Let's drop the first column from status df using 'drop_first = True'

status = pd.get_dummies(housing['furnishingstatus'], drop_first = True)



# Add the results to the original housing dataframe

housing = pd.concat([housing, status], axis = 1)



# Now let's see the head of our dataframe.

housing.head()
# Drop 'furnishingstatus' as we have created the dummies for it

housing.drop(['furnishingstatus'], axis = 1, inplace = True)



housing.head()
from sklearn.model_selection import train_test_split



# We specify this so that the train and test data set always have the same rows, respectively



df_train, df_test = train_test_split(housing, train_size = 0.7, test_size = 0.3, random_state = 100)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
# Apply scaler() to all the columns except the 'yes-no' and 'dummy' variables

num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']



df_train[num_vars] = scaler.fit_transform(df_train[num_vars])



df_train.head()
y_train = df_train.pop('price')

X_train = df_train
# Importing RFE and LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression
# Running RFE with the output number of the variable equal to 10

lm = LinearRegression()

lm.fit(X_train, y_train)



rfe = RFE(lm, 10)             # running RFE

rfe = rfe.fit(X_train, y_train)
list(zip(X_train.columns,rfe.support_,rfe.ranking_))
col = X_train.columns[rfe.support_]

col
X_train.columns[~rfe.support_]
# Creating X_test dataframe with RFE selected variables

X_train_rfe = X_train[col]
# Adding a constant variable 

import statsmodels.api as sm  

X_train_rfe = sm.add_constant(X_train_rfe)
lm = sm.OLS(y_train,X_train_rfe).fit()   # Running the linear model
#Let's see the summary of our linear model

print(lm.summary())
X_train_new = X_train_rfe.drop(["bedrooms"], axis = 1)
# Adding a constant variable 

import statsmodels.api as sm  

X_train_lm = sm.add_constant(X_train_new)
lm = sm.OLS(y_train,X_train_lm).fit()   # Running the linear model
#Let's see the summary of our linear model

print(lm.summary())
X_train_new.columns
X_train_new = X_train_new.drop(['const'], axis=1)
# Calculate the VIFs for the new model

from statsmodels.stats.outliers_influence import variance_inflation_factor



vif = pd.DataFrame()

X = X_train_new

vif['Features'] = X.columns

vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

vif['VIF'] = round(vif['VIF'], 2)

vif = vif.sort_values(by = "VIF", ascending = False)

vif
y_train_price = lm.predict(X_train_lm)
# Importing the required libraries for plots.

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
# Plot the histogram of the error terms

fig = plt.figure()

sns.distplot((y_train - y_train_price), bins = 20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)                         # X-label
num_vars = ['area', 'bedrooms', 'bathrooms', 'stories', 'parking','price']



df_test[num_vars] = scaler.transform(df_test[num_vars])
y_test = df_test.pop('price')

X_test = df_test
# Now let's use our model to make predictions.



# Creating X_test_new dataframe by dropping variables from X_test

X_test_new = X_test[X_train_new.columns]



# Adding a constant variable 

X_test_new = sm.add_constant(X_test_new)
# Making predictions

y_pred = lm.predict(X_test_new)
# Plotting y_test and y_pred to understand the spread.

fig = plt.figure()

plt.scatter(y_test,y_pred)

fig.suptitle('y_test vs y_pred', fontsize=20)              # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label
lm.params
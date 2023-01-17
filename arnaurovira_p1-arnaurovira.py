# some imports



from IPython.core.display import display, HTML

display(HTML("<style>.container { width:100% !important; }</style>"))



# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)

 

# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os



# to make this notebook's output stable across runs

np.random.seed(42)



# To plot pretty figures

%matplotlib inline

import matplotlib as mpl

import matplotlib.pyplot as plt

mpl.rc('axes', labelsize=14)

mpl.rc('xtick', labelsize=12)

mpl.rc('ytick', labelsize=12)

plt.rc('font', size=12) 

plt.rc('figure', figsize = (12, 5))



# Settings for the visualizations

import seaborn as sns

sns.set_style("whitegrid")

sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2,'font.family': [u'times']})



import pandas as pd

pd.set_option('display.max_rows', 25)

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 50)



# Ignore useless warnings (see SciPy issue #5998)

import warnings

warnings.filterwarnings(action="ignore", message="^internal gelsd")



# create output folder

if not os.path.exists('output'):

    os.makedirs('output')

if not os.path.exists('output/session1'):

    os.makedirs('output/session1')
## load data

train_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0) 

test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/test_set.csv',index_col=0)

# print the dataset size

print("There is", train_set.shape[0], "samples")

print("Each sample has", train_set.shape[1], "features")
# print the top elements from the dataset

train_set.head()
# As it can be seen the database contains several features, some of them numerical and some of them are categorical.

# It is important to check each of the to understand it.
# we can see the type of each features as follows

train_set.dtypes
# print those categorical features

train_set.select_dtypes(include=['object']).head()
# We can check how many different type there is in the dataset using the folliwing line

train_set["Type"].value_counts()
sns.countplot(y="Type", data=train_set, color="c")
sns.distplot(train_set["Price"])

plt.show()
## the features



features = ['Rooms','Landsize', 'BuildingArea', 'YearBuilt']

## DEFINE YOUR FEATURES

X = train_set[features].fillna(0)

y = train_set[['Price']]



## the model

# KNeighborsRegressor

from sklearn import neighbors

n_neighbors = 3 # you can modify this paramenter (ONLY THIS ONE!!!)

model = neighbors.KNeighborsRegressor(n_neighbors)



## fit the model

model.fit(X, y)



## predict training set

y_pred = model.predict(X)



## Evaluate the model and plot it

from sklearn.metrics import mean_squared_error, r2_score

print("----- EVALUATION ON TRAIN SET ------")

print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))

print("R^2: ",r2_score(y, y_pred))





plt.scatter(y, y_pred)

plt.xlabel('Price')

plt.ylabel('Predicted price');

plt.show()



## predict the test set and generate the submission file

X_test = test_set[features].fillna(0)

y_pred = model.predict(X_test)



df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



df_output.to_csv('output/session1/baseline.csv',index=False)
from sklearn import preprocessing

###I make a copy of both dataframes so the original doesn't change

train_set_change = train_set.copy()

test_set_change = test_set.copy()
###Process non-numerical data with few variants (did not work as well as i expected, so I decided not to use it)



###The idea was to address these non-numerical columns with few variants like this (h=0,u=1,v=2) and the most complex ones

###using the column mean (as I did in the next cell)



#lb_make = preprocessing.LabelEncoder()

#train_set_change['Type'] = lb_make.fit_transform(train_set_change['Type'])

#train_set_change['Method'] = lb_make.fit_transform(train_set_change['Method'])

#test_set_change['Type'] = lb_make.fit_transform(test_set_change['Type'])

#test_set_change['Method'] = lb_make.fit_transform(test_set_change['Method'])
###Process non-numerical data with lots of variants



###For each unique value, calculates this value's mean along the column and uses it instead

###of the non-numerical data (and postcode, since it's numerical but not related to price numerically speaking)



for column in train_set_change:

    if train_set_change[column].dtypes == 'object' or column=='Postcode':

        unique_list = list(train_set_change[column].unique())

        for item in unique_list:

            is_item = train_set_change[column]==item

            item_mean_price = train_set_change[is_item]['Price'].mean()

            train_set_change.loc[train_set_change[column] ==item, column] = item_mean_price

            test_set_change.loc[test_set_change[column] ==item, column] = item_mean_price

test_set_change.loc[test_set_change['CouncilArea'] == 'Moorabool', 'CouncilArea'] = test_set_change['CouncilArea'][0]
###Change biased/useless data



###I tried deleting outliers and substituting them by the closes value (second option worked better)

###I also tried different outliers limits (0.05, 0.1, 0.2 and 0.25) and the ones that worked the best were 0.1 and 0.9



for column in train_set_change:

    if column != 'Price' and train_set_change[column].dtypes != 'object':

        min_value,max_value = train_set_change[column].quantile([0.1,0.9])

        train_set_change.loc[train_set_change[column] < min_value, column] = min_value

        train_set_change.loc[train_set_change[column] > max_value, column] = max_value



for column in test_set_change:

    if column != 'Suburb' and column != 'Address' and column != 'SellerG' and train_set_change[column].dtypes != 'object':

        min_value,max_value = test_set_change[column].quantile([0.1,0.9])

        test_set_change.loc[test_set_change[column] < min_value, column] = min_value

        test_set_change.loc[test_set_change[column] > max_value, column] = max_value
###Instead of filling NaN with 0, fill them with the column mean value



for column in train_set_change:

    if train_set_change[column].dtypes != 'object':

        train_set_change[column].fillna(train_set_change[column].mean(), inplace=True)

    else:

        train_set_change[column].fillna(0, inplace=True)

        

for column in test_set_change:

    if test_set_change[column].dtypes != 'object':

        test_set_change[column].fillna(test_set_change[column].mean(), inplace=True)

    else:

        test_set_change[column].fillna(0, inplace=True)
###Normalize data between 0 and 1 (doesn't work as intended so I decided not to use it)



###Actual RMSE and R^2 values remain almost the same, but a 400k kaggle easily becomes a 1M score and i'm not sure why so i decided not to normalize



#x = train_set_change.loc[:, train_set_change.columns != 'Price'].values.astype(float)

#min_max_scaler = preprocessing.MinMaxScaler()

#x_scaled = min_max_scaler.fit_transform(x)

#train_set_change = pd.DataFrame(x_scaled,columns = train_set_change.loc[:, train_set_change.columns != 'Price'].columns)

#train_set_change = train_set_change.join(train_set['Price'])
## imports

from sklearn.linear_model import LinearRegression

from sklearn import neighbors



## the features

features = ['Rooms','Type','Distance','Postcode','Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt','CouncilArea','Lattitude','Longtitude','Regionname']

## DEFINE YOUR FEATURES

X = train_set_change[features].fillna(0)

y = train_set_change[['Price']]



## the model

# KNeighborsRegressor

from sklearn import neighbors

n_neighbors = 3 # you can modify this paramenter (ONLY THIS ONE!!!)

model = neighbors.KNeighborsRegressor(n_neighbors)



# LinealRegressor

#model = LinearRegression()



## fit the model

model.fit(X, y)



## predict training set

y_pred = model.predict(X)



## Evaluate the model and plot it

from sklearn.metrics import mean_squared_error, r2_score

print("----- EVALUATION ON TRAIN SET ------")

print("RMSE",np.sqrt(mean_squared_error(y, y_pred)))

print("R^2: ",r2_score(y, y_pred))





plt.scatter(y, y_pred)

plt.xlabel('Price')

plt.ylabel('Predicted price');

plt.show()



## predict the test set and generate the submission file

X_test = test_set_change[features].fillna(0)

y_pred = model.predict(X_test)



df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



df_output.to_csv('output/session1/baseline.csv',index=False)
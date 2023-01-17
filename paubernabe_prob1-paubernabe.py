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



#df_output.to_csv('output/session1/baseline.csv',index=False)
fig, ax = plt.subplots(figsize=(5, 5))

correlation = train_set.corr()

sns.heatmap(correlation, vmax=1, square=True)

plt.show()

correlation
from sklearn.preprocessing import LabelEncoder



#dummy

lb_make = LabelEncoder()

categories = ["Type", "Regionname"]

for category in categories:

    train_set[category] = lb_make.fit_transform(train_set[category])

    test_set[category] = lb_make.fit_transform(test_set[category])
cols = ['Landsize', 'BuildingArea', 'YearBuilt', 'Car', 'Distance', 'Propertycount', 'Rooms', 'Bedroom2', 'Bathroom']

for column in cols:

    train_set[column] = train_set[column].fillna(train_set[column].mean())

    test_set[column] = test_set[column].fillna(test_set[column].mean())

#OUTLIERS



cols = ['Landsize', 'BuildingArea', 'YearBuilt', 'Car', 'Distance', 'Propertycount', 'Rooms', 'Bedroom2', 'Bathroom']

for column in cols:

    quan_ba_low_tr = train_set[column].quantile(0.03)

    quan_ba_high_tr = train_set[column].quantile(0.97)

    quan_ba_low = test_set[column].quantile(0.03)

    quan_ba_high = test_set[column].quantile(0.97)

    maximum_in_quantile_tr = train_set[column].quantile(0.96)

    maximum_in_quantile = test_set[column].quantile(0.96)

    minimum_in_quantile_tr = train_set[column].quantile(0.04)

    minimum_in_quantile = test_set[column].quantile(0.04)



    train_set.loc[train_set[column] < quan_ba_low_tr, column] = minimum_in_quantile_tr

    train_set.loc[train_set[column] > quan_ba_high_tr, column] = maximum_in_quantile_tr



    test_set.loc[test_set[column] < quan_ba_low, column] = minimum_in_quantile

    test_set.loc[test_set[column] > quan_ba_high, column] = maximum_in_quantile



train_set
#POTENCIAR

train_set['Corr_sum'] = train_set['Rooms'] + train_set['Bedroom2'] + train_set['Bathroom'] + train_set['Car']

test_set['Corr_sum'] = test_set['Rooms'] + test_set['Bedroom2'] + test_set['Bathroom'] + test_set['Car']



train_set['Area_sum'] = train_set['Landsize'] + train_set['BuildingArea']

test_set['Area_sum'] = test_set['Landsize'] + test_set['BuildingArea']



cols = ['Landsize', 'BuildingArea', 'YearBuilt', 'Distance', 'Propertycount', 'Corr_sum', 'Rooms', 'Area_sum',

        'Type', 'Regionname']



#normalitzar

for col in cols:

    train_set[col] = (train_set[col] - train_set[col].min())/(train_set[col].max() - train_set[col].min())

    test_set[col] = (test_set[col] - test_set[col].min())/(test_set[col].max() - test_set[col].min())



fig, ax = plt.subplots(figsize=(5, 5))

correlation = train_set.corr()

sns.heatmap(correlation, vmax=.8, square=True)

plt.show()

correlation
## the features



features = ['Area_sum', 'YearBuilt', 'Distance', 'Propertycount', 'Corr_sum', 'Type', 'Regionname']

## DEFINE YOUR FEATURES

X = train_set[features].fillna(0)

y = train_set[['Price']]

## the model

# KNeighborsRegressor



from sklearn import neighbors



n_neighbors = 3 # you can modify this paramenter (ONLY THIS ONE!!!)



model = neighbors.KNeighborsRegressor(n_neighbors=n_neighbors)

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



df_output.to_csv('output/session1/problem_submission.csv',index=False)
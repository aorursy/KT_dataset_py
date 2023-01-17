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
## the features

features = ['Rooms','Bathroom','Car','Distance','Postcode','Landsize','BuildingArea','Propertycount','Lattitude','Longtitude','YearBuilt']

X = train_set[features].copy()



X['BuildingArea']=X['BuildingArea'].fillna(X['Landsize']-(X['Rooms']*50))

X['YearBuilt'] = X['YearBuilt'].fillna(X['YearBuilt'].mean())

X['Car'] = X['Car'].fillna(X['Rooms']/2)



#New feature: bathroom per bedroom

bxb = X['Bathroom']/X['Rooms']

X['BxB'] = bxb 



#New feature: Not builded land

fa = X['Landsize']-X['BuildingArea']

X['FreeArea'] = fa 



#Single column to multiples columns:Type

import pandas as pd

types = pd.get_dummies(train_set[['Type']])

X = pd.concat([X, types],axis=1)



ca = pd.get_dummies(train_set[['CouncilArea']])

X = pd.concat([X, ca,],axis=1)



rn = pd.get_dummies(train_set[['Regionname']])

X = pd.concat([X, rn],axis=1)



#Normalizing data

a, b = 0,1

x, y = X.min(), X.max()

X = (X - x) / (y - x) * (b - a) + a



y= train_set[['Price']]



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

X_test = test_set[features].copy()

X_test['BuildingArea']=X_test['BuildingArea'].fillna(X_test['Landsize']-(X_test['Rooms']*50))

X_test['YearBuilt'] = X_test['YearBuilt'].fillna(X_test['YearBuilt'].mean())

X_test['Car'] = X_test['Car'].fillna(X_test['Rooms']/2)



#New feature: bathroom per bedroom

bxb = X_test['Bathroom']/X_test['Rooms']

X_test['BxB'] = bxb 



#New feature: Not builded land

fa = X_test['Landsize']-X_test['BuildingArea']

X_test['FreeArea'] = fa 



#Single column to multiples columns:Type

types = pd.get_dummies(test_set[['Type']])

X_test = pd.concat([types, X_test], axis=1)



ca = pd.get_dummies(test_set[['CouncilArea']])

X_test = pd.concat([ca, X_test], axis=1)



rn = pd.get_dummies(test_set[['Regionname']])

X_test = pd.concat([rn, X_test], axis=1)



a, b = 0,1

x, y = X.min(), X.max()

X = (X - x) / (y - x) * (b - a) + a



y_pred = model.predict(X_test)



df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



df_output.to_csv('output/session1/baseline.csv',index=False)
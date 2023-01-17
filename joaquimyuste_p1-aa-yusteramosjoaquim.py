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
##NO TOCAR
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn import neighbors
from scipy.stats import zscore

## data cleaning

train_set_cleaned = train_set.copy()


### eliminamos el ruido
train_set_cleaned = train_set_cleaned[train_set['Price'].between(train_set_cleaned['Price'].quantile(0.01), train_set_cleaned['Price'].quantile(0.99))]


### convertimos categorical data to numerical tanto en el train como en el test data
cat_columns = train_set_cleaned.select_dtypes(['object']).columns
train_set_cleaned[cat_columns] = train_set_cleaned[cat_columns].applymap(hash)

train_set_cleaned = train_set_cleaned.fillna(train_set_cleaned.mean())


test_set_cleaned = test_set.copy()

cat_columns = test_set_cleaned.select_dtypes(['object']).columns
test_set_cleaned[cat_columns] = test_set_cleaned[cat_columns].applymap(hash)

test_set_cleaned = test_set_cleaned.fillna(test_set_cleaned.mean())


## the features

features = ['Suburb','Type','Regionname','Rooms','Landsize', 'BuildingArea', 'YearBuilt', 'Car', 'Bedroom2', 'Bathroom','Postcode']



X = train_set_cleaned[features]
y = train_set_cleaned[['Price']]

## the model

n_neighbors = 2 # you can modify this paramenter (ONLY THIS ONE!!!)
#model = LinearRegression()
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

X_test = test_set_cleaned[features].fillna(0)
y_pred = model.predict(X_test)

df_output = pd.DataFrame(y_pred)
df_output = df_output.reset_index()
df_output.columns = ['index','Price']

df_output.to_csv('output/session1/baseline.csv',index=False)

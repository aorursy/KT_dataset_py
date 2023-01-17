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
categorical = train_set.columns[train_set.dtypes == object].tolist()

categorical
total_col = 0

for cf in categorical:

    n_values = len(train_set[cf].unique())

    total_col += n_values

    print("Categorical feature ", cf, " has ", n_values, " different values")



print("\nIf we do dummies the dataframe will result with ", total_col, " additional columns")
categorical_features_mask = train_set.dtypes == object

categorical_features = list(train_set.columns[categorical_features_mask])

categorical_features.append("Postcode") # We may also include Postcode



# Codification in both sets, train and test

for cf in categorical_features:

    fe = train_set.groupby(cf).size()/len(train_set)

    train_set.loc[:, str(cf)+"_freqencode"] = train_set[cf].map(fe)

    

    fe2 = test_set.groupby(cf).size()/len(test_set)

    test_set.loc[:, str(cf)+"_freqencode"] = test_set[cf].map(fe2)

    

bencode_features_mask = train_set.dtypes != object

bencode_features = list(train_set.columns[bencode_features_mask])

bencode_features.remove('Price')

bencode_features.remove('Postcode')

print("After encoding: ", bencode_features)
train_set.corr()
features = ['Rooms', 'Distance', 'Lattitude', 'Longtitude', 'Suburb_freqencode', 'Type_freqencode']
def find_outliers(x):

    q1 = np.percentile(x, 25)

    q3 = np.percentile(x, 75)

    iqr = q3 - q1

    floor = q1 - 1.5 * iqr

    ceiling = q3 + 1.5 * iqr

    outlier_indices = list(x.index[(x < floor) | (x > ceiling)])



    return outlier_indices
#Useless fillna operation

X = train_set[features].fillna(train_set[features].median())

X.describe()
print("Maximum number of rooms before outlier detection: ", max(X['Rooms']))

print("Minimum number of rooms before outlier detection: ", min(X['Rooms']))
for f in features:

    while len(find_outliers(X[f])) != 0:

        temp_idx = find_outliers(X[f])

        for idx in temp_idx:

            X.loc[idx, f] = X[f].median()
print("Maximum number of rooms after outlier detection: ", int(max(X['Rooms'])))

print("Minimum number of rooms after outlier detection: ", int(min(X['Rooms'])))
for f in features:

    print("The number of outliers in feature", f,"has decreased to ", len(find_outliers(X[f])))
#Distribution on Rooms

sns.distplot(X['Rooms'])

plt.show()
#Distribution on Distance

sns.distplot(X['Distance'])

plt.show()
#Distribution on Lattitude

sns.distplot(X['Lattitude'])

plt.show()
#Distribution on Longtitude

sns.distplot(X['Longtitude'])

plt.show()
#Distribution on Suburb_freqencode

sns.distplot(X['Suburb_freqencode'])

plt.show()
#Distribution on Suburb_freqencode

sns.distplot(X['Type_freqencode'])

plt.show()
from sklearn.preprocessing import StandardScaler

std = StandardScaler()



## DEFINE YOUR FEATURES

X = std.fit_transform(X[features].fillna(X[features].median()))

y = train_set[['Price']]



## the model

# KNeighborsRegressor

from sklearn import neighbors

n_neighbors = 4 # you can modify this paramenter (ONLY THIS ONE!!!)

model = neighbors.KNeighborsRegressor(n_neighbors)



## fit the model

model.fit(X, y)



## predict training set

y_pred = model.predict(X)



## Evaluate the model and plot it

from sklearn.metrics import mean_squared_error, r2_score

print("----- EVALUATION ON TRAIN SET ------")

print("RMSE",np.sqrt(mean_squared_error(y, y_pred)), " n_neighbours", n_neighbors)

print("R^2: ",r2_score(y, y_pred))

    

plt.scatter(y, y_pred)

plt.xlabel('Price')

plt.ylabel('Predicted price');

plt.show()



## predict the test set and generate the submission file

X_test = std.fit_transform(test_set[features].fillna(test_set[features].median()))

y_pred = model.predict(X_test)



df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



df_output.to_csv('output/session1/testingknn_sunday.csv',index=False)
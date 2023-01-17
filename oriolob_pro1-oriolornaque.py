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

#train_set = pd.read_csv('dataset/housing-snapshot/train_set.csv',index_col=0) 

#test_set = pd.read_csv('dataset/housing-snapshot/test_set.csv',index_col=0)

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
from sklearn import neighbors

from sklearn.linear_model import LinearRegression

from sklearn import preprocessing

from pandas.api.types import is_object_dtype, is_categorical

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error, r2_score
def preprocess_data(data, dummify=True, fillna=True, normalize=True):

    from sklearn import preprocessing

    if dummify:

        categoricals = data.select_dtypes(include=['object'])

        dummies = pd.get_dummies(categoricals)

        data = pd.concat([data, dummies], axis='columns')

        data = data.drop(categoricals.columns, axis='columns')

    if fillna:

        data = data.fillna(data.mean())

    if normalize:

        data = preprocessing.normalize(data)

    return data



def dummify(data):

    categoricals = data.select_dtypes(include=['object'])

    dummies = pd.get_dummies(categoricals)

    data2 = pd.concat([data, dummies], axis='columns')

    data2 = data2.drop(categoricals.columns, axis='columns')

    return data2



def dummify_column(data, column):

    dummies = pd.get_dummies(column)

    data2 = pd.concat([data, dummies], axis='columns')

    data2 = data2.drop(column, axis='columns')

    return data2



def fillna(data):

    data2 = data.fillna(data.mean())

    return data2



def normalize(data):

    from sklearn import preprocessing

    data2 = preprocessing.normalize(data)

    return data2
## load data

#train_set = pd.read_csv('dataset/housing-snapshot/train_set.csv',index_col=0) 

#test_set = pd.read_csv('dataset/housing-snapshot/test_set.csv',index_col=0)

train_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0) 

test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/test_set.csv',index_col=0)
#encoder = preprocessing.OneHotEncoder()

#encoder.fit_transform(train_set[['Postcode']])

#encoder.categories_

pd.get_dummies(train_set.Postcode)

train_set.Postcode
sns.catplot(x="Price", y="CouncilArea", data=train_set, kind="violin")

plt.show()
sns.catplot(x="Price", y="Type", data=train_set, kind="violin")

plt.show()
sns.catplot(x="Price", y="Suburb", data=train_set)

plt.show()
sns.catplot(x="Price", y="Regionname", data=train_set, kind="violin")

plt.show()
sns.jointplot(data=train_set, x="Price", y="Distance")

plt.show()
sns.jointplot(data=train_set, x="Price", y="Rooms")

plt.show()
price_vs_area = train_set[["Price", "BuildingArea"]]

sns.jointplot(data=price_vs_area.fillna(price_vs_area.mean()), x="Price", y="BuildingArea")

plt.show()
sns.jointplot(data=train_set, x="Price", y="YearBuilt")

plt.show()
price_vs_landsize = train_set[["Price", "Landsize"]]

sns.jointplot(data=price_vs_landsize.fillna(price_vs_landsize.mean()), x="Price", y="Landsize")

plt.show()
price_vs_bathroom = train_set[["Price", "Bathroom"]]

sns.jointplot(data=price_vs_bathroom.fillna(price_vs_bathroom.mean()), x="Price", y="Bathroom")

plt.show()
features = ['Distance', 'YearBuilt', 'Rooms', 'Bathroom', 'Type', 'Regionname', 'CouncilArea']



X = dummify(train_set[features])

#X = dummify_column(X, train_set.Postcode)

X = fillna(X)

X = normalize(X)

y = train_set[['Price']]



#model = LinearRegression()

model = neighbors.KNeighborsRegressor(n_neighbors=5)



scores = cross_val_score(model, X, y, cv=10)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
# Own cross validation

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
model.fit(X_train, y_train)



y_pred = model.predict(X_train)



print("----- EVALUATION ON TRAIN SET ------")

print("RMSE",np.sqrt(mean_squared_error(y_train, y_pred)))

print("R^2: ",r2_score(y_train, y_pred))

plt.scatter(y_train, y_pred)

plt.xlabel('Price')

plt.ylabel('Predicted price');

plt.show()



y_pred = model.predict(X_test)



print("----- EVALUATION ON TEST SET ------")

print("RMSE",np.sqrt(mean_squared_error(y_test, y_pred)))

print("R^2: ",r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)

plt.xlabel('Price')

plt.ylabel('Predicted price');

plt.show()
## retrain with all the data, not just a split

# https://stats.stackexchange.com/questions/331250/how-to-train-the-final-model-after-cross-validation

model.fit(X, y)



## predict the test set and generate the submission file



X_test = dummify(test_set[features])

X_test = fillna(X_test)

X_test = normalize(X_test)

#y_test = test_set[['Price']]



y_pred = model.predict(X_test)



df_output = pd.DataFrame(y_pred)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



#df_output.to_csv('output/session1/baseline.csv',index=False)

df_output.to_csv('baseline.csv',index=False)
# check that the file has been created

import time



if os.path.exists("baseline.csv"):

    statbuf = os.stat("baseline.csv")

    print("last modified: %s" % time.ctime(os.path.getmtime("baseline.csv")))
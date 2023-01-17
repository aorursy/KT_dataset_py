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

test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0) 

train_set.describe()
from scipy.stats import zscore

fig, axes = plt.subplots(1, 2,figsize=(25, 6))



axes[0].set_title("normal feature")

sns.distplot(train_set["Landsize"], ax=axes[0])



train_set['z_score_Landsize']=zscore(train_set['Landsize'])

train_set = train_set.loc[train_set['z_score_Landsize'].abs()<=3]



axes[1].set_title("with z-score")

sns.distplot(train_set["Landsize"], ax=axes[1])



plt.show()
fig, axes = plt.subplots(1, 3,figsize=(25, 6))



axes[0].set_title("normal feature")

sns.distplot(train_set["Car"], ax=axes[0])





axes[1].set_title("with nanmean")

train_set["Car"] = train_set["Car"].fillna(np.nanmean(train_set["Car"]))

sns.distplot(train_set["Car"], ax=axes[1])





axes[2].set_title("with z-score")

train_set['z_score_Car']=zscore(train_set['Car'])

train_set = train_set.loc[train_set['z_score_Car'].abs()<=3]

sns.distplot(train_set["Car"], ax=axes[2])



plt.show()
fig, axes = plt.subplots(1, 2,figsize=(25, 6))



axes[0].set_title("normal feature")

sns.distplot(train_set["Rooms"], ax=axes[0])



train_set['z_score_Rooms']=zscore(train_set['Rooms'])

train_set = train_set.loc[train_set['z_score_Rooms'].abs()<=3]



axes[1].set_title("with z-score")

sns.distplot(train_set["Rooms"], ax=axes[1])





plt.show()
fig, axes = plt.subplots(1, 3,figsize=(25, 6))



axes[0].set_title("normal feature")

sns.distplot(train_set["YearBuilt"], ax=axes[0])



axes[1].set_title("with nanmean")

train_set["YearBuilt"] = train_set["YearBuilt"].fillna(np.nanmean(train_set["YearBuilt"]))

sns.distplot(train_set["YearBuilt"], ax=axes[1])





axes[2].set_title("with z-score")

train_set['z_score_YearBuilt']=zscore(train_set['YearBuilt'])

train_set = train_set.loc[train_set['z_score_YearBuilt'].abs()<=3]

sns.distplot(train_set["YearBuilt"], ax=axes[2])



plt.show()
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
def dateStr(d):

    

    dia,mes,anny = d.split('/')



    diah = int(dia)*24

    mesh = int(mes)*30*24

    anyh = int(anny)*365*24



    suma = diah + mesh + anyh

    return suma
from scipy.stats import zscore



## load data

train_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/train_set.csv',index_col=0) 

test_set = pd.read_csv('/kaggle/input/mlub-housing-house-prediction/test_set.csv',index_col=0)





## the features

features = ['Rooms','Distance','Bedroom2','Bathroom','TypeDumm_t','TypeDumm_u','RegionnameDumm_Eastern Victoria',  'RegionnameDumm_Northern Metropolitan','RegionnameDumm_Northern Victoria','RegionnameDumm_South-Eastern Metropolitan',

           'RegionnameDumm_Southern Metropolitan', 'RegionnameDumm_Western Metropolitan', 'RegionnameDumm_Western Victoria']



#test_set['Date'] = test_set['Date'].apply(dateStr)

#train_set['Date'] = train_set['Date'].apply(dateStr)





# Dummy features to pass from categorical to numeral (train).

types_dumm_train = pd.get_dummies(train_set.Type, drop_first=True, prefix='TypeDumm')

train_set = pd.concat([train_set, types_dumm_train], axis=1)



regionname_dumm_train = pd.get_dummies(train_set.Regionname, drop_first=True, prefix='RegionnameDumm')

train_set = pd.concat([train_set, regionname_dumm_train], axis=1)





# Dummy features to pass from categorical to numeral (test).

types_dumm_test = pd.get_dummies(test_set.Type, drop_first=True, prefix='TypeDumm')

test_set = pd.concat([test_set, types_dumm_test], axis=1)



regionname_dumm_test = pd.get_dummies(test_set.Regionname, drop_first=True, prefix='RegionnameDumm')

test_set = pd.concat([test_set, regionname_dumm_test], axis=1)





# Fill NaN values with the mean values of that feature

train_set = train_set.fillna(train_set.mean())





# Removing some outliers with zscore method

train_set['z_score_Landsize']=zscore(train_set['Landsize'])

train_set = train_set.loc[train_set['z_score_Landsize'].abs()<=3]



train_set['z_score_BuildingArea']=zscore(train_set['BuildingArea'])

train_set = train_set.loc[train_set['z_score_BuildingArea'].abs()<=3]



train_set['z_score_YearBuilt']=zscore(train_set['YearBuilt'])

train_set = train_set.loc[train_set['z_score_YearBuilt'].abs()<=3]



# Manual removing of outliers from Bathroom

train_set = train_set.drop(train_set[train_set.Bathroom > 6].index)





# train set with normalization and only with the features that have selected

X=(train_set[features]-train_set[features].min())/(train_set[features].max()-train_set[features].min())

y = train_set[['Price']]



## the model

# KNeighborsRegressor

from sklearn import neighbors

n_neighbors = 14 # I've chosen a high value to avoid noise effects

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



# Fill NaN values with the mean values of that feature

X_test1 = test_set[features].fillna(test_set.mean())



# Test set with normalization and only with the features that have selected

X_test1=(X_test1-X_test1.min())/(X_test1.max()-X_test1.min())



# predict training set

y_pred1 = model.predict(X_test1)



df_output = pd.DataFrame(y_pred1)

df_output = df_output.reset_index()

df_output.columns = ['index','Price']



df_output.to_csv('output/session1/baseline.csv',index=False)

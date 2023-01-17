# global setting to print all the outputs of a code cell (not just the last one)

from IPython.core.interactiveshell import InteractiveShell  

InteractiveShell.ast_node_interactivity = "all"
#### flag for only executing necessary code (without time-consuming outputs)

speed_up = True



#### flag for output only useful for debugging purposes



debug = True
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt 

import seaborn as sns

import warnings



pd.options.mode.chained_assignment = None

warnings.filterwarnings("ignore", category=DeprecationWarning)



# exploratory data analysis (EDA) of a Pandas Dataframe

import pandas_profiling



import os

#print(os.listdir("../input"))

# read training and test set 

train = pd.read_csv('../input/bike-sharing-demand/train.csv')

test = pd.read_csv('../input/bike-sharing-demand/test.csv')
# create EDA report - takes a moment!

if speed_up != True:

  train.profile_report()
if speed_up != True:

    type(train)  # display type of train

    train.shape  # display shape of train

    train.dtypes # inspect types of featrues
_=sns.boxplot(x = 'count', data = train)

_=plt.title("Boxplot of target variable \"count\"")

plt.show()
#### this code section is devoted to calculate the outliers of target variable 'count'

Q1 = train['count'].quantile(0.25)

Q3 = train['count'].quantile(0.75)

IQR = Q3 - Q1

outliers_high = len(train[(train['count'] > (Q3 + 1.5 * IQR))])

outliers_low  = len(train[(train['count'] < (Q1 - 1.5 * IQR)) ])

outliers_all  = outliers_high + outliers_low

print("Outliers exceeding 'maximum' ( > Q3 + 1.5 * IQR): ", outliers_high)

print("Outliers below 'minimum' ( < Q1 - 1.5 * IQR)):    ", outliers_low)

print((outliers_all/len(train))*100)
# use "_" as temporary object to suppress output of function calls

_=sns.distplot(train['count'], bins=25)

_=plt.title("Probability Density of target variable \"count\"")

_=plt.ylabel('Probability Density')

# Limits for the Y axis

_=plt.xlim(0,1100)

plt.show()
# The boxcox() SciPy function implements the Box-Cox method. (Data must be positiv)

# The Box-Cox method is a data transform method that is able to perform a range of power transforms.

# More than that, it can be configured to evaluate a suite of transforms automatically and select a best fit.

# It takes an argument, called lambda, that controls the type of transform to perform.

# Below are some common values for lambda:

# lambda = -1. is a reciprocal transform.

# lambda = -0.5 is a reciprocal square root transform.

# lambda = 0.0 is a log transform.

# lambda = 0.5 is a square root transform.

# lambda = 1.0 is no transform.
# drop reduandant columns 'registered'and 'atemp'

train.drop(['atemp','registered', 'casual'],axis=1,inplace=True)

# same for test ('registered' and 'casual' are not contained in test set anyway)

test.drop(['atemp'],axis=1,inplace=True)
if speed_up != True:

    train.shape  # display shape of train

    train.dtypes # inspect types of featrues
#### this code section is devoted to extract features 'year', 'month' and 'hour' information from 'datetime'

#### the no longer needed 'datetime' column will be dropped afterwards



# convert categorical feature 'datetime' to type datetime

train.datetime = pd.to_datetime(train.datetime)



# extract needed features from date information

train['year'] = train['datetime'].dt.year

train['month'] = train['datetime'].dt.month          # january=0 - december=12

train['day_of_week'] = train['datetime'].dt.weekday  # monday=0 - sunday=6

train['hour'] = train['datetime'].dt.hour

#drop column "datetime", which is no longer needed

train.drop(['datetime'],axis=1,inplace=True)



#### same for test data set

test.datetime = pd.to_datetime(test.datetime)

test['year'] = test['datetime'].dt.year

test['month'] = test['datetime'].dt.month

test['day_of_week'] = test['datetime'].dt.weekday  # monday=0 - sunday=6

test['hour'] = test['datetime'].dt.hour

test.drop(['datetime'],axis=1,inplace=True)



if debug != True:

    train.dtypes

    train.head()
if speed_up != True:

    # first make shure that these 'categorical' feaures really only contains

    # the values as defined in the data description

    print("Check for value count for feature 'weather':")

    my_value_counts = train['weather'].value_counts(sort=False)

    my_value_counts.rename_axis('weather categories').reset_index(name='counts')



    print("Check for value count for feature 'season':")

    my_value_counts = train['season'].value_counts(sort=False)

    my_value_counts.rename_axis('season categories').reset_index(name='counts')



    print("Check for value count for feature 'holiday':")

    my_value_counts = train['holiday'].value_counts(sort=False)

    my_value_counts.rename_axis('holiday categories').reset_index(name='counts')



    print("Check for value count for feature 'workingday':")

    my_value_counts = train['workingday'].value_counts(sort=False)

    my_value_counts.rename_axis('workingday categories').reset_index(name='counts')
# convert features to type categorical

train['weather'] = train['weather'].astype('category')

train['season'] = train['season'].astype('category')

train['holiday'] = train['holiday'].astype('category')

train['workingday'] = train['workingday'].astype('category')



# same for test

test['weather'] = test['weather'].astype('category')

test['season'] = test['season'].astype('category')

test['holiday'] = test['holiday'].astype('category')

test['workingday'] = test['workingday'].astype('category')

if debug != True:

    train.dtypes

    train.head()
if speed_up != True:

    train.profile_report()   # takes a moment to execute
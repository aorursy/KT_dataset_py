# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.metrics import accuracy_score

from matplotlib import pyplot as plt

import seaborn as sns

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline

from sklearn.linear_model import (

    LinearRegression,

    Ridge,

    Lasso

)

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



def min_max_normalization(x):

    x_min = x.min()

    x_max = x.max()

    x_norm = (x - x_min) / ( x_max - x_min)

    return x_norm

        

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/pchs-data-science-covid-19-competition/Train.csv')

test = pd.read_csv('../input/pchs-data-science-covid-19-competition/Test.csv')
train.corr()
train.isnull().sum()
cat_list1 = {*train.State, *test.State}

cat_list2 = {*train.County, *test.County}

# 方法1

train.State = train.State.astype(pd.CategoricalDtype(cat_list1))

# 方法2

test.State = pd.Categorical(test.State, cat_list1)

# 方法1

train.County = train.County.astype(pd.CategoricalDtype(cat_list2))

# 方法2

test.County = pd.Categorical(test.County, cat_list2)

training = pd.get_dummies(train, columns=["State"], drop_first=True)

tester = pd.get_dummies(test, columns=["State"], drop_first=True)

training
##training.columns.str.contains('County')

##training.iloc[:, training.columns.str.contains('County')]
training["Population 2018"] = min_max_normalization(training["Population 2018"])*100

training["Median Household Income 2018 ($)"] = min_max_normalization(training["Median Household Income 2018 ($)"])*100
from sklearn.linear_model import LogisticRegression

cols = ["Population 2018", "Unemployment Rate 2018 (%)", "Poverty 2018 (%)", "Confirmed Cases Per 100,000 people", "Deaths Per 100,000 people", "Mortality Rate (%)", "White Alone (%)", "Black Alone (%)", "Native American Alone (%)", "Asian Alone (%)", "Hispanic (%)", "Less than a High School Diploma (%)", "Only a High School Diploma (%)", "Some College/Associate's Degree (%)", "Bachelor's Degree or Higher (%)"]

X = training.drop(['Unnamed: 0', 'FIPS', 'County', 'Confirmed Cases', 'Confirmed Deaths', 'Confirmed Cases Per 100,000 people', 'Deaths Per 100,000 people', 'Mortality Rate (%)'], axis=1)

y1 = training['Confirmed Cases']

y2 = training['Confirmed Deaths']

# Build a logreg and compute the feature importances

model1 = LogisticRegression(max_iter=1000)

model2 = LogisticRegression(max_iter=1000)

# create the RFE model and select 8 attributes

model1.fit(X,y1)

model2.fit(X,y2)
from sklearn.metrics import mean_squared_error

y1_train_predict = model1.predict(X)

mean_squared_error(y1, y1_train_predict)
y1_train_predict
from sklearn.metrics import mean_squared_error

y2_train_predict = model2.predict(X)

mean_squared_error(y2, y2_train_predict)
y2_train_predict
model3 = LogisticRegression(max_iter=1000)

X['confirm cases'] = min_max_normalization(y1_train_predict)*100

model3.fit(X,y2)

y3_train_predict = model3.predict(X)

mean_squared_error(y2, y3_train_predict)
test.isnull().sum()
tester["Population 2018"] = min_max_normalization(tester["Population 2018"])*100

tester["Median Household Income 2018 ($)"] = min_max_normalization(tester["Median Household Income 2018 ($)"])*100

tester
X_test = tester.drop(['Unnamed: 0', 'FIPS', 'County'], axis=1)

print(X_test.dtypes)

test1_predicted = model1.predict(X_test)

test2_predicted = model2.predict(X_test)

X_test['confirm cases'] = min_max_normalization(test1_predicted)*100

test3_predicted = model3.predict(X_test)
test1_predicted
test2_predicted
test3_predicted
sub = pd.read_csv('../input/pchs-data-science-covid-19-competition/Sample.csv')

sub['Confirmed'] = list(map(int, test1_predicted))

sub['Deaths'] = list(map(int, test3_predicted))

sub.to_csv('submission.csv', index=False)

sub
# An academic exercise to predict sale price in auctions. This dataset is used as Lesson 1 in the Fast.AI intro to ML for coders course. Thanks to Fast.AI for the code.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#All the standard fast ai and other imports such as sklearn, pandas.
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display

from sklearn import metrics
#reading the trainandvalid.csv file into dataframe df_raw while parsing a datefield in saledate. This will help us extract more features from the date field.
df_raw = pd.read_csv("../input/TrainAndValid.csv", low_memory=False, 
                     parse_dates=["saledate"])
#Defining a display all function in order to ensure forcing of display up to a certain number of rows. 
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
display_all(df_raw.tail().T)
display_all(df_raw.describe(include='all').T)
#Converting sale price into a log form, and replacing the column with this new log scale price column.
df_raw.SalePrice = np.log(df_raw.SalePrice)
df_raw.SalePrice
#Extracting more features from the date field. This function add_datepart() takes a date dtype and strips it into multiple features or columns such as weekend, weekday.
# and several other such useful date slices. This is quite useful and according to Jeremy Howard of Fast.ai - the more features you can feed a model, the better
# basically the "curse of dimensionality" is a myth with real-world data since 
add_datepart(df_raw, 'saledate')

#Quickly checking to ensure that there are features extracted from the date field of 'saledate' such as saleYear
df_raw.saleYear.head()
#Check the dataframe df_raw once more. Notice all the new date fields in the dataframe.
df_raw.head()
#Change columns of categorical or non-continuous string data into columns of categorical values. Also specify order for the categorical variables
train_cats(df_raw)
df_raw.UsageBand.cat.categories
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
#Change to codes rather than just these values
df_raw.UsageBand = df_raw.UsageBand.cat.codes
#Count missing values
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))

#Replace categories with numerical codes
df, y, nas = proc_df(df_raw, 'SalePrice')
#Pass dataframe to random forest
m = RandomForestRegressor(n_jobs=-1)
%timeit m.fit(df, y)
m.score(df,y)
#This is purely an exercise. This is from the Fast.AI course, and provided by them. I have used this for learning purposes. 
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
#We just set up a train and validation split. Now we define a root mean square error function and a function to print the score. 
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
m = RandomForestRegressor(n_jobs=-1)
%time m.fit(X_train, y_train)
print_score(m)

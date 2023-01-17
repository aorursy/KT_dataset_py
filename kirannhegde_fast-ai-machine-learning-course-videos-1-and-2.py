# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

#Jupyter notebook tricks
#https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/

#https://ipython.readthedocs.io/en/stable/config/extensions/autoreload.html
#The below 2 lines will automatically reload any changed modules before executing any line of code.
#load_ext is an IPython magic command
#More about magic commands:
#1)https://ipython.org/ipython-doc/3/interactive/tutorial.html
#2)https://jakevdp.github.io/PythonDataScienceHandbook/01.03-magic-commands.html
#3)https://ipython.org/ipython-doc/3/interactive/reference.html - good explanation of magic commands
#autoreload is a IPython extension to automatically reload modules.
!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887
!apt update && apt install -y libsm6 libxext6
%load_ext autoreload
%autoreload 2

#The below line is used to plot charts inline in the notebook, instead of having the charts displayed in a seperate window.
%matplotlib inline

import pandas as pd
import numpy as np

#Import the necessary libraries
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import metrics
from fastai.imports  import *
from fastai.structured  import *

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#We are executing a shell command here. You can do that by using the ! character before the command as
#shown here. You can execute any command you want here by preceding the command with the ! character
#Format: !<command to execute>
PATH = "../input/"
!ls {PATH}
#Read the contents of train.csv into a dataframe using the Pandas library
df_raw = pd.read_csv(f'{PATH}TrainAndValid.csv',low_memory=False,parse_dates=["saledate"])
#Lets look at the top 5 rows using the Pandas DataFrame head() method
df_raw.head()
#The info method is useful to get a quick description of the data(# of columns, #of rows,datatypes of each column )
df_raw.info()
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
display_all(df_raw.tail().T)
display_all(df_raw.describe(include='all').T)
df_raw.SalePrice = np.log(df_raw.SalePrice)
#n_jobs=-1 indicates that the algorithm should use parallelism as part of its fit/predict phases. With n_jobs=-1
#scikit-learn will use all the CPU's available.
#For more information on this parameter, you can take a look at: http://scikit-learn.org/stable/glossary.html#term-n-jobs
#m = RandomForestRegressor(n_jobs=-1)
# The following code is supposed to fail due to string values in the input data
#m.fit(df_raw.drop('SalePrice',axis=1),df_raw.SalePrice)
add_datepart(df_raw,'saledate')
df_raw.saleYear.head()

#train_cats will not change the way the dataframe looks but behind the scenes it assign numbers to each
#of the categories.
train_cats(df_raw)
df_raw.UsageBand.cat.categories

#Check the columns in the dataframe
df_raw.columns

#There is a kind of categorical variable called “ordinal”. An ordinal categorical variable has some kind of order (e.g. “Low” < “Medium” < “High”). 
#Random forests are not terribly sensitive for that fact, but it is worth noting.

df_raw.UsageBand.cat.set_categories(['High','Medium','Low'],ordered=True,inplace=True)
df_raw.UsageBand = df_raw.UsageBand.cat.codes
#https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.isnull.html
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
os.makedirs('tmp',exist_ok=True)
df_raw.to_feather('tmp/bulldozers-raw')
df_raw = pd.read_feather('tmp/bulldozers-raw')
df, y, nas = proc_df(df_raw,'SalePrice')
m=RandomForestRegressor(n_jobs=-1)
m.fit(df,y)
m.score(df,y)
#With DataFrame, slicing inside of [] slices the rows. This is provided largely as a convenience since it is such a common operation.
#https://pandas.pydata.org/pandas-docs/stable/indexing.html

def split_vals(a,n): return a[:n].copy(), a[n:].copy()

n_valid = 12000  # same as Kaggle's test set size
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(df_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
def rmse(x,y): return math.sqrt(((x - y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train),y_train), rmse(m.predict(X_valid),y_valid),
          m.score(X_train,y_train), m.score(X_valid,y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


m = RandomForestRegressor(n_jobs=-1)
%time (m.fit(X_train,y_train))
print_score(m)
df_trn, y_trn, nas = proc_df(df_raw,'SalePrice', subset=30000, na_dict=nas)
X_train, _ = split_vals(df_trn,20000)
y_train, _ = split_vals(y_trn,20000)

m = RandomForestRegressor(n_jobs=-1)
%time (m.fit(X_train,y_train))
print_score(m)
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train,y_train)
print_score(m)
draw_tree(m.estimators_[0], df_trn, precision=3)
#Here we have removed the depth parameter to see if that makes a difference
#As you can see, the R2 is better than the earlier R2. However, its still not up to the mark.
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
#Good explanation of slicing multi dimensional numpy arrays can be found in the book: Python for Data Analysis by Wes McKinney. Refer Chapter 4
preds = np.stack([t.predict(X_valid) for t in m.estimators_]) 
preds[:,0],np.mean(preds[:,0]),y_valid[0]
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);


m = RandomForestRegressor(n_estimators=20, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=80, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
#Possible explanation as to why the oob score is better here than the R2 score of the validation set, whereas Jeremy says that it should generally be lower
#https://forums.fast.ai/t/oob-then-and-now-2017-11-vs-2018-10/23913
#The below OOB score also proves that the validation set time difference is making a difference here. The OOB score was calculated on data points in the same time range
#and we got a higher OOB score. This proves that the time difference in the validation set is what is making the difference when compared to the training data.

m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')
X_train, X_valid = split_vals(df_trn, n_trn)
y_train, y_valid = split_vals(y_trn, n_trn)
set_rf_samples(20000)
m = RandomForestRegressor(n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
reset_rf_samples()
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)
%time m.fit(X_train, y_train)
print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, 
                          n_jobs=-1, oob_score=True) 
%time m.fit(X_train, y_train) 
print_score(m)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3,max_features=0.5, n_jobs=-1, oob_score=True) 
m.fit(X_train, y_train)
print_score(m)
!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887



!apt update &amp;&amp; apt install -y libsm6 libxext6
#Load "autoreload" extension for automatically reloading imported modules before each code execution

%load_ext autoreload

#Reload all modules except those specified in %aimport (none in this case)

%autoreload 2



#Display matplotlib plots

%matplotlib inline
#Import fastai modules

#Does not follow PEP8 style (wildcard "*" imports)

#"Data science is not software engineering", "follow prototyping best practices", "interactive and iterative"



#Import all other relevant modules

from fastai.imports import *

#Import all relevant functions

from fastai.structured import *



#Import "DataFrameSummary" class which extends Pandas dataframe's describe() method

from pandas_summary import DataFrameSummary

#Import scikit-learn's classes for building random forests models (regression and classification)

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

#Import IPython's display() function for displaying Python objects

from IPython.display import display



#Import scikit-learn's metrics module for measuring model performance

from sklearn import metrics
#Fix

import feather
#Path for directory containing input data

PATH = "../input/"
#!ls: execute shell command in notebook (list files in current working directory)

#{PATH}: pass python variable (PATH) into shell command

!ls {PATH}
#We have already imported pandas as pd in "from fastai.imports import *"

#"low_memory=False": Avoid mixed type inference (from internally processing the csv file in chunks)

#"parse_dates=["saledate"]": parse the "saledate" column as a date column

df_raw = pd.read_csv(f'{PATH}train/Train.csv', low_memory=False, parse_dates=["saledate"])
#Function for displaying all the columns of the dataframe

def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)
#We want to have an overview of all the variables

#Take the last 5 (default value) rows in the dataframe, transpose and display it

display_all(df_raw.tail().T)
#Get descriptive statistics for all columns

display_all(df_raw.describe(include='all').T)
#Get natural logarithm of SalePrice

df_raw.SalePrice = np.log(df_raw.SalePrice)
#Create model object

#n_jobs = -1: use all CPUs

m = RandomForestRegressor(n_jobs=-1)

#Build model

#The following code is supposed to fail due to string values in the input data

m.fit(df_raw.drop('SalePrice', axis=1), df_raw.SalePrice)
add_datepart(df_raw, 'saledate')

df_raw.saleYear.head()
train_cats(df_raw)
df_raw.UsageBand.cat.categories
df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
df_raw.UsageBand = df_raw.UsageBand.cat.codes
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
os.makedirs('tmp', exist_ok=True)

df_raw.to_feather('tmp/bulldozers-raw')
df_raw = feather.read_dataframe('tmp/bulldozers-raw')
df, y, nas = proc_df(df_raw, 'SalePrice')
df.head()
m = RandomForestRegressor(n_jobs=-1)

m.fit(df, y)

m.score(df,y)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()



#Set number of rows for validation set

n_valid = 12000  # same as Kaggle's test set size

#Remaining number of rows is used for training set

n_trn = len(df)-n_valid



#Split unprocessed dataframe

raw_train, raw_valid = split_vals(df_raw, n_trn)

#Split processed dataframe and series: see proc_df() function

X_train, X_valid = split_vals(df, n_trn)

y_train, y_valid = split_vals(y, n_trn)



X_train.shape, y_train.shape, X_valid.shape
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice', subset=30000, na_dict=nas)

X_train, _ = split_vals(df_trn, 20000)

y_train, _ = split_vals(y_trn, 20000)
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
# n_estimators=1: one tree

# bootstrap=False: do not use bootstrap samples

# " random forest randomizes bunch of things, we want to turn that off by this parameter"

m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
draw_tree(m.estimators_[0], df_trn, precision=3)
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
# default number of trees is 10

m = RandomForestRegressor(n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
# Get the predictions of each tree using the validation set

# estimators_: List of trees

preds = np.stack([t.predict(X_valid) for t in m.estimators_])



# preds[:,0]: Predicted saleprice of each tree for first data in validation set

# np.mean(preds[:,0]): Mean predicted saleprice (of the 10 trees)

# y_valid[0]: Actual saleprice

preds[:,0], np.mean(preds[:,0]), y_valid[0]
preds.shape
# Plot r squared against the number of trees used (1 to 10)

# Compare predictions against the validation set

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
# oob_score=True: "use out-of-bag samples to estimate the R^2 on unseen data"

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
def dectree_max_depth(tree):

    children_left = tree.children_left

    children_right = tree.children_right



    def walk(node_id):

        if (children_left[node_id] != children_right[node_id]):

            left_max = 1 + walk(children_left[node_id])

            right_max = 1 + walk(children_right[node_id])

            return max(left_max, right_max)

        else: # leaf

            return 1



    root_node_id = 0

    return walk(root_node_id)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
t=m.estimators_[0].tree_
dectree_max_depth(t)
# min_samples_leaf=5: "The minimum number of samples required to be at a leaf node."

# "The numbers that work well are 1, 3, 5, 10, 25, but it is relative to your overall dataset size."

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
t=m.estimators_[0].tree_
dectree_max_depth(t)
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
# max_features=0.5: "The number of features to consider when looking for the best split".

# Consider half of all available features at each split

# "The idea is that the less correlated your trees are with each other, the better"

# "if every tree always splits on the same thing the first time, you will not get much variation in those trees"

# "Good values to use are 1, 0.5, log2, or sqrt"

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
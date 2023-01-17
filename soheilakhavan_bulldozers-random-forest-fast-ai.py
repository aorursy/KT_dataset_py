%load_ext autoreload

%autoreload 2



%matplotlib inline
!pip install git+https://github.com/fastai/fastai@2e1ccb58121dc648751e2109fc0fbf6925aa8887

!apt update && apt install -y libsm6 libxext6
from fastai.imports import *

from fastai.structured import *



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display



from sklearn import metrics
# path is direction to bulldozers data
PATH = '../input/'

!ls {PATH}
# reading dataset 

# dtype of saledate is date thts the reason we use parse_dates

# train

df_raw = pd.read_csv(f'{PATH}train/Train.csv', low_memory=False, parse_dates=["saledate"])

# test

df_test = pd.read_csv(f'{PATH}/Test.csv', low_memory=False, parse_dates=['saledate'])
def display_all(df):

    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 

        display(df)
#display_all(df_raw.tail().T)
#display_all(df_raw.describe(include='all').T)
# kaggle use RMSLE (root mean squared log error)
df_raw.SalePrice = np.log(df_raw.SalePrice)
# add_datepart to add all data out of column date that we define from parsdate in begining
# train

add_datepart(df_raw, 'saledate')

# test

add_datepart(df_test, 'saledate')
# train_cats to convert strings to pandas categories
train_cats(df_raw)
# test

apply_cats(df_test,df_raw)
# We can specify the order to use for categorical variables 
# train

df_raw.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
# Test

df_test.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
# replace Usageband col by usageband codes

# Train

df_raw.UsageBand = df_raw.UsageBand.cat.codes
# Train

df_test.UsageBand = df_test.UsageBand.cat.codes
# dispalying missing values,
# display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
# We'll replace categories with their numeric codes, handle missing continuous values, 

# and split the dependent variable into a separate variable.
# Train

X, y, nas = proc_df(df_raw, 'SalePrice')
# Test

X_test, _, nas = proc_df(df_test, na_dict=nas)
X, y , nas = proc_df(df_raw, 'SalePrice', na_dict=nas)
m = RandomForestRegressor(n_jobs=-1)

m.fit(X, y)

m.score(X,y)
# R^2 ia .983
# Split the dataset into Training and Validation dataset
def split_vals(a,n): return a[:n].copy(), a[n:].copy()



n_valid = 12000  # same as Kaggle's test set size

n_trn = len(X)-n_valid

raw_train, raw_valid = split_vals(df_raw, n_trn)

X_train, X_valid = split_vals(X, n_trn)

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
#An r^2 in the high-80's isn't bad at all (and the RMSLE puts us around rank 100 of 470 

#on the Kaggle leaderboard), but we can see from the validation set score that

#we're over-fitting badly. To understand this issue, let's simplify things down to a single small tree.
## Single tree
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
draw_tree(m.estimators_[0], X_train, precision=3)
m = RandomForestRegressor(n_estimators=1, bootstrap=False, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
#bagging when n_estimators is greater than 1
m = RandomForestRegressor(n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
preds = np.stack([t.predict(X_valid) for t in m.estimators_])

preds[:,0], np.mean(preds[:,0]), y_valid[0]
preds.shape
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(10)]);
#The shape of this curve suggests

#that adding more trees (more than 8) isn't going to help us much. 

#Let's check. (Compare this to our original model on a sample)
m = RandomForestRegressor(n_estimators=20, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=40, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=80, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
### Out-of-bag (OOB) score
m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
## Reducing over-fitting

"""

df_trn, y_trn, nas = proc_df(df_raw, 'SalePrice')

X_train, X_valid = split_vals(df_trn, n_trn)

y_train, y_valid = split_vals(y_trn, n_trn)

"""
"""

set_rf_samples(20000)

"""
"""

m = RandomForestRegressor(n_jobs=-1, oob_score=True)

%time m.fit(X_train, y_train)

print_score(m)

"""
"""

m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)

"""
# another way to reduce overftiing by adding max_features
"""

reset_rf_samples()

"""
"""

m = RandomForestRegressor(n_estimators=40, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)

"""
"""

m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)

"""
m = RandomForestRegressor(n_estimators=40, min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)

m.fit(X_train, y_train)

print_score(m)
# Submiting to kaggle
# Get predictions on processed test dataset.

predictions = m.predict(X_test)
submission = pd.DataFrame({'SalesID': df_test.SalesID, 'SalePrice': predictions})

submission.to_csv('submission.csv', index=False)
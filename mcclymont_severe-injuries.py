%load_ext autoreload

%autoreload|

%matplotlib inline
#You must install a verpy specific version of fastai

!pip install fastai==0.7.0
from fastai.imports import *

from fastai.structured import *



from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

import csv

from sklearn import metrics

import pickle
# Set path to data

path = '../input'

!ls {path}
# Create a pandas dataframe from the csv

df_raw = pd.read_csv(f'{path}/severeinjury.csv', encoding = "ISO-8859-1", low_memory=False)
# here we have the first 5 rows of the df_raw

df_raw.head()
len(df_raw)
#this will list all the columns for df_raw

df_raw.columns
# Here we can see the n of unique values for each column 

for column in df_raw.columns:

    print(column, ': ', (len(df_raw[column].unique())))
# trains all the columns that are categorical in nature

train_cats(df_raw)
# split df in the features and the label(y)

train_df, y, nas = proc_df(df_raw, 'Hospitalized')
# sets hospitalized to a boolean expression 

for index, n in enumerate(y):

    if n > 0:

        y[index] = 1.
counts = np.unique(y, return_counts=True)

counts
def split_vals(a, n):

    return a[:n].copy(), a[n:].copy()
# Here we set the lengths of train and valid - here we are using 0.3 as the split, so 70% train and 30% valid

n_valid = round(len(df_raw) * 0.3)

n_trn = len(df_raw) - n_valid

n_valid, n_trn
# Here we use the above values to split the raw dataframe, the training dataframe and the validation dataframe

raw_train, raw_valid = split_vals(df_raw, n_trn)

# This splits the features

X_train, X_valid = split_vals(train_df, n_trn)

# This splits the labels

y_train, y_valid = split_vals(y, n_trn)
# function that calculate root mean squared error

def rmse(x,y):     

    return math.sqrt(((x-y)**2).mean())
def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    print(res)
# the output shows a 90 ish % accuracy - note it only took 480ms to run the model

m = RandomForestClassifier(n_jobs=-1, n_estimators=10)

%time m.fit(X_train, y_train)

print_score(m)
# this shows the change in accuracy over the 10 estimators - it looks from the graph that there may still be some accuracy to gain

preds = np.stack([t.predict(X_valid) for t in m.estimators_])

plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0))

          for i in range(10)])
m = RandomForestClassifier(n_jobs=-1, n_estimators=15)

%time m.fit(X_train, y_train)

print_score(m)
preds = np.stack([t.predict(X_valid) for t in m.estimators_])

plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0))

          for i in range(15)])
# here we can determine those columns that had the most imopact on the final prediction - the higher the number the higher the relationship

fi = rf_feat_importance(m, train_df)

fi
def plot_fi(fi): 

    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi)
# all the the unique variables 

for n in df_raw['NatureTitle'].unique():

    print(n)
len(df_raw['NatureTitle'].unique())
x = raw_valid.copy()
raw_valid
x.Hospitalized = y_valid
# Here we are checking the number of each diagnosis in the raw data set 

x['pred_std'] = np.std(preds, axis=0)

x['pred'] = np.mean(preds, axis=0)

plotter = x.NatureTitle.value_counts().plot.barh(figsize=(20,40))

np.unique(x.Hospitalized, return_counts=True)
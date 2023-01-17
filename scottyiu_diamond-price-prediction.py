%load_ext autoreload

%autoreload 2





%matplotlib inline
import numpy as np

import pandas as pd



from fastai.imports import *

#from fastai.structured import *

#from structured import *

from fastai.tabular import *

from fastai import *

#from fastai_structured import *





from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from IPython.display import display



from sklearn import metrics

from sklearn.model_selection import train_test_split, ParameterGrid, GridSearchCV



import seaborn as sns

import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib import cm



from scipy.cluster import hierarchy as hc



from fastai_structured import *
!ls ../input/
PATH = "../input/"

!ls {PATH}
## Training dataset

df_raw = pd.read_csv(f'{PATH}diamonds.csv', low_memory=False,

                    index_col = 'Unnamed: 0')
df_raw.head()
df_raw.describe()
plt.hist(list(df_raw.carat), bins=10, edgecolor='white')

plt.ylabel("Occurance")

plt.xlabel("Carats")

plt.title("Occurance of varying carats")
new_index = ['Ideal', 'Premium', 'Very Good', 'Good', 'Fair']

plt.bar(df_raw.groupby('cut').count().iloc[:,0].reindex(new_index).index,

       df_raw.groupby('cut').count().iloc[:,0].reindex(new_index)/len(df_raw.cut))

plt.ylabel("Occurance")

plt.xlabel("Cuts")

plt.title("Occurance of varying cuts")
df_raw.color.unique()
new_index = ['E', 'F', 'G', 'H', 'I', 'J']

plt.bar(df_raw.groupby('color').count().iloc[:,0].reindex(new_index).index,

       df_raw.groupby('color').count().iloc[:,0].reindex(new_index)/len(df_raw.cut))

plt.ylabel("Occurance")

plt.xlabel("Color")

plt.title("Occurance of varying color")
plt.hist(list(df_raw.depth), bins=100, edgecolor='white')

plt.ylabel("Occurance")

plt.xlabel("Depth")

plt.xlim(55,70)

plt.title("Occurance of varying depth")
plt.hist(list(df_raw.table), bins=30, edgecolor='white')

plt.ylabel("Occurance")

plt.xlabel("Table")

plt.xlim(50,70)

plt.title("Occurance of varying table")
plt.hist(list(df_raw.x), bins=30, edgecolor='white')

plt.ylabel("Occurance")

plt.xlabel("X")

plt.xlim(3,10)

plt.title("Occurance of varying x")
plt.hist(list(df_raw.y), bins=100, edgecolor='white')

plt.ylabel("Occurance")

plt.xlabel("Y")

plt.xlim(3,10)

plt.title("Occurance of varying y")
plt.hist(list(df_raw.z), bins=100, edgecolor='white')

plt.ylabel("Occurance")

plt.xlabel("Z")

plt.xlim(2,6)

plt.title("Occurance of varying z")
plt.hist(list(df_raw.price), bins=100, edgecolor='white')

plt.ylabel("Occurance")

plt.xlabel("Price")

plt.title("Occurance of varying price")
df_raw['price'] = np.log(df_raw['price']); df_raw['price'].head()
plt.hist(list(df_raw.price), bins=100, edgecolor='white')

plt.ylabel("Occurance")

plt.xlabel("Price")

plt.title("Occurance of varying price")
train_cats(df_raw)

#apply_cats(df_test_raw,df_raw)
(df_raw.isnull().sum().sort_values(ascending=False)/len(df_raw)).head(10)
df, y, nas = proc_df(df_raw, 'price')
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.25, random_state=42)
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),

                m.score(X_train, y_train), m.score(X_valid, y_valid)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
import IPython

import graphviz



def mydraw_tree(t, df, size=10, ratio=0.6, precision=0):

    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,

                      special_characters=True, rotate=True, precision=precision)

    IPython.display.display(graphviz.Source(re.sub('Tree {',

       f'Tree {{ size={size}; ratio={ratio}', s)))
## Draw tree is fastai

mydraw_tree(m.estimators_[0], X_train, precision=3)
df_nosize = df.copy()
df_nosize.columns
df_nosize = df_nosize.drop(['carat','depth','table','x','y','z'],axis=1)
df_nosize.head()
X_train,X_valid,y_train,y_valid = train_test_split(df_nosize,y,test_size=0.25, random_state=42)
m = RandomForestRegressor(n_jobs=-1)

%time m.fit(X_train, y_train)

print_score(m)
X_train
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)

m.fit(X_train, y_train)

print_score(m)
import IPython

import graphviz



def mydraw_tree(t, df, size=10, ratio=0.6, precision=0):

    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,

                      special_characters=True, rotate=True, precision=precision)

    IPython.display.display(graphviz.Source(re.sub('Tree {',

       f'Tree {{ size={size}; ratio={ratio}', s)))
## Draw tree is fastai

mydraw_tree(m.estimators_[0], X_train, precision=3)
X_train,X_valid,y_train,y_valid = train_test_split(df,y,test_size=0.25, random_state=42)
m = RandomForestRegressor(n_estimators=100,n_jobs=-1,oob_score=True)

m.fit(X_train, y_train)

print_score(m)
## Each tree is found in m.estimators_



preds = np.stack([t.predict(X_valid) for t in m.estimators_])

preds[:,0], np.mean(preds[:,0]), y_valid[0]
plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis=0)) for i in range(100)]);
param_grid = {

    'min_samples_leaf': [1, 5, 10],

    'max_features': [0.5, 1],

    'n_estimators': [20],

    'n_jobs': [-1],

    'random_state': [42]

}



m = RandomForestRegressor(n_estimators=20)



grid_search = GridSearchCV(m, param_grid=param_grid, cv=5, iid=False,

                           verbose=1, scoring='neg_mean_squared_error');

grid_search.fit(X_train, y_train);

#print(grid_search.cv_results_)
print(grid_search.best_score_)
print(grid_search.best_params_)
myscoredf = pd.DataFrame(grid_search.cv_results_)[['param_min_samples_leaf','param_max_features','mean_test_score']]; myscoredf.head(10)
myscoredf = myscoredf.pivot('param_min_samples_leaf','param_max_features','mean_test_score')
ax = sns.heatmap(myscoredf, annot=True, fmt=".5g", cmap=cm.coolwarm)
m = RandomForestRegressor(n_estimators=20,

                          min_samples_leaf=grid_search.best_params_['min_samples_leaf'],

                          max_features=grid_search.best_params_['max_features'],

                          n_jobs=-1)

m.fit(X_train, y_train);

print_score(m)
fi = rf_feat_importance(m, df); fi[:10]
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)

plot_fi(fi[:30]);
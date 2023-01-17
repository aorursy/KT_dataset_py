!pip install fastai==0.7.0
import numpy as np 
from pandas import *
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn import metrics
import math
from matplotlib import pyplot as plt, rcParams, animation
from sklearn.tree import export_graphviz
import IPython, graphviz, re
from pdpbox import pdp
from fastai.imports import *
from fastai.structured import *

from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)
    
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
        
def split_vals(a,n): return a[:n].copy(), a[n:].copy()

def plot_pdp(feat, model_features, clusters=None, feat_name=None):
    feat_name = feat_name or feat
    p = pdp.pdp_isolate(m, x, model_features, feat)
    return pdp.pdp_plot(p, feat_name, plot_lines=True, 
                        cluster=clusters is not None, 
                        n_cluster_centers=clusters)
def get_sample(df, n):
    idxs = sorted(np.random.permutation(len(df))[:n])
    return df.iloc[idxs].copy()
# the dataframe along the columns
PATH = "../input/"
df_raw = pd.read_csv(f'{PATH}train.csv', low_memory=False)
display_all(df_raw.tail().T)
# descibing the dataframe along the columns
display_all(df_raw.describe(include='all').T)
# taking the log of the dependent variable
df_raw.SalePrice = np.log(df_raw.SalePrice)
# convert the string datatype to categories
train_cats(df_raw)
# make the categories values ordered so that the tree has not to split much often.
# I found out the columns which are not having category values in order
df_raw.LotShape.cat.set_categories(['Reg','IR1', 'IR2', 'IR3'], ordered=True, inplace=True)
df_raw.LandContour.cat.set_categories(['Lvl', 'Bnk', 'HLS', 'Low'], ordered=True, inplace=True)
df_raw.LotConfig.cat.set_categories(['Inside', 'Corner', 'CulDSac', 'FR2', 'FR3'], ordered=True, inplace=True)
df_raw.ExterQual.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa'], ordered=True, inplace=True)
df_raw.BsmtQual.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa'], ordered=True, inplace=True)
df_raw.BsmtCond.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa'], ordered=True, inplace=True)
df_raw.BsmtExposure.cat.set_categories(['Gd', 'Av', 'Mn', 'No'], ordered=True, inplace=True)
df_raw.BsmtFinType1.cat.set_categories(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf', 'NA'], ordered=True, inplace=True)
df_raw.BsmtFinType2.cat.set_categories(['GLQ', 'ALQ', 'BLQ', 'Rec', 'LwQ', 'Unf'], ordered=True, inplace=True)
df_raw.HeatingQC.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po'], ordered=True, inplace=True)
df_raw.Electrical.cat.set_categories(['SBrkr', 'FuseA', 'FuseF', 'FuseP', 'Mix'], ordered=True, inplace=True)
df_raw.KitchenQual.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po'], ordered=True, inplace=True)
df_raw.Functional.cat.set_categories(['Typ', 'Min1', 'Min2', 'Mod', 'Maj1', 'Maj2', 'Sev', 'Sal'], ordered=True, inplace=True)
df_raw.FireplaceQu.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po'], ordered=True, inplace=True)
df_raw.GarageQual.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po'], ordered=True, inplace=True)
df_raw.GarageCond.cat.set_categories(['Ex', 'Gd', 'TA', 'Fa', 'Po'], ordered=True, inplace=True)
df_raw.PoolQC.cat.set_categories(['Ex', 'Gd', 'Fa'], ordered=True, inplace=True)
df_raw.Fence.cat.set_categories(['GdPrv', 'MnPrv', 'GdWo', 'MnWw'], ordered=True, inplace=True)
df_raw.SaleType.cat.set_categories(['WD', 'CWD', 'New', 'COD', 'Con', 'ConLw', 'ConLI', 'ConLD', 'Oth'], ordered=True, inplace=True)
df_raw.SaleCondition.cat.set_categories(['Normal', 'Abnorml', 'AdjLand', 'Alloca', 'Family', 'Partial'], ordered=True, inplace=True)
# check which column has null values
display_all(df_raw.isnull().sum().sort_index()/len(df_raw))
# Sorting the data by YrSold and MoSold, so that we are doing the prediction on the future date data
df_raw = df_raw.sort_values(['YrSold', 'MoSold'])
# processng the dataframe: 
# handling the missing values, creating additional columns with na subscript for the missing values columns
# the missing values in a column has been assigned the median of that column
# splitting the dependent variables and independent variable
df, y, nas = proc_df(df_raw, 'SalePrice')
display_all(df.tail().T)
# columns having the null values
nas
# lets now check which column has null values for df
display_all(df.isnull().sum().sort_index()/len(df_raw))
# Creating a validation and training sets
n_valid = 50
n_trn = len(df)-n_valid
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)
# Lets train a RF using one estimator
m = RandomForestRegressor(n_estimators=1, max_depth=3, bootstrap=False, n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
# Lets draw the tree and observe which variables are getting splitted
draw_tree(m.estimators_[0], df, precision=3)
# Lets train a RF using 30 estimators
m = RandomForestRegressor(n_estimators=30,  min_samples_leaf=3, max_features=0.5, n_jobs=-1, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
# sorting according to feature importance
fi = rf_feat_importance(m, df); fi[:20]
fi.plot('cols', 'imp', figsize=(10,6), legend=False);
fi[:20].plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
# Getting features having importance greater than 0.01
to_keep = fi[fi.imp>0.01].cols; len(to_keep)
df_keep = df[to_keep].copy()
# finding correlation between the features
import scipy
from scipy.cluster import hierarchy as hc
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()

# As shown in the graph, Garage cars and Garage Area are somewhat correlated
# Lets plot the interaction plot between Garage Cars and Garage Area
x = get_sample(X_train, 500) # take 500 samples from X_train
feats = ['GarageCars', 'GarageArea']
p = pdp.pdp_interact(m, x, X_train.columns, feats)
pdp.pdp_interact_plot(p, feats)
# partial dependence plot between Sale Price and OverallQuality
plot_pdp('OverallQual', model_features=x.columns)
plot_pdp('GrLivArea', model_features=x.columns)

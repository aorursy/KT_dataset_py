%load_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np 

import pandas as pd

from IPython.display import display

from pandas_summary import DataFrameSummary

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.model_selection import train_test_split

import os

from pandas_summary import DataFrameSummary

from matplotlib import pyplot as plt

import math



from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier, export_graphviz

import graphviz

import re



import shap

import eli5

from eli5.sklearn import PermutationImportance

from pdpbox import pdp, get_dataset, info_plots



import IPython

from IPython.display import display

print(os.listdir("../input/"))
train_df = pd.read_csv("../input/reducing-memory-size-for-ieee/train.csv")

test_df = pd.read_csv("../input/reducing-memory-size-for-ieee/test.csv")

train_df.head()
test_df.head()
import fastai_structured   ## Adding structured.py from fastai v0.7 as a utility script to the kernel

fastai_structured.train_cats(train_df)

fastai_structured.apply_cats(test_df, train_df)
nas = {}

df_trn, y_trn, nas = fastai_structured.proc_df(train_df, 'isFraud', na_dict=nas)   ## Avoid creating NA columns as total cols may not match later

df_test, _, _ = fastai_structured.proc_df(test_df, na_dict=nas)

df_trn.head()
df_test.head()
del train_df, test_df
from imblearn.under_sampling import RandomUnderSampler



ran=RandomUnderSampler(return_indices=True) ##intialize to return indices of dropped rows

df_trn_sm,y_trn_sm,dropped = ran.fit_sample(df_trn,y_trn)



#print("The number of removed indices are ",len(dropped))

#plot_2d_space(X_rs,y_rs,X,y,'Random under sampling')
train_X, val_X, train_y, val_y = train_test_split(df_trn_sm, y_trn_sm, test_size=0.33, random_state=42)
from sklearn.metrics import roc_auc_score



def print_score(m):

    res = [roc_auc_score(m.predict(train_X), train_y), roc_auc_score(m.predict(val_X), val_y)]

    print(res)
##To reduce CPU load, and for faster iteration

fastai_structured.set_rf_samples(200000)

del df_trn, y_trn, df_trn_sm, y_trn_sm
%time

m = RandomForestClassifier(n_estimators=1, min_samples_leaf=5, max_depth = 3) ## Use all CPUs available

m.fit(train_X, train_y)



print_score(m)
def draw_tree(t, df, size=10, ratio=0.6, precision=0):

    """ Draws a representation of a random forest in IPython.

    Parameters:

    -----------

    t: The tree you wish to draw

    df: The data used to train the tree. This is used to get the names of the features.

    """

    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True,

                      special_characters=True, rotate=True, precision=precision)

    IPython.display.display(graphviz.Source(re.sub('Tree {',

       f'Tree {{ size={size}; ratio={ratio}', s)))
#draw_tree(m.estimators_[0], train_X, precision=3)
%time

m = RandomForestClassifier(n_estimators=30, min_samples_leaf=20, max_features=0.7, 

                                n_jobs=-1, oob_score=True) ## Use all CPUs available

m.fit(train_X, train_y)



print_score(m)
## pred = m.predict(df_test)          ## Gets an AUC of ~0.8

pred = m.predict_proba(df_test)[:,1]  ## Gets an AUC of ~0.9

submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv')

submission.head()
submission['isFraud'] = pred   

submission.to_csv('rf_submission_vf.csv', index=False)
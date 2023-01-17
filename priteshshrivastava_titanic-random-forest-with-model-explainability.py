#!pip install --upgrade pip

!pip install fastai==0.7.0 ## Based on Fast.ai ML course



%load_ext autoreload

%autoreload 2

%matplotlib inline
import numpy as np 

import pandas as pd

from IPython.display import display

from fastai.imports import *

from fastai.structured import *

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
train_df = pd.read_csv("../input/train.csv")

test_df = pd.read_csv("../input/test.csv")

train_df.head()
test_df.head()
train_cats(train_df)

apply_cats(test_df, train_df)
df_trn, y_trn, nas = proc_df(train_df, 'Survived')

df_test, _, _ = proc_df(test_df, na_dict=nas)

df_trn.head()
df_test.head()
## Let's remove the NA columns that were introduced by proc_df as the test and train datasets have different no of columns

df_trn.drop(['Age_na'], axis =1, inplace = True)

df_test.drop(['Age_na', 'Fare_na'], axis =1, inplace = True)

df_test.head()
def rmse(x,y): return math.sqrt(((x-y)**2).mean())



def print_score(m):

    res = [rmse(m.predict(train_X), train_y), rmse(m.predict(val_X), val_y),     ## RMSE of log of prices

                m.score(train_X, train_y), m.score(val_X, val_y)]

    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)

    print(res)
train_X, val_X, train_y, val_y = train_test_split(df_trn, y_trn, test_size=0.33, random_state=42)
%time

m = RandomForestClassifier(n_estimators=1, min_samples_leaf=10, n_jobs=-1, max_depth = 3, oob_score=True) ## Use all CPUs available

m.fit(train_X, train_y)



print_score(m)
draw_tree(m.estimators_[0], train_X, precision=3)
%time

m = RandomForestClassifier(n_estimators=20, min_samples_leaf=10, max_features=0.7, n_jobs=-1, oob_score=True) ## Use all CPUs available

m.fit(train_X, train_y)



print_score(m)
perm = PermutationImportance(m, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
for feat_name in val_X.columns:

#for feat_name in base_features:

    #pdp_dist = pdp.pdp_isolate(model=m, dataset=val_X, model_features=base_features, feature=feat_name)

    pdp_dist = pdp.pdp_isolate(model = m, dataset=val_X, model_features=val_X.columns, feature=feat_name)



    pdp.pdp_plot(pdp_dist, feat_name)



    plt.show()
explainer = shap.TreeExplainer(m)

shap_values = explainer.shap_values(val_X)



# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

# shap.force_plot(explainer.expected_value, shap_values[1,:], val_X.iloc[1,:], matplotlib=True) ## Not for classification
shap.summary_plot(shap_values, val_X, plot_type="bar")
%time

df_trn.drop(['Embarked', 'Fare', 'Cabin', 'Parch'], axis =1, inplace = True)

df_test.drop(['Embarked', 'Fare', 'Cabin', 'Parch'], axis =1, inplace = True)

train_X, val_X, train_y, val_y = train_test_split(df_trn, y_trn, test_size=0.33, random_state=42)

m = RandomForestClassifier(n_estimators=20, min_samples_leaf=10, max_features=0.7, n_jobs=-1, oob_score=True) ## Use all CPUs available

m.fit(train_X, train_y)

print_score(m)
pred = m.predict(df_test)

submission = pd.read_csv('../input/gender_submission.csv')

submission.head()
submission['Survived'] = pred   

submission.to_csv('rf_submission_v2.csv', index=False)
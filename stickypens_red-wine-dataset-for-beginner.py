#Importing required packages.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
%matplotlib inline
import os
wine_raw = pd.read_csv("../input/winequality-red.csv", low_memory=False)
wine_raw.describe()
#!pip install fastai==0.7.0
from fastai.imports import *
#from fastai.structured import *
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

from sklearn import metrics
bins = [0, 4, 6, 10]
labels = ["poor","normal","excellent"]
wine_raw['quality_cat'] = pd.cut(wine_raw['quality'], bins=bins, labels=labels)
wine_raw = wine_raw.drop('quality', axis = 1)
wine_raw.tail(50)
y = wine_raw['quality_cat']
df = wine_raw.drop('quality_cat', axis=1)
df.sample(7)
#df, y, nas = proc_df(wine_raw, 'quality_cat')
wine_raw = wine_raw.sample(frac=1, axis=0).reset_index(drop=True)
def split_vals(a,n): return a[:n].copy(), a[n:].copy()
n_valid = 399
n_trn = len(df)-n_valid
raw_train, raw_valid = split_vals(wine_raw, n_trn)
X_train, X_valid = split_vals(df, n_trn)
y_train, y_valid = split_vals(y, n_trn)

X_train.shape, y_train.shape, X_valid.shape
m = RandomForestClassifier(n_jobs=-1)
%time m.fit(X_train, y_train)
import math
#The aim is to reduce the rmse error and increase the score
def rmse(x,y): return math.sqrt((np.subtract(x-y)**2).mean())

def print_score(m):
  res = [#rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
        m.score(X_train, y_train), m.score(X_valid, y_valid)]
  if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
  print(res)
print_score(m)
preds = np.stack([t.predict(X_valid) for t in m.estimators_])
preds[:,0], np.mean(preds[:,0]), y_valid[0]
preds
y_plot = y_valid.cat.codes.tolist()
y_plot = np.array(y_plot)
y_plot.astype(np.float)
preds1 = preds.astype(float)
metrics.accuracy_score(y_valid, m.predict(X_valid))
m = RandomForestClassifier(n_jobs=-1, n_estimators=20)
m.fit(X_train, y_train)
print_score(m)
#preds = np.stack([t.predict(X_valid) for t in m.estimators_])
#preds[:,0], np.mean(preds[:,0]), y_valid[0]
#plt.plot([metrics.r2_score(y_valid, np.mean(preds[:i+1], axis = 0)) for i in range(20)])
from sklearn.ensemble import ExtraTreesClassifier
e = ExtraTreesClassifier(n_jobs=1)
e.fit(X_train, y_train)
print_score(e)
m = RandomForestClassifier(n_jobs=-1, n_estimators=10, oob_score=True)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs=-1, n_estimators=40, oob_score=True, max_features=0.6)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs=-1, n_estimators=40, oob_score=True, max_features=0.7)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs=-1, n_estimators=40, oob_score=True, max_features=0.5)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs=-1, n_estimators=40, oob_score=True, max_features=0.4)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs=-1, n_estimators=40, oob_score=True, max_features=0.3)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs=-1, n_estimators=40, oob_score=True, max_features=0.3, max_depth=3)
m.fit(X_train, y_train)
print_score(m)
#draw_tree(m.estimators_[0], df, precision=3)
#set_rf_samples(300)
m = RandomForestClassifier(n_jobs=-1, min_samples_leaf=3, max_features=0.5,  n_estimators =40, oob_score = True)
%time m.fit(X_train, y_train)
print_score(m)
#reset_rf_samples()
m = RandomForestClassifier(n_jobs=-1, n_estimators=40, oob_score=True, max_features=0.4)
m.fit(X_train, y_train)
print_score(m)
raw_train.sample(1)
X_train.columns
m.feature_importances_
fi = pd.DataFrame({'cols':X_train.columns, 'imp':m.feature_importances_}
                       ).sort_values('imp', ascending=False)
#fi = rf_feat_importance(m, X_train)
fi[:15]
fi.plot('cols', 'imp', figsize=(10,6), legend=False )
def plot_fi(fi): return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
plot_fi(fi[:30])
m = RandomForestClassifier(n_jobs=-1, n_estimators=100, oob_score=True, max_features=0.6)
m.fit(X_train, y_train)
print_score(m)
m = RandomForestClassifier(n_jobs=-1, n_estimators=1000, oob_score=True, max_features=0.3)
m.fit(X_train, y_train)
print_score(m)

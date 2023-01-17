# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%load_ext autoreload
%autoreload 2
%matplotlib inline

from fastai.imports import *
from fastai.structured import *

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.cluster import hierarchy as hc

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
PATH = '../input/'

train_x = pd.read_csv(PATH+'training_set_values.csv')
train_y = pd.read_csv(PATH+'training_set_labels.csv')
test = pd.read_csv(PATH+'test_set_values.csv')
train_x.shape,test.shape
# This adds few columns of date 
add_datepart(train_x, 'date_recorded')
train_cats(train_x)
#since the output is in string format we will encode it with number
def func(data):
    if data=='functional':
        return 0
    elif data=='non functional':
        return 1 
    elif data=='functional needs repair':
        return 2
y = train_y.status_group.apply(func)
# adding the taget labels to the training set to perform uniform encoding in later step
train_x['status_group'] = y
df,y,nas = proc_df(train_x,'status_group')
# function to split the data into training and validation/test set
def split_vals(a,n):
    return a[:n].copy(), a[n:].copy()
n_valid = 14850 # same as the size of test set
n_trn = len(df)- n_valid
x_train, x_val = split_vals(df,n_trn)
y_train, y_val = split_vals(y, n_trn)
m = RandomForestClassifier(n_estimators=200,min_samples_leaf=3 ,n_jobs=-1,max_features=0.3)
%time m.fit(x_train, y_train)
y_pred= m.predict(x_val)
accuracy_score(y_val, y_pred)
feature_imp = rf_feat_importance(m, df)
feature_imp.plot('cols', 'imp', figsize = (10, 6), legend = False)
def plot_fi(fi):
    return fi.plot('cols', 'imp', 'barh', figsize=(12,10), legend = False)
plot_fi(feature_imp)
## Before executing this line please execute 
keep = feature_imp[feature_imp.imp > 0.0097].cols
df_keep = df[keep].copy()
corr = np.round(scipy.stats.spearmanr(df_keep).correlation, 4)
corr_condensed = hc.distance.squareform(1-corr)
z = hc.linkage(corr_condensed, method='average')
fig = plt.figure(figsize=(16,10))
dendrogram = hc.dendrogram(z, labels=df_keep.columns, orientation='left', leaf_font_size=16)
plt.show()
df_keep.drop(['quantity_group','date_recordedWeek'], axis = 1, inplace = True)
x_train, x_val = split_vals(df_keep,n_trn)
m = RandomForestClassifier(n_estimators=200,min_samples_leaf=3 ,n_jobs=-1,max_features=0.3)
%time m.fit(x_train, y_train)
y_pred= m.predict(x_val)
accuracy_score(y_val, y_pred)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.ensemble import RandomForestRegressor

from sklearn import metrics 

from IPython.display import display



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_table('../input/train.tsv', engine='c')

test = pd.read_table('../input/test.tsv', engine='c')



# Any results you write to the current directory are saved as output.
train['missing'] = train.brand_name.isnull()
train.brand_name[train['brand_name'].isnull()] = 'None'
train.dtypes
train['name'] = pd.Series(train.name, dtype="category").cat.codes

train['category_name'] = pd.Series(train.category_name, dtype="category").cat.codes

train['brand_name'] = pd.Series(train.brand_name, dtype="category").cat.codes

train['item_description'] = pd.Series(train.item_description, dtype="category").cat.codes

train['missing'] = pd.Series(train.missing, dtype="category").cat.codes
train.dtypes
training = train.sample(frac=0.8,random_state=200)

validation = train.drop(training.index)
xtrain = training.drop('price', axis=1)

ytrain = training.price



xvalid = validation.drop('price', axis=1)

yvalid = validation.price
rf = RandomForestRegressor(n_jobs=-1, n_estimators=10)

rf.fit(xtrain, ytrain)
from sklearn import metrics

rf_score = rf.score(xtrain, ytrain)

rf_score
feature_names = xtrain.columns
feature_imp = pd.DataFrame({'cols':feature_names, 'imp':rf.feature_importances_}).sort_values('imp', ascending=False)

feature_imp
feature_imp.plot('cols', 'imp', figsize=(10,6), legend=False);
xtrain_name = xtrain.copy()
xtrain_name['name'] = np.random.permutation(xtrain_name.name)
rf.score(xtrain_name, ytrain)
xtrain_item_description = xtrain.copy()

xtrain_item_description['item_description'] = np.random.permutation(xtrain_item_description.item_description)

rf.score(xtrain_item_description, ytrain)
xtrain_missing = xtrain.copy()

xtrain_missing['missing'] = np.random.permutation(xtrain_missing.missing)

rf.score(xtrain_missing, ytrain)
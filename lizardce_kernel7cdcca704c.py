# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
df.shape , test.shape
df.head()

df.dtypes
df['area']=[float(a) for a in [i.replace(',','') for i in list(df['area'].values)]]
df['densidade_dem']=[float(i) for i in [i.replace(',','') for i in [str(i) for i in df['densidade_dem']]]]
df.dtypes
test['area']=[float(a) for a in [i.replace(',','') for i in list(test['area'].values)]]
test['densidade_dem']=[float(i) for i in [i.replace(',','') for i in [str(i) for i in test['densidade_dem']]]]
df.columns , test.columns
pd.Series(df.columns).isin(pd.Series(test.columns))
df.columns[25]
import seaborn as sns



sns.heatmap(df.isnull())
sns.heatmap(test.isnull())
test['ranking_igm'].isnull().sum(),sum(test['ranking_igm'].isnull()==False)
df.loc[:,[c for c in (df.isnull().sum()>50)==True]].shape
test.loc[:,[c for c in (test.isnull().sum()>50)==True]].shape
a=list(test.loc[:,[c for c in (test.isnull().sum()>50)==True]].columns)
b=list(df.loc[:,[c for c in (df.isnull().sum()>50)==True]].columns)
d=list(pd.Series(a+b).unique())
feats=[a for a in df.columns if a not in ['nota_mat']+d]
feats
for i in df[feats].columns:

    if df[i].isnull().sum()>0:

        if df[i].dtypes!='object':

            df[i]=df[i].fillna(df[i].mean())

        else:

            continue

    else:

        continue
for i in df.columns:

    if df[i].dtypes=='object':

        df[i].replace(list(df[i].unique()),list(range(0,len(df[i].unique()))),inplace=True)

    else:

        continue
df[feats].isnull().sum()
for i in test[feats].columns:

    if test[i].isnull().sum()>0:

        if test[i].dtypes!='object':

            test[i]=test[i].fillna(test[i].mean())

        else:

            continue

    else:

        continue
for i in test.columns:

    if test[i].dtypes=='object':

        test[i].replace(list(test[i].unique()),list(range(0,len(test[i].unique()))),inplace=True)

    else:

        continue
df.dtypes
df.loc[:,feats].isnull().sum()
for i in range(0,len(df.columns)):

    df[df.columns[i]] = np.log(df[df.columns[i]] + 1)
for i in range(0,len(test.columns)):

    test[test.columns[i]] = np.log(test[test.columns[i]] + 1)





from sklearn.model_selection import train_test_split
train, valid = train_test_split(df, random_state=42)
train.shape, valid.shape
'''from sklearn.model_selection import RandomizedSearchCV

from pprint import pprint

# Number of trees in random forest

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Number of features to consider at every split

max_features = ['auto', 'sqrt']

# Maximum number of levels in tree

max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]

max_depth.append(None)

# Minimum number of samples required to split a node

min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node

min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree

bootstrap = [True, False]

# Create the random grid

random_grid = {'n_estimators': n_estimators,

               'max_features': max_features,

               'max_depth': max_depth,

               'min_samples_split': min_samples_split,

               'min_samples_leaf': min_samples_leaf,

               'bootstrap': bootstrap}

pprint(random_grid)

{'bootstrap': [True, False],

 'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],

 'max_features': ['auto', 'sqrt'],

 'min_samples_leaf': [1, 2, 4],

 'min_samples_split': [2, 5, 10],

 'n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}'''
'''# Use the random grid to search for best hyperparameters

# First create the base model to tune

rf = RandomForestRegressor()

# Random search of parameters, using 3 fold cross validation, 

# search across 100 different combinations, and use all available cores

rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model

rf_random.fit(train[feats], train['nota_mat'])'''
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestRegressor

# Create the parameter grid based on the results of random search 

param_grid = {

    'bootstrap': [True],

    'max_depth': [80, 90, 100, 110],

    'max_features': [2, 3],

    'min_samples_leaf': [3, 4, 5],

    'min_samples_split': [8, 10, 12],

    'n_estimators': [100, 200, 300, 1000]

}

# Create a based model

rf = RandomForestRegressor()

# Instantiate the grid search model

grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 

                          cv = 3, n_jobs = -1, verbose = 2)
# Fit the grid search to the data

grid_search.fit(train[feats], train['nota_mat'])
grid_search.best_params_
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(random_state=42, n_jobs=-1, min_samples_leaf=3,max_features=3,min_samples_split=8,bootstrap=True,max_depth=90, 

                           n_estimators=200)
rf.fit(train[feats], train['nota_mat'])
from sklearn.metrics import mean_squared_error
mean_squared_error(rf.predict(valid[feats]), valid['nota_mat'])**(1/2)
pd.Series(rf.feature_importances_, index=feats).sort_values().plot.barh()
#feats=[f for f in feats if f not in pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False).tail(5).index]
#feats.remove('exp_vida')
final=pd.DataFrame([np.exp(test.codigo_mun).values,np.exp(rf.predict(test[feats]))]).T
final.columns=['codigo_mun','nota_mat']
test2=pd.read_csv('../input/test.csv')
test2.codigo_mun=[i.replace("ID_ID_","") for i in test2.codigo_mun]
final.codigo_mun=test2.codigo_mun.unique()
final.codigo_mun=[int(i) for i in final.codigo_mun]
final
final.to_csv('Breno.csv',index=False)
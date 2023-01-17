# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
structures = pd.read_csv('../input/structures.csv')

train = pd.read_csv('../input/train.csv')

test= pd.read_csv('../input/test.csv')
train_copy = train.copy()
train_copy['atom_index_0']=train['atom_index_1']

train_copy['atom_index_1']=train['atom_index_0']
def map_atom_info(df, atom_idx):

    df = pd.merge(df, structures, how = 'left',

                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],

                  right_on = ['molecule_name',  'atom_index'])

    

    df = df.drop('atom_index', axis=1)

    df = df.rename(columns={'atom': f'atom_{atom_idx}',

                            'x': f'x_{atom_idx}',

                            'y': f'y_{atom_idx}',

                            'z': f'z_{atom_idx}'})

    return df



train = map_atom_info(train, 0)

train = map_atom_info(train, 1)



train_copy = map_atom_info(train_copy, 0)

train_copy = map_atom_info(train_copy, 1)



test = map_atom_info(test, 0)

test = map_atom_info(test, 1)
train_p_0 = train[['x_0', 'y_0', 'z_0']].values

train_p_1 = train[['x_1', 'y_1', 'z_1']].values

train_copy_p_0 = train_copy[['x_0', 'y_0', 'z_0']].values

train_copy_p_1 = train_copy[['x_1', 'y_1', 'z_1']].values

test_p_0 = test[['x_0', 'y_0', 'z_0']].values

test_p_1 = test[['x_1', 'y_1', 'z_1']].values



train['dist'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)

train_copy['dist']= np.linalg.norm(train_copy_p_0 - train_copy_p_1, axis=1)

test['dist'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)
GroupedBy = structures.groupby(by='molecule_name')



molecule_count = GroupedBy.count().drop(['atom_index','x','y','z'],axis=1)

molecule_count = molecule_count.rename(columns={'atom':'molecule_size'})



molecule_mean = GroupedBy.mean().drop(['atom_index'],axis=1)

molecule_mean = molecule_mean.rename(columns={'x':'x_mean','y':'y_mean','z':'z_mean'})



molecule_max = GroupedBy.max().drop(['atom_index','atom'],axis=1)

molecule_max = molecule_max.rename(columns={'x':'x_max','y':'y_max','z':'z_max'})



molecule_min = GroupedBy.min().drop(['atom_index','atom'],axis=1)

molecule_min = molecule_min.rename(columns={'x':'x_min','y':'y_min','z':'z_min'})



molecule_std = GroupedBy.std().drop(['atom_index'],axis=1)

molecule_std = molecule_std.rename(columns={'x':'x_std','y':'y_std','z':'z_std'})



atom_count=pd.get_dummies(structures.rename(columns={'atom':'atom_count'}),columns=['atom_count']).drop(['atom_index','x','y','z'],axis=1)

atom_count=atom_count.groupby('molecule_name').sum()
def add_features_1(df):

    df2 = df[['molecule_name','type']].rename(columns={'type':'type_count'})

    type_count=pd.get_dummies(df2,columns=['type_count']).groupby('molecule_name').sum()

    

    df = pd.merge(df, molecule_count, how = 'left',left_on  = ['molecule_name'],right_on = ['molecule_name'])

    df = pd.merge(df,molecule_mean,how='left',left_on  = ['molecule_name'],right_on = ['molecule_name'])

    df = pd.merge(df,molecule_min,how='left',left_on  = ['molecule_name'],right_on = ['molecule_name'])

    df = pd.merge(df,molecule_max,how='left',left_on  = ['molecule_name'],right_on = ['molecule_name'])

    df = pd.merge(df,molecule_std,how='left',left_on  = ['molecule_name'],right_on = ['molecule_name'])

    df = pd.merge(df,atom_count,how='left',left_on  = ['molecule_name'],right_on = ['molecule_name'])

    df = pd.merge(df,type_count,how='left',left_on = ['molecule_name'],right_on = ['molecule_name'])

    return(df)
train=add_features_1(train)

train_copy=add_features_1(train_copy)

test=add_features_1(test)
#print(test['atom_0'].value_counts())

#train = train.drop(['atom_0'],axis=1)

#test = test.drop(['atom_0'],axis=1)
#condition = ((train.type=='2JHH') | (train.type=='3JHH'))

#train[condition].head()
def add_features_2(df):

    df['couples_number'] = df.groupby(['molecule_name'])['id'].transform('count')

    df['avg_dist']=df.groupby(['molecule_name'])['dist'].transform('mean')

    df['min_dist']=df.groupby(['molecule_name'])['dist'].transform('min')

    df['max_dist']=df.groupby(['molecule_name'])['dist'].transform('max')

    df['dist_std']=df.groupby(['molecule_name'])['dist'].transform('std')

    

    

    df_p_0 = df[['x_0', 'y_0', 'z_0']].values

    df_p_1 = df[['x_1', 'y_1', 'z_1']].values

    df_p_mean = df[['x_mean','y_mean','z_mean']].values



    df['dist_0_to_mean'] = np.linalg.norm(df_p_0 - df_p_mean, axis=1)

    df['dist_1_to_mean'] = np.linalg.norm(df_p_1 - df_p_mean, axis=1)

    return(df)
train=add_features_2(train)

train_copy=add_features_2(train_copy)

test= add_features_2(test)
train_final = pd.concat([train, train_copy])
train_final = train_final.fillna(0)

train_final.isna().sum()
#sns.distplot(train.scalar_coupling_constant)
#sns.countplot(molecule_count.molecule_size)
#sns.countplot(train.type)
#sns.countplot(train['atom_1'])
#sns.boxplot(x=train.atom_1,y=train.scalar_coupling_constant,palette='rainbow')
#sns.boxplot(x=train.type,y=train.scalar_coupling_constant,palette='rainbow')
#sns.distplot(train.dist)
#plt.scatter(train.dist,train.scalar_coupling_constant)
#plt.scatter(train.molecule_size,train.scalar_coupling_constant)
X = train_final.drop(['molecule_name','scalar_coupling_constant','id'],axis=1)

X= pd.get_dummies(X)



Y= train_final['scalar_coupling_constant']



id_test = test['id']

X_test = test.drop(['molecule_name','id'],axis=1)

X_test = pd.get_dummies(X_test)
# Get missing columns in the training test

missing_cols = set( X.columns ) - set( X_test.columns )

# Add a missing column in test set with default value equal to 0

for c in missing_cols:

    X_test[c] = 0

# Ensure the order of column in the test set is in the same order than in train set

X_test = X_test[X.columns]
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import AdaBoostRegressor

from sklearn.metrics import mean_absolute_error



groups = train['type']



def group_mean_log_mae(y_true, y_pred, groups, floor=1e-9):

    maes = (y_true-y_pred).abs().groupby(groups).mean()

    return np.log(maes.map(lambda x: max(x, floor))).mean()
#model=GradientBoostingRegressor()

#scores = cross_val_score(model,X,Y,cv=5)
#X_train, X_val, Y_train, Y_val = train_test_split(X[:1000000],Y[:1000000],test_size=0.2)
#model=RandomForestRegressor(n_estimators=10,max_features = 0.3,max_depth=20, verbose=1)

#model.fit(X_train,Y_train)

#pred= model.predict(X_val)

#group_mean_log_mae(Y_val,pred,groups)
#importances = model.feature_importances_

#indices = np.argsort(importances)[::-1]



#plt.figure(figsize=(10,5))

#plt.title("Feature importances")

#plt.bar(range(X.shape[1]), importances[indices],color="r", align="center")

#plt.xticks(range(X.shape[1]), X.columns[indices],rotation='vertical')

#plt.xlim([-1, X.shape[1]])

#plt.show()
#importances
model=RandomForestRegressor(n_estimators=10,max_features =0.3,max_depth=15, verbose=1)

#model=AdaBoostRegressor(n_estimators=60,verbose=1)

model.fit(X,Y)

pred=model.predict(X_test)
test_output = pd.DataFrame({"id" : id_test,"scalar_coupling_constant": pred})

test_output.set_index("id", inplace=True)

test_output.to_csv("prediction2.csv")
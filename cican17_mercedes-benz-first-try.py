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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import xgboost as xgb
color = sns.color_palette()

%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')
print("Train shape : ", df_train.shape)
print("Test shape : ", df_test.shape)
df_train.head()
plt.figure(figsize=(10,5))
plt.scatter(range(df_train.shape[0]), np.sort(df_train.y.values))
plt.xlabel('index',fontsize=10)
plt.ylabel('y value',fontsize=10)
plt.grid()
plt.show()
plt.figure(figsize=(10,5))
sns.distplot(df_train.y.values, bins=50, kde=False)
plt.xlabel('y value distribution',fontsize=10)
plt.grid()
plt.show()
dtype_df = df_train.dtypes.reset_index()
dtype_df.columns = ['Count', "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
dtype_df.head(10)
df_miss = df_train.isnull().sum(axis=0).reset_index()
df_miss.columns = ['column name', 'missing count']
df_miss = df_miss.loc[df_miss['missing count']>0]
df_miss = df_miss.sort_values(by='missing count')
df_miss
for f in ["X0","X1","X2","X3","X4","X5","X6","X8"]:
    label = preprocessing.LabelEncoder()
    label.fit(list(df_train[f].values))
    df_train[f] = label.transform(list(df_train[f].values))
    
train_y = df_train['y'].values
train_X = df_train.drop(["ID","y"],axis=1)
df_train.tail()
def xgb_r2_score(preds, dtrain):
    labels = dtrain.get_label()
    return 'r2', r2_score(labels, preds)
xgb_params = {
    'eta': 0.05,
    'max_depth': 3,
    'subsample' : 0.7,
    'colsample_bytree' : 0.7,
    'objective' : 'reg:linear',
    'silent' : 1
}
dtrain = xgb.DMatrix(train_X, train_y,feature_names=train_X.columns.values)
model = xgb.train(dict(xgb_params,silent=0),dtrain,num_boost_round=1000,feval=xgb_r2_score,maximize=True)
fig, ax = plt.subplots(figsize=(10,18))
xgb.plot_importance(model, max_num_features=50,height=0.9,ax=ax)
plt.show()
for f in ["X0","X1","X2","X3","X4","X5","X6","X8"]:
    label = preprocessing.LabelEncoder()
    label.fit(list(df_test[f].values))
    df_test[f] = label.transform(list(df_test[f].values))
    
df_test.tail()
from sklearn.decomposition import PCA, FastICA
n_comp = 15

# PCA
pca = PCA(n_components=n_comp, random_state=42)
pca2_results_train = pca.fit_transform(df_train.drop(["y"], axis=1))
pca2_results_test = pca.transform(df_test)

# ICA
ica = FastICA(n_components=n_comp, random_state=42)
ica2_results_train = ica.fit_transform(df_train.drop(["y"], axis=1))
ica2_results_test = ica.transform(df_test)

# Append decomposition components to datasets
for i in range(1, n_comp+1):
    df_train['pca_' + str(i)] = pca2_results_train[:,i-1]
    df_test['pca_' + str(i)] = pca2_results_test[:, i-1]
    
    df_train['ica_' + str(i)] = ica2_results_train[:,i-1]
    df_test['ica_' + str(i)] = ica2_results_test[:, i-1]
    
y_train = df_train["y"]
y_mean = np.mean(y_train)
xgb_params = {
    'n_trees': 500, 
    'eta': 0.005,
    'max_depth': 4,
    'subsample': 0.95,
    'objective': 'reg:linear',
    'base_score': y_mean, # base prediction = mean(target)
    'silent': 1
}

# form DMatrices for Xgboost training
dtrain = xgb.DMatrix(df_train.drop('y', axis=1), y_train)
dtest = xgb.DMatrix(df_test)

# xgboost, cross-validation
cv_result = xgb.cv(xgb_params, 
                   dtrain, 
                   num_boost_round=800, # increase to have better results (~700)
                   early_stopping_rounds=50,
                   verbose_eval=50, 
                   show_stdv=False
                  )

num_boost_rounds = len(cv_result)
print(num_boost_rounds)

# train model
model = xgb.train(dict(xgb_params, silent=0), dtrain,feval=xgb_r2_score, num_boost_round=num_boost_rounds)
from sklearn.metrics import r2_score

print(r2_score(dtrain.get_label(), model.predict(dtrain)))
y_pred = model.predict(dtest)
output = pd.DataFrame({'id': df_test['ID'].astype(np.int32), 'y': y_pred})
output.to_csv('xgboost-depth{}-pca-ica.csv'.format(xgb_params['max_depth']), index=False)

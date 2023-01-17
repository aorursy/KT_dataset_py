import numpy as np
import pandas as pd
import gc

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import xgboost as xgb

from IPython.display import display # Allows the use of display() for DataFrames

import warnings
warnings.filterwarnings('ignore')
# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))
# check and remove constant columns
colsToRemove = []
for col in X_train.columns:
    if X_train[col].std() == 0: 
        colsToRemove.append(col)
        
# remove constant columns in the training set
X_train.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
X_test.drop(colsToRemove, axis=1, inplace=True) 

print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))
print(colsToRemove)
gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))
%%time
# The other way to drop duplicate columns is to transpose our DatFrame and use the pandas routine - drop_duplicates. 
# df.T.drop_duplicates().T. However, transposing is a bad idea when working with large DataFrames. But this option is fine in this case.
# Check and remove duplicate columns
colsToRemove = []
colsScaned = []
dupList = {}

columns = X_train.columns

for i in range(len(columns)-1):
    v = X_train[columns[i]].values
    dupCols = []
    for j in range(i+1,len(columns)):
        if np.array_equal(v, X_train[columns[j]].values):
            colsToRemove.append(columns[j])
            if columns[j] not in colsScaned:
                dupCols.append(columns[j]) 
                colsScaned.append(columns[j])
                dupList[columns[i]] = dupCols
                
# remove duplicate columns in the training set
X_train.drop(colsToRemove, axis=1, inplace=True) 

# remove duplicate columns in the testing set
X_test.drop(colsToRemove, axis=1, inplace=True)

print("Removed `{}` Duplicate Columns\n".format(len(dupList)))
print(dupList)
gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))
def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID','target']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test
%%time
X_train, X_test = drop_sparse(X_train, X_test)
gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))
def add_SumZeros(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target']]
    if 'SumZeros' in features:
        train.insert(1, 'SumZeros', (train[flist] == 0).astype(int).sum(axis=1))
        test.insert(1, 'SumZeros', (test[flist] == 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in ['ID','target']]

    return train, test
%%time
X_train, X_test = add_SumZeros(X_train, X_test, ['SumZeros'])
gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))
def add_SumValues(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target']]
    if 'SumValues' in features:
        train.insert(1, 'SumValues', (train[flist] != 0).astype(int).sum(axis=1))
        test.insert(1, 'SumValues', (test[flist] != 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in ['ID','target']]

    return train, test

%%time
X_train, X_test = add_SumValues(X_train, X_test, ['SumValues'])
gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))
def add_OtherAgg(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target','SumZeros','SumValues']]
    if 'OtherAgg' in features:
        train['Mean']   = train[flist].mean(axis=1)
        train['Median'] = train[flist].median(axis=1)
        train['Mode']   = train[flist].mode(axis=1)
        train['Max']    = train[flist].max(axis=1)
        train['Var']    = train[flist].var(axis=1)
        train['Std']    = train[flist].std(axis=1)
        
        test['Mean']   = test[flist].mean(axis=1)
        test['Median'] = test[flist].median(axis=1)
        test['Mode']   = test[flist].mode(axis=1)
        test['Max']    = test[flist].max(axis=1)
        test['Var']    = test[flist].var(axis=1)
        test['Std']    = test[flist].std(axis=1)
    flist = [x for x in train.columns if not x in ['ID','target','SumZeros','SumValues']]

    return train, test
%%time
X_train, X_test = add_OtherAgg(X_train, X_test, ['OtherAgg'])
gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))
flist = [x for x in X_train.columns if not x in ['ID','target']]

flist_kmeans = []
for ncl in range(2,11):
    cls = KMeans(n_clusters=ncl)
    cls.fit_predict(X_train[flist].values)
    X_train['kmeans_cluster_'+str(ncl)] = cls.predict(X_train[flist].values)
    X_test['kmeans_cluster_'+str(ncl)] = cls.predict(X_test[flist].values)
    flist_kmeans.append('kmeans_cluster_'+str(ncl))
print(flist_kmeans)
gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))
flist = [x for x in X_train.columns if not x in ['ID','target']]

n_components = 20
flist_pca = []
pca = PCA(n_components=n_components)
x_train_projected = pca.fit_transform(normalize(X_train[flist], axis=0))
x_test_projected = pca.transform(normalize(X_test[flist], axis=0))
for npca in range(0, n_components):
    X_train.insert(1, 'PCA_'+str(npca+1), x_train_projected[:, npca])
    X_test.insert(1, 'PCA_'+str(npca+1), x_test_projected[:, npca])
    flist_pca.append('PCA_'+str(npca+1))
print(flist_pca)
gc.collect()
print("Train set size: {}".format(X_train.shape))
print("Test set size: {}".format(X_test.shape))
X_train.head()
X_test.head()
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : 0.01,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.7,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgtrain, lgval], early_stopping_rounds=100, 
                      verbose_eval=200, evals_result=evals_result)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model, evals_result
import lightgbm as lgb
# Training LGB
seeds = [42, 2018]
pred_test_full_seed = 0
for seed in seeds:
    kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=seed)
    pred_test_full = 0
    for dev_index, val_index in kf.split(X_train):
        dev_X, val_X = X_train.loc[dev_index,:], X_train.loc[val_index,:]
        dev_y, val_y = y_train[dev_index], y_train[val_index]
        pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
        pred_test_full += pred_test
    pred_test_full /= 5.
    pred_test_full = np.expm1(pred_test_full)
    pred_test_full_seed += pred_test_full
    print("Seed {} completed....".format(seed))
pred_test_full_seed /= np.float(len(seeds))

print("LightGBM Training Completed...")
# feature importance
print("Features Importance...")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:15])
sub = pd.read_csv('../input/sample_submission.csv')
sub["target"] = pred_test_full_seed
print(sub.head())
sub.to_csv('LGB1.csv', index=False)
!ls

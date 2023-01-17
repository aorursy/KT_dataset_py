#Processing
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.model_selection import KFold
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import gc
#Plotting
import seaborn as sns

#Models
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
train_df.head()
test_df.head()
X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)
train_df.info()
test_df.info()
print("Total Train Features with NaN Values = " + str(train_df.columns[train_df.isnull().sum() != 0].size))
if (train_df.columns[train_df.isnull().sum() != 0].size):
    print("Features with NaN => {}".format(list(train_df.columns[train_df.isnull().sum() != 0])))
    train_df[train_df.columns[train_df.isnull().sum() != 0]].isnull().sum().sort_values(ascending = False)
zero_count = []
for col in X_train.columns[2:]:
    zero_count.append([i[1] for i in list(X_train[col].value_counts().items()) if i[0] == 0][0])
    
print('{0} features of 4491 have zeroes in 99% or more samples.'.format(len([i for i in zero_count if i >= 4459 * 0.99])))
print('{0} features of 4491 have zeroes in 98% or more samples.'.format(len([i for i in zero_count if i >= 4459 * 0.98])))
print('{0} features of 4491 have zeroes in 97% or more samples.'.format(len([i for i in zero_count if i >= 4459 * 0.97])))
print('{0} features of 4491 have zeroes in 96% or more samples.'.format(len([i for i in zero_count if i >= 4459 * 0.96])))
print('{0} features of 4491 have zeroes in 95% or more samples.'.format(len([i for i in zero_count if i >= 4459 * 0.95])))

cols_to_drop = [col for col in X_train.columns[2:] if [i[1] for i in list(X_train[col].value_counts().items()) if i[0] == 0][0] >= 4459 * 0.98]

X_train.drop(cols_to_drop, axis=1, inplace=True)
X_test.drop(cols_to_drop, axis=1, inplace=True)

print('\nTrain shape: {}\nTest shape: {}'.format(X_train.shape, X_test.shape))
colsToRemove = []
for col in X_train.columns:
    if X_train[col].std() == 0: 
        colsToRemove.append(col)
        
# remove constant columns in the training set
train_df.drop(colsToRemove, axis=1, inplace=True)

# remove constant columns in the test set
test_df.drop(colsToRemove, axis=1, inplace=True) 

print("Removed `{}` Constant Columns\n".format(len(colsToRemove)))
print(colsToRemove)
print('\nTrain shape: {}\nTest shape: {}'.format(X_train.shape, X_test.shape))
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

print('\nTrain shape: {}\nTest shape: {}'.format(X_train.shape, X_test.shape))
def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID','target']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test
X_train, X_test = drop_sparse(X_train, X_test)

print('\nTrain shape: {}\nTest shape: {}'.format(X_train.shape, X_test.shape))
def add_SumZeros(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target']]
    if 'SumZeros' in features:
        train.insert(1, 'SumZeros', (train[flist] == 0).astype(int).sum(axis=1))
        test.insert(1, 'SumZeros', (test[flist] == 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in ['ID','target']]

    return train, test
# X_train, X_test = add_SumZeros(X_train, X_test, ['SumZeros'])
def add_SumValues(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target']]
    if 'SumValues' in features:
        train.insert(1, 'SumValues', (train[flist] != 0).astype(int).sum(axis=1))
        test.insert(1, 'SumValues', (test[flist] != 0).astype(int).sum(axis=1))
    flist = [x for x in train.columns if not x in ['ID','target']]

    return train, test
# X_train, X_test = add_SumValues(X_train, X_test, ['SumValues'])
def add_OtherAgg(train, test, features):
    flist = [x for x in train.columns if not x in ['ID','target','SumZeros','SumValues']]
    if 'OtherAgg' in features:
        train['Mean'] = train.mean(axis=1)
        train['Median'] = train.median(axis=1)
        train['Mode'] = train.mode(axis=1)
        train['Max'] = train.max(axis=1)
        train['Var'] = train.var(axis=1)
        train['Std'] = train.std(axis=1)
        
        test['Mean'] = test.mean(axis=1)
        test['Median'] = test.median(axis=1)
        test['Mode'] = test.mode(axis=1)
        test['Max'] = test.max(axis=1)
        test['Var'] = test.var(axis=1)
        test['Std'] = test.std(axis=1)
    flist = [x for x in train.columns if not x in ['ID','target','SumZeros','SumValues']]

    return train, test
def kmeans(X_Tr,Xte):
    flist = [x for x in X_Tr.columns if not x in ['ID','target']]
    flist_kmeans = []
    for ncl in range(2,11):
        cls = KMeans(n_clusters=ncl)
        cls.fit_predict(X_train[flist].values)
        X_Tr['kmeans_cluster_'+str(ncl)] = cls.predict(X_Tr[flist].values)
        Xte['kmeans_cluster_'+str(ncl)] = cls.predict(Xte[flist].values)
        flist_kmeans.append('kmeans_cluster_'+str(ncl))
    print(flist_kmeans)
    
    return X_Tr,Xte
def pca(X_Tr,Xte):
    flist = [x for x in X_Tr.columns if not x in ['ID','target']]
    n_components = 20
    flist_pca = []
    pca = PCA(n_components=n_components)
    x_train_projected = pca.fit_transform(StandardScaler(X_Tr[flist], axis=0))
    x_test_projected = pca.transform(StandardScaler(X_test[flist], axis=0))
    for npca in range(0, n_components):
        X_Tr.insert(1, 'PCA_'+str(npca+1), x_train_projected[:, npca])
        Xte.insert(1, 'PCA_'+str(npca+1), x_test_projected[:, npca])
        flist_pca.append('PCA_'+str(npca+1))
    print(flist_pca)
print('\nTrain shape: {}\nTest shape: {}'.format(X_train.shape, X_test.shape))
PERC_TRESHOLD = 0.98   ### Percentage of zeros in each feature ###
N_COMP = 97            ### Number of decomposition components ###

print("\nStart decomposition process...")
print("PCA")
pca = PCA(n_components=N_COMP, random_state=17)
pca_results_train = pca.fit_transform(X_train)
pca_results_test = pca.transform(X_test)
print(pca.explained_variance_ratio_)

print("tSVD")
tsvd = TruncatedSVD(n_components=N_COMP, random_state=17)
tsvd_results_train = tsvd.fit_transform(X_train)
tsvd_results_test = tsvd.transform(X_test)

print("ICA")
ica = FastICA(n_components=N_COMP, random_state=17)
ica_results_train = ica.fit_transform(X_train)
ica_results_test = ica.transform(X_test)

print("GRP")
grp = GaussianRandomProjection(n_components=N_COMP, eps=0.1, random_state=17)
grp_results_train = grp.fit_transform(X_train)
grp_results_test = grp.transform(X_test)

print("SRP")
srp = SparseRandomProjection(n_components=N_COMP, dense_output=True, random_state=17)
srp_results_train = srp.fit_transform(X_train)
srp_results_test = srp.transform(X_test)

print("Append decomposition components to datasets...")
for i in range(1, N_COMP + 1):
    X_train['pca_' + str(i)] = pca_results_train[:, i - 1]
    X_test['pca_' + str(i)] = pca_results_test[:, i - 1]
    
    X_train['ica_' + str(i)] = ica_results_train[:, i - 1]
    X_test['ica_' + str(i)] = ica_results_test[:, i - 1]

    X_train['tsvd_' + str(i)] = tsvd_results_train[:, i - 1]
    X_test['tsvd_' + str(i)] = tsvd_results_test[:, i - 1]

    X_train['grp_' + str(i)] = grp_results_train[:, i - 1]
    X_test['grp_' + str(i)] = grp_results_test[:, i - 1]

    X_train['srp_' + str(i)] = srp_results_train[:, i - 1]
    X_test['srp_' + str(i)] = srp_results_test[:, i - 1]
print('\nTrain shape: {}\nTest shape: {}'.format(X_train.shape, X_test.shape))

print(tsvd.explained_variance_ratio_)
sum1 = 0
for i in range(len(tsvd.explained_variance_ratio_)):
    sum1 = sum1 + tsvd.explained_variance_ratio_[i]

print(sum1)
sum = 0
for i in range(len(pca.explained_variance_ratio_)):
    sum = sum + pca.explained_variance_ratio_[i]

print(sum)
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
    'task': 'train',
    'boosting_type': 'dart',
    'objective': 'regression',
    'metric': 'rmse',
    "learning_rate": 0.01,
    "num_leaves": 200,
    "feature_fraction": 0.50,
    "bagging_fraction": 0.50,
    'bagging_freq': 4,
    "max_depth": -1,
    "reg_alpha": 0.3,
    "reg_lambda": 0.1,
    #"min_split_gain":0.2,
    "min_child_weight":10,
    'zero_as_missing':True
}
    
#     {'learning_rate':0.008,
#         'boosting_type':'gbdt',
#         'objective':'regression',
#         'metric':'rmse',
#         'sub_feature':0.5,
#         'num_leaves':180,
#         'feature_fraction': 0.5,
#         'bagging_fraction': 0.5,
#         'min_data':50,
#         'max_depth':-1,
#         'reg_alpha': 0.3, 
#         'reg_lambda': 0.1, 
#         'min_child_weight': 10, 
#         'verbose': 1,
#         'nthread':5,
#         'max_bin':512,
#         'subsample_for_bin':200,
#         'min_split_gain':0.0001,
#         'min_child_samples':5
#        }


#     {
#         "objective" : "regression",
#         "metric" : "rmse",
#         "num_leaves" : 40,
#         "learning_rate" : 0.005,
#         "bagging_fraction" : 0.7,
#         "feature_fraction" : 0.5,
#         "bagging_frequency" : 5,
#         "bagging_seed" : 42,
#         "verbosity" : -1,
#         "random_seed": 42
#     }
#     {
#     'task': 'train',
#     'boosting_type': 'dart',
#     'objective': 'regression',
#     'metric': 'rmse',
#     "learning_rate": 0.01, # Try 0.05
#     "num_leaves": 200, # 24
#     "feature_fraction": 0.50, 
#     "bagging_fraction": 0.90,
#     'bagging_freq': 4,
#     "max_depth": -1,
#     "reg_alpha": 0.3,
#     "reg_lambda": 0.1,
#     #"min_split_gain":0.2,
#     "min_child_weight":10,
#     'zero_as_missing':True
#     'nthread': 32,
#     'lambda_l1':0.1, 
#     'lambda_l2':0.1,
    
    
# }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=50, 
                      evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result
def run_xgb(train_X, train_y, val_X, val_y, test_X):
    params = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'eta': 0.005, 'max_depth': 10, 'subsample': 0.7, 'colsample_bytree': 0.5, 'alpha':0, 'silent': True, 'random_state':5}
# {'objective': 'reg:linear','eval_metric': 'rmse','eta': 0.005,'max_depth': 15,'subsample': 0.7,'colsample_bytree': 0.5,'alpha':0,'random_state':42,'silent': True}
    
    tr_data = xgb.DMatrix(train_X, train_y)
    va_data = xgb.DMatrix(val_X, val_y)
    
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    
    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 30, verbose_eval=100)
    
    dtest = xgb.DMatrix(test_X)
    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))
    
    return xgb_pred_y, model_xgb
#Split data
dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)


# Training LGB
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
print("LightGBM Training Completed...")

print("Features Importance...")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:15])

# Training XGB
pred_test_xgb, model_xgb = run_xgb(dev_X, dev_y, val_X, val_y, X_test)
print("XGB Training Completed...")
sub = pd.read_csv('../input/sample_submission.csv')

sub_lgb = pd.DataFrame()
sub_lgb["target"] = pred_test
sub_lgb["ID"] = sub["ID"]
sub_lgb.to_csv("sub_lgb.csv", index=False)

sub_xgb = pd.DataFrame()
sub_xgb["target"] = pred_test_xgb
sub_xgb["ID"] = sub["ID"]
sub_lgb.to_csv("sub_xgb.csv", index=False)


sub["target"] = (sub_lgb["target"] + sub_xgb["target"] )/2

print(sub.head())
sub.to_csv('sub_lgb_xgb.csv', index=False)


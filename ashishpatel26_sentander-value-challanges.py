### Import required libraries

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb
import xgboost as xgb

from IPython.display import display # Allows the use of display() for DataFrames

import warnings
warnings.filterwarnings('ignore')
import gc

%%time
# Read train and test files
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

gc.collect()
%%time
# training set
print ("Training set:")
n_data  = len(train_df)
n_features = train_df.shape[1]
print ("Number of Records: {}".format(n_data))
print ("Number of Features: {}".format(n_features))

# testing set
print ("\nTesting set:")
n_data  = len(test_df)
n_features = test_df.shape[1]
print ("Number of Records: {}".format(n_data))
print ("Number of Features: {}".format(n_features))
gc.collect()
X_train = train_df.drop(["ID", "target"], axis=1)
y_train = np.log1p(train_df["target"].values)

X_test = test_df.drop(["ID"], axis=1)
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
def drop_sparse(train, test):
    flist = [x for x in train.columns if not x in ['ID','target']]
    for f in flist:
        if len(np.unique(train[f]))<2:
            train.drop(f, axis=1, inplace=True)
            test.drop(f, axis=1, inplace=True)
    return train, test
X_train, X_test = drop_sparse(X_train, X_test)

print('\nTrain shape: {}\nTest shape: {}'.format(X_train.shape, X_test.shape))
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
    x_train_projected = pca.fit_transform(normalize(X_Tr[flist], axis=0))
    x_test_projected = pca.transform(normalize(X_test[flist], axis=0))
    for npca in range(0, n_components):
        X_Tr.insert(1, 'PCA_'+str(npca+1), x_train_projected[:, npca])
        Xte.insert(1, 'PCA_'+str(npca+1), x_test_projected[:, npca])
        flist_pca.append('PCA_'+str(npca+1))
    print(flist_pca)
print('\nTrain shape: {}\nTest shape: {}'.format(X_train.shape, X_test.shape))
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
PERC_TRESHOLD = 0.98   ### Percentage of zeros in each feature ###
N_COMP = 20            ### Number of decomposition components ###

print("\nStart decomposition process...")
print("PCA")
pca = PCA(n_components=N_COMP, random_state=17)
pca_results_train = pca.fit_transform(X_train)
pca_results_test = pca.transform(X_test)

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

print(X_train.shape, y_train.shape)
print(X_test.shape)
# Find feature importance
clf_gb = GradientBoostingRegressor(random_state = 42)
clf_gb.fit(X_train, y_train)
print(clf_gb)
# GradientBoostingRegressor feature importance - top 100
feat_importances = pd.Series(clf_gb.feature_importances_, index=X_train.columns)
feat_importances = feat_importances.nlargest(100)
plt.figure(figsize=(16,15))
feat_importances.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
# GradientBoostingRegressor feature importance - top 25
feat_importances_gb = pd.Series(clf_gb.feature_importances_, index=X_train.columns)
feat_importances_gb = feat_importances_gb.nlargest(100)
plt.figure(figsize=(16,8))
feat_importances_gb.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print(pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(100))
# Find feature importance
clf_rf = RandomForestRegressor(random_state = 42)
clf_rf.fit(X_train, y_train)
print(clf_rf)
# RandomForestRegressor feature importance - top 25
feat_importances_rf = pd.Series(clf_rf.feature_importances_, index=X_train.columns)
feat_importances_rf = feat_importances_rf.nlargest(100)
plt.figure(figsize=(16,8))
feat_importances_gb.plot(kind='barh')
plt.gca().invert_yaxis()
plt.show()
print(pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(10))
plt.figure()
fig, ax = plt.subplots(1, 2, figsize=(16,6))
feat_importances_gb.plot(kind='barh', ax=ax[0])
feat_importances_rf.plot(kind='barh', ax=ax[1])
ax[0].invert_yaxis()
ax[1].invert_yaxis()
plt.show()
s1 = pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(10).index
s2 = pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(10).index

common_features = pd.Series(list(set(s1).intersection(set(s2)))).values

print(common_features)
df_plot = X_train[['f190486d6', 'eeb9cd3aa', '58e2e02e6', '58232a6fb', '15ace8c9f', '9fd594eec']]
df_plot['target'] = y_train

g = sns.pairplot(df_plot, diag_kind="kde", palette="BuGn_r")
g.fig.suptitle('Pairplot of Top 6 Important Features',fontsize=26)
# PLot Correlation HeatMap for top 20 features from GB and RF Models
s1 = pd.Series(clf_gb.feature_importances_, index=X_train.columns).nlargest(100).index
s2 = pd.Series(clf_rf.feature_importances_, index=X_train.columns).nlargest(100).index

common_features = pd.Series(list(set(s1).union(set(s2)))).values

print(common_features)
print(len(common_features))
df_plot = pd.DataFrame(X_train, columns = common_features)
corr = df_plot.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 16))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Correlation HeatMap", fontsize=15)
plt.show()
X_train = pd.DataFrame(X_train, columns = common_features)
X_test = pd.DataFrame(X_test, columns= common_features)
X_train.shape,y_train.shape
dev_X, val_X, dev_y, val_y = train_test_split(X_train, y_train, test_size = 0.2, random_state = 42)
def run_lgb(train_X, train_y, val_X, val_y, test_X):
#     params = {
#         "objective" : "regression",
#         "metric" : "rmse",
#         "num_leaves" : 40,
#         "learning_rate" : 0.005,
#         "bagging_fraction" : 0.7,
#         "feature_fraction" : 0.6,
#         "bagging_frequency" : 6,
#         "bagging_seed" : 42,
#         "verbosity" : -1,
#         "seed": 42
#     }
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
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    evals_result = {}
    model = lgb.train(params, lgtrain, 5000, 
                      valid_sets=[lgtrain, lgval], 
                      early_stopping_rounds=100, 
                      verbose_eval=150, 
                      evals_result=evals_result)
    
    pred_test_y = np.expm1(model.predict(test_X, num_iteration=model.best_iteration))
    return pred_test_y, model, evals_result
# Training LGB# Traini 
pred_test, model, evals_result = run_lgb(dev_X, dev_y, val_X, val_y, X_test)
print("LightGBM Training Completed...")
print("Features Importance...")
gain = model.feature_importance('gain')
featureimp = pd.DataFrame({'feature':model.feature_name(), 
                   'split':model.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(featureimp[:15])
def run_xgb(train_X, train_y, val_X, val_y, test_X):
#     params = {'objective': 'reg:linear', 
#           'eval_metric': 'rmse',
#           'eta': 0.001,
#           'max_depth': 10, 
#           'subsample': 0.6, 
#           'colsample_bytree': 0.6,
#           'alpha':0.001,
#           'random_state': 42, 
#           'silent': True}
    params = {'objective': 'reg:linear', 'eval_metric': 'rmse', 'eta': 0.005, 'max_depth': 10, 'subsample': 0.7, 'colsample_bytree': 0.5, 'alpha':0, 'silent': True, 'random_state':5}
    
    tr_data = xgb.DMatrix(train_X, train_y)
    va_data = xgb.DMatrix(val_X, val_y)
    
    watchlist = [(tr_data, 'train'), (va_data, 'valid')]
    
    model_xgb = xgb.train(params, tr_data, 2000, watchlist, maximize=False, early_stopping_rounds = 100, verbose_eval=100)
    
    dtest = xgb.DMatrix(test_X)
    xgb_pred_y = np.expm1(model_xgb.predict(dtest, ntree_limit=model_xgb.best_ntree_limit))
    
    return xgb_pred_y, model_xgb
# Training XGB
pred_test_xgb, model_xgb = run_xgb(dev_X, dev_y, val_X, val_y, X_test)
print("XGB Training Completed...")
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
# cb_model = CatBoostRegressor(iterations=500,
#                              learning_rate=0.05,
#                              depth=10,
#                              eval_metric='RMSE',
#                              random_seed = 42,
#                              bagging_temperature = 0.2,
#                              od_type='Iter',
#                              metric_period = 50,
#                              od_wait=20)
cb_model = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=10, l2_leaf_reg=20, bootstrap_type='Bernoulli', eval_metric='RMSE', metric_period=50, od_type='Iter', od_wait=45, random_seed=42, allow_writing_files=False)
cb_model.fit(dev_X, dev_y,
             eval_set=(val_X, val_y),
             use_best_model=True,
             verbose=True)
pred_test_cat = np.expm1(cb_model.predict(X_test))
sub = pd.read_csv('../input/sample_submission.csv')

sub_lgb = pd.DataFrame()
sub_lgb["target"] = pred_test

sub_xgb = pd.DataFrame()
sub_xgb["target"] = pred_test_xgb

sub_cat = pd.DataFrame()
sub_cat["target"] = pred_test_cat

sub["target"] = (sub_lgb["target"] + sub_xgb["target"] + sub_cat["target"])/3
print(sub.head())
sub.to_csv('final.csv', index=False)

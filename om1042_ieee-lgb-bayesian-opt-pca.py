#  Libraries

import numpy as np 

import pandas as pd 

# Data processing, metrics and modeling

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split, StratifiedKFold

from bayes_opt import BayesianOptimization

from datetime import datetime

from sklearn.metrics import precision_score, recall_score, confusion_matrix, accuracy_score, roc_auc_score, f1_score, roc_curve, auc

from sklearn import metrics

from sklearn import preprocessing



from sklearn.decomposition import PCA, FastICA,SparsePCA,KernelPCA

from sklearn.decomposition import TruncatedSVD

from sklearn.manifold import TSNE

from sklearn.random_projection import GaussianRandomProjection

from sklearn.random_projection import SparseRandomProjection



# Lgbm

import lightgbm as lgb

# Suppr warning

import warnings

warnings.filterwarnings("ignore")



import itertools

from scipy import interp



# Plots

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
%%time

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')
# merge 

train_df = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test_df = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)



print("Train shape : "+str(train_df.shape))

print("Test shape  : "+str(test_df.shape))
pd.set_option('display.max_columns', 500)
# GPreda, missing data

def missing_data(data):

    total = data.isnull().sum()

    percent = (data.isnull().sum()/data.isnull().count()*100)

    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    types = []

    for col in data.columns:

        dtype = str(data[col].dtype)

        types.append(dtype)

    tt['Types'] = types

    return(np.transpose(tt))
display(missing_data(train_df), missing_data(test_df))
#fillna

train_df = train_df.fillna(-999)

test_df = test_df.fillna(-999)
del train_transaction, train_identity, test_transaction, test_identity
# Label Encoding

for f in train_df.columns:

    if  train_df[f].dtype=='object': 

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[f].values) + list(test_df[f].values))

        train_df[f] = lbl.transform(list(train_df[f].values))

        test_df[f] = lbl.transform(list(test_df[f].values))  

train_df = train_df.reset_index()

test_df = test_df.reset_index()
#PCA/ICA for dimensionality reduction



n_comp = 12



# tSVD

tsvd = TruncatedSVD(n_components=n_comp, random_state=420)

tsvd_results_train = tsvd.fit_transform(train_df.drop(["isFraud"], axis=1))

tsvd_results_test = tsvd.transform(test_df)



# # PCA

# pca = PCA(n_components=n_comp, random_state=420)

# pca2_results_train = pca.fit_transform(train.drop(["y"], axis=1))

# pca2_results_test = pca.transform(test)

#



#Polynomial features

# poly = PolynomialFeatures(degree=1)

# poly_results_train = poly.fit_transform(train.drop(["y"], axis=1))

# poly_results_test = poly.transform(test)



#sparse PCA

spca = SparsePCA(n_components=n_comp, random_state=420)

spca2_results_train = spca.fit_transform(train_df.drop(["isFraud"], axis=1))

spca2_results_test = spca.transform(test_df)



#Kernel PCA

# kpca = KernelPCA(n_components=n_comp, random_state=420)

# kpca2_results_train = kpca.fit_transform(train)

# kpca2_results_test = kpca.transform(test)



# ICA

ica = FastICA(n_components=n_comp, random_state=420)

ica2_results_train = ica.fit_transform(train_df.drop(["isFraud"], axis=1))

ica2_results_test = ica.transform(test_df)



# GRP

grp = GaussianRandomProjection(n_components=n_comp, eps=0.1, random_state=420)

grp_results_train = grp.fit_transform(train_df.drop(["isFraud"], axis=1))

grp_results_test = grp.transform(test_df)



# SRP

srp = SparseRandomProjection(n_components=n_comp, dense_output=True, random_state=420)

srp_results_train = srp.fit_transform(train_df.drop(["isFraud"], axis=1))

srp_results_test = srp.transform(test_df)







# Append decomposition components to datasets

for i in range(1, n_comp + 1):

    # train['pca_' + str(i)] = pca2_results_train[:, i - 1]

    # test['pca_' + str(i)] = pca2_results_test[:, i - 1]



    # train['poly_' + str(i)] = poly_results_train[:, i - 1]

    # test['poly_' + str(i)] = poly_results_test[:, i - 1]



    train_df['spca_' + str(i)] = spca2_results_train[:, i - 1]

    test_df['spca_' + str(i)] = spca2_results_test[:, i - 1]



#     train['kpca_' + str(i)] = kpca2_results_train[:, i - 1]

#     test['kpca_' + str(i)] = kpca2_results_test[:, i - 1]



    train_df['ica_' + str(i)] = ica2_results_train[:, i - 1]

    test_df['ica_' + str(i)] = ica2_results_test[:, i - 1]



    train_df['grp_' + str(i)] = grp_results_train[:, i - 1]

    test_df['grp_' + str(i)] = grp_results_test[:, i - 1]



    train_df['srp_' + str(i)] = srp_results_train[:, i - 1]

    test_df['srp_' + str(i)] = srp_results_test[:, i - 1]





print("After PCA/ICA")

print (len(list(train_df)))

print (len(list(test_df)))
features = list(train_df)

features.remove('isFraud')

target = 'isFraud'
#cut tr and val

bayesian_tr_idx, bayesian_val_idx = train_test_split(train_df, test_size = 0.5, random_state = 42, stratify = train_df[target])

bayesian_tr_idx = bayesian_tr_idx.index

bayesian_val_idx = bayesian_val_idx.index
#black box LGBM 

def LGB_bayesian(

    learning_rate,

    num_leaves, 

    bagging_fraction,

    feature_fraction,

    min_child_samples, 

    min_child_weight,

    subsample, 

    min_data_in_leaf,

    max_depth,

    colsample_bytree,

    reg_alpha,

    reg_lambda

     ):

    

    # LightGBM expects next three parameters need to be integer. 

    num_leaves = int(num_leaves)

    min_data_in_leaf = int(min_data_in_leaf)

    max_depth = int(max_depth)



    assert type(num_leaves) == int

    assert type(min_data_in_leaf) == int

    assert type(max_depth) == int

    



    param = {

              'num_leaves': num_leaves, 

              'min_child_samples': min_child_samples, 

              'min_data_in_leaf': min_data_in_leaf,

              'min_child_weight': min_child_weight,

              'bagging_fraction' : bagging_fraction,

              'feature_fraction' : feature_fraction,

              'subsample': subsample, 

              'max_depth': max_depth,

              'colsample_bytree': colsample_bytree,

              'reg_alpha': reg_alpha,

              'reg_lambda': reg_lambda,

              'objective': 'binary',

              'save_binary': True,

              'seed': 1337,

              'feature_fraction_seed': 1337,

              'bagging_seed': 1337,

              'drop_seed': 1337,

              'data_random_seed': 1337,

              'boosting_type': 'gbdt',

              'verbose': 1,

              'is_unbalance': False,

              'boost_from_average': True,

              'metric':'auc'}    

    

    oof = np.zeros(len(train_df))

    trn_data= lgb.Dataset(train_df.iloc[bayesian_tr_idx][features].values, label=train_df.iloc[bayesian_tr_idx][target].values)

    val_data= lgb.Dataset(train_df.iloc[bayesian_val_idx][features].values, label=train_df.iloc[bayesian_val_idx][target].values)



    clf = lgb.train(param, trn_data,  num_boost_round=50, valid_sets = [trn_data, val_data], verbose_eval=50, early_stopping_rounds = 50)

    

    oof[bayesian_val_idx]  = clf.predict(train_df.iloc[bayesian_val_idx][features].values, num_iteration=clf.best_iteration)  

    

    score = roc_auc_score(train_df.iloc[bayesian_val_idx][target].values, oof[bayesian_val_idx])



    return score
# Bounded region of parameter space

bounds_LGB = {

    'num_leaves': (5, 200), 

    'min_data_in_leaf': (5, 200),

    'bagging_fraction' : (0.1,0.9),

    'feature_fraction' : (0.1,0.9),

    'learning_rate': (0.01, 0.3),

    'min_child_weight': (0.00001, 0.01),   

    'min_child_samples':(100, 500), 

    'subsample': (0.2, 0.8),

    'colsample_bytree': (0.4, 0.6), 

    'reg_alpha': (1, 2), 

    'reg_lambda': (1, 2),

    'max_depth':(-1,15),

}
LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=42)
print(LGB_BO.space.keys)
init_points = 30

n_iter = 3
print('-' * 130)



with warnings.catch_warnings():

    warnings.filterwarnings('ignore')

    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)
LGB_BO.max['target']
LGB_BO.max['params']
# Confusion matrix 

def plot_confusion_matrix(cm, classes,

                          normalize = False,

                          title = 'Confusion matrix"',

                          cmap = plt.cm.Blues) :

    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation = 0)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])) :

        plt.text(j, i, cm[i, j],

                 horizontalalignment = 'center',

                 color = 'white' if cm[i, j] > thresh else 'black')

 

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
param_lgb = {

        'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']), 

        'num_leaves': int(LGB_BO.max['params']['num_leaves']), 

        'learning_rate': LGB_BO.max['params']['learning_rate'],

        'min_child_weight': LGB_BO.max['params']['min_child_weight'],

        'colsample_bytree' : LGB_BO.max['params']['colsample_bytree'],

        'bagging_fraction': LGB_BO.max['params']['bagging_fraction'], 

        'min_child_samples': LGB_BO.max['params']['min_child_samples'],

        'subsample': LGB_BO.max['params']['subsample'],

        'reg_lambda': LGB_BO.max['params']['reg_lambda'],

        'reg_alpha': LGB_BO.max['params']['reg_alpha'],

        'max_depth': int(LGB_BO.max['params']['max_depth']), 

        'objective': 'binary',

        'save_binary': True,

        'seed': 1337,

        'feature_fraction_seed': 1337,

        'bagging_seed': 1337,

        'drop_seed': 1337,

        'data_random_seed': 1337,

        'boosting_type': 'gbdt',

        'verbose': 1,

        'is_unbalance': False,

        'boost_from_average': True,

        'metric':'auc'

    }
nfold = 5

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)



oof = np.zeros(len(train_df))

mean_fpr = np.linspace(0,1,100)

cms= []

tprs = []

aucs = []

recalls = []

roc_aucs = []

f1_scores = []

accuracies = []

precisions = []

predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()



i = 1

for train_idx, valid_idx in skf.split(train_df, train_df.isFraud.values):

    print("\nfold {}".format(i))

    trn_data = lgb.Dataset(train_df.iloc[train_idx][features].values,

                                   label=train_df.iloc[train_idx][target].values

                                   )

    val_data = lgb.Dataset(train_df.iloc[valid_idx][features].values,

                                   label=train_df.iloc[valid_idx][target].values

                                   )   

    

    clf = lgb.train(param_lgb, trn_data, num_boost_round=800, valid_sets = [trn_data, val_data], verbose_eval=100, early_stopping_rounds = 100)

    oof[valid_idx] = clf.predict(train_df.iloc[valid_idx][features].values) 

    

    predictions += clf.predict(test_df[features]) / nfold

    

    # Scores 

    roc_aucs.append(roc_auc_score(train_df.iloc[valid_idx][target].values, oof[valid_idx]))

    accuracies.append(accuracy_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))

    recalls.append(recall_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))

    precisions.append(precision_score(train_df.iloc[valid_idx][target].values ,oof[valid_idx].round()))

    f1_scores.append(f1_score(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))

    

    # Roc curve by fold

    fpr, tpr, t = roc_curve(train_df.iloc[valid_idx][target].values, oof[valid_idx])

    tprs.append(interp(mean_fpr, fpr, tpr))

    roc_auc = auc(fpr, tpr)

    aucs.append(roc_auc)

    plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (i,roc_auc))

    i= i+1

    

    # Confusion matrix by folds

    cms.append(confusion_matrix(train_df.iloc[valid_idx][target].values, oof[valid_idx].round()))

    

    # Features imp

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = features

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = nfold + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



# Metrics



roc_score  = np.mean(roc_aucs)

print(

        '\nCV roc score        : {0:.4f}, std: {1:.4f}.'.format(roc_score, np.std(roc_aucs)),

        '\nCV accuracy score   : {0:.4f}, std: {1:.4f}.'.format(np.mean(accuracies), np.std(accuracies)),

        '\nCV recall score     : {0:.4f}, std: {1:.4f}.'.format(np.mean(recalls), np.std(recalls)),

        '\nCV precision score  : {0:.4f}, std: {1:.4f}.'.format(np.mean(precisions), np.std(precisions)),

        '\nCV f1 score         : {0:.4f}, std: {1:.4f}.'.format(np.mean(f1_scores), np.std(f1_scores))

)



plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'grey')

mean_tpr = np.mean(tprs, axis=0)

mean_auc = auc(mean_fpr, mean_tpr)

plt.plot(mean_fpr, mean_tpr, color='blue',

         label=r'Mean ROC (AUC = %0.4f)' % (np.mean(roc_aucs)),lw=2, alpha=1)



plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('LGB ROC curve by folds')

plt.legend(loc="lower right")

plt.show()



# Confusion maxtrix & metrics

cm = np.average(cms, axis=0)

class_names = [0,1]

plt.figure()

plot_confusion_matrix(cm, 

                      classes=class_names, 

                      title= 'LGB Confusion matrix [averaged/folds]')

plt.show()
plt.style.use('dark_background')

cols = (feature_importance_df[["Feature", "importance"]]

    .groupby("Feature")

    .mean()

    .sort_values(by="importance", ascending=False)[:30].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]



plt.figure(figsize=(10,10))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False),

        edgecolor=('white'), linewidth=2, palette="rocket")

plt.title('LGB Features importance (averaged/folds)', fontsize=18)

plt.tight_layout()
sample_submission['isFraud'] = predictions

sample_submission.to_csv('LGB_Bayesian_PCA_{}.csv'.format(roc_score))
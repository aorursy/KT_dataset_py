ver = 'linear_v34'

import warnings

warnings.filterwarnings('ignore')



import gc

import numpy as np

import pandas as pd

import seaborn as sns

from tqdm import tqdm_notebook as tqdm

import time

from datetime import datetime



%matplotlib inline

from matplotlib import pyplot as plt

plt.style.use(['seaborn-darkgrid'])

plt.rcParams['font.family'] = 'DejaVu Sans'



from sklearn import metrics

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold

from sklearn.linear_model import LogisticRegression, Lasso, Ridge, SGDClassifier

from sklearn.feature_selection import RFECV



import eli5

from eli5.sklearn import PermutationImportance



from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import roc_auc_score, log_loss, accuracy_score, confusion_matrix

import os

print(os.listdir("../input"))



from hyperopt import fmin, tpe, hp, anneal, Trials, STATUS_OK



RANDOM_STATE = 78

noise_std = 0.01
filename = 'subm_{}_{}_'.format(ver, datetime.now().strftime('%Y-%m-%d'))



def sub_prp(clf, filename, X_test):

    prediction_ = clf.predict_proba(X_test)[:,1]

    submission_ = pd.read_csv('../input/sample_submission.csv')

    submission_['target'] = prediction_

    submission_.to_csv(filename, index=False)

    print(submission_.head())

    

def sub_pr(clf, filename, X_test):

    prediction_ = clf.predict(X_test)

    submission_ = pd.read_csv('../input/sample_submission.csv')

    submission_['target'] = prediction_

    submission_.to_csv(filename, index=False)

    print(submission_.head())
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.shape, test.shape
train.describe()
train['target'].value_counts().sort_index(ascending=False).plot(kind='barh', figsize=(15,6))

plt.title('Target', fontsize=18)
X_train = train.drop(['id', 'target'], axis=1)

y_train = train['target']

X_tst = test.drop(['id'], axis=1)
sc0 = StandardScaler()

sc0.fit(X_train)

X_train = sc0.transform(X_train)

X_test = sc0.transform(X_tst)

X_train += np.random.normal(0, noise_std, X_train.shape)
#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

repfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=RANDOM_STATE)

logreg0 = LogisticRegression(C=0.5, random_state=RANDOM_STATE, solver='liblinear', penalty='l1')

lass0 = Lasso(alpha=0.031, tol=0.01, selection='random', random_state=RANDOM_STATE)

ridg0 = Ridge(alpha=20, fit_intercept=True, solver='auto', tol=0.0025, random_state=RANDOM_STATE)

sgd0 = SGDClassifier(eta0=1, max_iter=1000, tol=0.0001, random_state=RANDOM_STATE, loss='log')
logreg0.fit(X_train, y_train)

sc = cross_val_score(logreg0, X_train, y_train, scoring='roc_auc', cv=repfold, n_jobs=-1, verbose=1)

print(sc.mean())
perm = PermutationImportance(logreg0, random_state=RANDOM_STATE).fit(X_train, y_train)

eli5.show_weights(perm, top=10)
top_feat = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(perm).feature]

top_feat1 = top_feat[:15]

top_feat1.append('target')

corr = train[top_feat1].corr()

corr.target.sort_values(ascending=False)
mask = np.zeros_like(corr, dtype=np.bool)



plt.subplots(figsize = (15,12))

sns.heatmap(corr, 

            annot=True,

            #mask = mask,

            cmap = 'RdBu',

            linewidths=0.1, 

            linecolor='white',

            vmax = .2,

            square=True)

plt.title("Correlations", y = 1.03,fontsize = 20);
el_df =pd.Series(logreg0.coef_[0], index=range(len(X_train.T)))

el_df = el_df[(logreg0.coef_[0]<=-0.2) | (logreg0.coef_[0]>=0.2)].sort_values(ascending=False)

plt.figure(figsize=(8,6))

el_df.plot(kind='barh')

plt.xlabel("Importance",fontsize=12)

plt.ylabel("Features",fontsize=12)

plt.title("Top Features",fontsize=16)

plt.show()
el_df.index
logreg01 = LogisticRegression(C=0.5, random_state=RANDOM_STATE, solver='liblinear', penalty='l1')
logreg01.fit(X_train.T[el_df.index].T, y_train)

sc = cross_val_score(logreg01, X_train.T[el_df.index].T, y_train, scoring='roc_auc', cv=repfold, n_jobs=-1, verbose=1)

print(sc.mean())
param_lr = {'class_weight' : ['balanced', None], 

                'penalty' : ['l2','l1'],  

                'C' : [0.001, 0.01, 0.1, 1, 10, 100, 1000]

           }
grid_lr = GridSearchCV(estimator = logreg0, param_grid = param_lr , scoring = 'roc_auc', verbose = 1, n_jobs = -1, cv=repfold)



grid_lr.fit(X_train,y_train)



print("Best Score:" + str(grid_lr.best_score_))

print("Best Parameters: " + str(grid_lr.best_params_))

best_parameters_lr = grid_lr.best_params_

logreg = LogisticRegression(**best_parameters_lr)

logreg.fit(X_train,y_train)

sc = cross_val_score(logreg, X_train, y_train, scoring='roc_auc', cv=repfold, n_jobs=-1, verbose=1)

print(sc.mean())



selector_lr = RFECV(logreg, min_features_to_select=12, scoring='roc_auc', step=15, verbose=0, cv=repfold, n_jobs=-1)

selector_lr.fit(X_train,y_train)

#sc = cross_val_score(selector_lr, X_train, y_train, scoring='roc_auc', cv=repfold, n_jobs=-1, verbose=1)

#print(sc.mean())
selector_lr.score(X_train, y_train)
def acc_model(params):

    clf = LogisticRegression(**params)

    return cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=repfold).mean()
space4lr = {'C': hp.uniform('C', .001, 50.0), 

            'solver' : hp.choice('solver', ['liblinear']),

            'penalty' : hp.choice('penalty', ['l1', 'l2']),

            #'dual' : hp.choice('dual', [True, False]),

            #'fit_intercept': hp.choice('fit_intercept', ['True', 'False']),

            'class_weight': hp.choice('class_weight', ['balanced', None]),

            'max_iter': hp.choice('max_iter', [50000]),

            'random_state': RANDOM_STATE, #hp.uniformint('random_state', 1, 100),

            #'n_jobs': -1

           }



best = 0

pr = []

def f(params):

    global best

    acc = acc_model(params)

    if acc > best:

        best = acc

        print ('new best:', best, params)

        pr.append(params)

    return {'loss': -acc, 'status': STATUS_OK}



trials = Trials()

best = fmin(f, space4lr, algo=tpe.suggest, max_evals=2000, trials=trials)

print ('best for logreg: ')

print (pr[-1])
print(pr[-1])

best_lr = pr[-1]

logreg1 = LogisticRegression(**best_lr)

logreg1.fit(X_train,y_train)

sc = cross_val_score(logreg1, X_train, y_train, scoring='roc_auc', cv=repfold, n_jobs=-1, verbose=1)

print(sc.mean())

selector_lr1 = RFECV(logreg1, min_features_to_select=12, scoring='roc_auc', step=15, verbose=0, cv=repfold, n_jobs=-1)

selector_lr1.fit(X_train,y_train)

#print(selector_lr1.score(X_train, y_train))
perm = PermutationImportance(logreg1, random_state=RANDOM_STATE).fit(X_train, y_train)

eli5.show_weights(perm, top=10)
top_feat = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(perm).feature]

top_feat1 = top_feat[:15]

top_feat1.append('target')

corr = train[top_feat1].corr()

corr.target.sort_values(ascending=False)
mask = np.zeros_like(corr, dtype=np.bool)



plt.subplots(figsize = (15,12))

sns.heatmap(corr, 

            annot=True,

            #mask = mask,

            cmap = 'RdBu',

            linewidths=0.1, 

            linecolor='white',

            vmax = .2,

            square=True)

plt.title("Correlations", y = 1.03,fontsize = 20);
lass0.fit(X_train, y_train)

sc = cross_val_score(lass0, X_train, y_train, scoring='roc_auc', cv=repfold, n_jobs=-1, verbose=1)

print(sc.mean())
perm = PermutationImportance(lass0, random_state=RANDOM_STATE).fit(X_train, y_train)

eli5.show_weights(perm, top=10)
top_feat = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(perm).feature]

top_feat1 = top_feat[:15]

top_feat1.append('target')

corr = train[top_feat1].corr()

corr.target.sort_values(ascending=False)
mask = np.zeros_like(corr, dtype=np.bool)



plt.subplots(figsize = (15,12))

sns.heatmap(corr, 

            annot=True,

            #mask = mask,

            cmap = 'RdBu',

            linewidths=0.1, 

            linecolor='white',

            vmax = .2,

            square=True)

plt.title("Correlations", y = 1.03,fontsize = 20);
el_df =pd.Series(lass0.coef_,index=train.drop(['id', 'target'], axis=1).columns)

el_df = el_df[(lass0.coef_<=-0.05) | (lass0.coef_>=0.05)].sort_values(ascending=False)

plt.figure(figsize=(8,6))

el_df.plot(kind='barh')

plt.xlabel("Importance",fontsize=12)

plt.ylabel("Features",fontsize=12)

plt.title("Top Features",fontsize=16)

plt.show()
param_lass = {

            'alpha' : [0.022, 0.021, 0.02, 0.019, 0.023, 0.024, 0.025, 0.026, 0.027, 0.029, 0.031],

            'tol'   : [0.0013, 0.0014, 0.001, 0.0015, 0.0011, 0.0012, 0.0016, 0.0017]

        }
grid_lass = GridSearchCV(estimator = lass0, param_grid = param_lass , scoring = 'roc_auc', verbose = 1, n_jobs = -1, cv=repfold)



grid_lass.fit(X_train,y_train)



print("Best Score:" + str(grid_lass.best_score_))

print("Best Parameters: " + str(grid_lass.best_params_))
best_parameters_lass = grid_lass.best_params_

lass = Lasso(**best_parameters_lass)

lass.fit(X_train,y_train)

sc = cross_val_score(lass, X_train, y_train, scoring='roc_auc', cv=repfold, n_jobs=-1, verbose=1)

print(sc.mean())



selector_lass = RFECV(lass, min_features_to_select=12, scoring='roc_auc', step=15, verbose=0, cv=repfold, n_jobs=-1)

selector_lass.fit(X_train,y_train)

def acc_model(params):

    clf = Lasso(**params)

    return cross_val_score(clf, X_train, y_train, scoring='roc_auc', cv=repfold).mean()
space4lass = {'alpha' : hp.uniform('alpha', .01, 1),

            'tol'   : hp.uniform('tol', .001, 0.1),

            'random_state': RANDOM_STATE, #hp.uniformint('random_state', 1, 100),

            'max_iter': hp.choice('max_iter', [50000]),

             }



best = 0

pr = []

def f(params):

    global best

    acc = acc_model(params)

    if acc > best:

        best = acc

        print ('new best:', best, params)

        pr.append(params)

    return {'loss': -acc, 'status': STATUS_OK}



trials = Trials()

best = fmin(f, space4lass, algo=tpe.suggest, max_evals=3000, trials=trials)

print ('best for lasso: ')

print (pr[-1])
print(pr[-1])

best_lass = pr[-1]

lass1 = Lasso(**best_lass)

lass1.fit(X_train,y_train)

sc = cross_val_score(lass1, X_train, y_train, scoring='roc_auc', cv=repfold, n_jobs=-1, verbose=1)

print(sc.mean())

selector_lass1 = RFECV(lass1, min_features_to_select=12, scoring='roc_auc', step=15, verbose=0, cv=repfold, n_jobs=-1)

selector_lass1.fit(X_train,y_train)
perm = PermutationImportance(lass1, random_state=RANDOM_STATE).fit(X_train, y_train)

eli5.show_weights(perm, top=10)
top_feat = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(perm).feature]

top_feat1 = top_feat[:15]

top_feat1.append('target')

corr = train[top_feat1].corr()

corr.target.sort_values(ascending=False)
mask = np.zeros_like(corr, dtype=np.bool)



plt.subplots(figsize = (15,12))

sns.heatmap(corr, 

            annot=True,

            #mask = mask,

            cmap = 'RdBu',

            linewidths=0.1, 

            linecolor='white',

            vmax = .2,

            square=True)

plt.title("Correlations", y = 1.03,fontsize = 20);
def cross_validation(train_, target_, params,

                            num_folds = 5, repeats = 20, rs = 0):

    

    print(params)

    

    clfs = []

    folds = RepeatedStratifiedKFold(n_splits = num_folds, n_repeats = repeats, random_state = rs)

    

    valid_pred = pd.DataFrame(index = train_.index)

    

    # Cross-validation cycle

    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(target_, target_)):

        print('--- Fold {} started at {}'.format(n_fold, time.ctime()))

        

        train_x, train_y = train_.iloc[train_idx], target_.iloc[train_idx]

        valid_x, valid_y = train_.iloc[valid_idx], target_.iloc[valid_idx]

        

        clf = LogisticRegression(**params)

        clf.fit(train_x, train_y)

    

        clfs.append(clf)



        predict = clf.predict_proba(valid_x)[:, 1]

    

        tn, fp, fn, tp = confusion_matrix(valid_y, (predict >= .5) * 1).ravel()

        auc = roc_auc_score(valid_y, predict)

        acc = accuracy_score(valid_y, (predict >= .5) * 1)

        loss = log_loss(valid_y, predict)

        print('TN =', tn, 'FN =', fn, 'FP =', fp, 'TP =', tp)

        print('AUC = ', auc, 'Loss =', loss, 'Acc =', acc)

        

        valid_pred[n_fold] = pd.Series(predict, index = valid_x.index)



        del train_x, train_y, valid_x, valid_y, predict

        gc.collect()



    return clfs, valid_pred



def save_submit(test_, clfs_, filename):

    subm = pd.DataFrame(np.zeros(test_.shape[0]), index = test_.index, columns = ['target'])

    for clf in clfs_:

        subm['target'] += clf.predict_proba(test_)[:, 1]

    subm['target'] /= len(clfs_)

    subm = subm.reset_index()

    subm.columns = ['id', 'target']

    subm.to_csv(filename, index = False)

    #print(subm)

clfs, pred = cross_validation(pd.DataFrame(X_train), y_train, best_lr)
sub_prp(logreg1, filename+'lr1.csv', X_test)

sub_prp(selector_lr1, filename+'sel_lr1.csv', X_test)
sub_pr(lass1, filename+'lass1.csv', X_test)

sub_pr(selector_lass1, filename+'sel_lass1.csv', X_test)
save_submit(pd.DataFrame(X_test, index=test.id), clfs, filename+'lr_cv_Anna.csv')
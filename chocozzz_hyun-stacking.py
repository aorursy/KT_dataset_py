import warnings

warnings.filterwarnings('ignore')



import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm_notebook

from sklearn.naive_bayes import GaussianNB, BernoulliNB



from sklearn import svm, neighbors, linear_model, neural_network



from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer

from sklearn.model_selection import StratifiedKFold

from sklearn.feature_selection import VarianceThreshold

from sklearn.decomposition import PCA, TruncatedSVD, KernelPCA



from sklearn.mixture import GaussianMixture as GMM

from sklearn.metrics import silhouette_score





import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn import preprocessing

from sklearn import svm, neighbors, linear_model

import gc

warnings.filterwarnings('ignore')





from sklearn.neural_network import MLPClassifier

from sklearn.naive_bayes import GaussianNB, BernoulliNB

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import Matern, RationalQuadratic

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

from sklearn.decomposition import FastICA, TruncatedSVD, PCA

from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

import xgboost as xgb

import catboost as cat



from tqdm import *
%%time

train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_columns = [c for c in train_df.columns if c not in ['id','target','wheezy-copper-turtle-magic']]



magic_variance_over2 = {}

for magic in sorted(train_df['wheezy-copper-turtle-magic'].unique()):

    temp = train_df.loc[train_df['wheezy-copper-turtle-magic']==magic]

    std = temp[train_columns].std()

    magic_variance_over2[magic] = list(std.index.values[np.where(std >2)])
class hist_model(object):

    

    def __init__(self, bins=50):

        self.bins = bins

        

    def fit(self, X):

        

        bin_hight, bin_edge = [], []

        

        for var in X.T:

            # get bins hight and interval

            bh, bedge = np.histogram(var, bins=self.bins)

            bin_hight.append(bh)

            bin_edge.append(bedge)

        

        self.bin_hight = np.array(bin_hight)

        self.bin_edge = np.array(bin_edge)

   



    def predict(self, X):

        

        scores = []

        for obs in X:

            obs_score = []

            for i, var in enumerate(obs):

                # find wich bin obs is in

                bin_num = (var > self.bin_edge[i]).argmin()-1

                obs_score.append(self.bin_hight[i, bin_num]) # find bin hitght

            

            scores.append(np.mean(obs_score))

        

        return np.array(scores)
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import Lasso, LassoLars
random_state = 42

debug = True

debug = False
svnu_params = {'probability':True, 'kernel':'poly','degree':4,'gamma':'auto','nu':0.4,'coef0':0.08, 'random_state':4}

svnu2_params = {'probability':True, 'kernel':'poly','degree':2,'gamma':'auto','nu':0.4,'coef0':0.08, 'random_state':4}

svc_params = {'probability':True,'kernel':'poly','degree':4,'gamma':'auto', 'random_state':4}

lr_params = {'solver':'liblinear','penalty':'l1','C':0.05,'n_jobs':-1, 'random_state':42}

mlp16_params = {'activation':'relu','solver':'lbfgs','tol':1e-06, 'hidden_layer_sizes':(16, ), 'random_state':42}

mlp128_params = {'activation':'relu','solver':'lbfgs','tol':1e-06, 'hidden_layer_sizes':(128, ), 'random_state':42}

gnb_params = {}
def get_oofs(random_state):

    oof_nusvc = np.zeros(len(train_df))

    preds_nusvc = np.zeros(len(test_df))



    oof_nusvc2 = np.zeros(len(train_df))

    preds_nusvc2 = np.zeros(len(test_df))



    oof_qda = np.zeros(len(train_df))

    preds_qda = np.zeros(len(test_df))



    oof_svc = np.zeros(len(train_df))

    preds_svc = np.zeros(len(test_df))

    

    oof_knn = np.zeros(len(train_df))

    preds_knn = np.zeros(len(test_df))

    

    oof_lr = np.zeros(len(train_df))

    preds_lr = np.zeros(len(test_df))

    

    oof_gnb = np.zeros(len(train_df))

    preds_gnb = np.zeros(len(test_df))

    

    cols = [c for c in train_df.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



    for i in tqdm_notebook(range(512)):



        # each magic

        train = train_df[train_df['wheezy-copper-turtle-magic'] == i]

        test = test_df[test_df['wheezy-copper-turtle-magic'] == i]



        # for oof

        train_idx_origin = train.index

        test_idx_origin = test.index





        # start point



        # new cols

        cols = magic_variance_over2[i]



        X_train = train.reset_index(drop=True)[cols].values

        y_train = train.reset_index(drop=True).target



        X_test = test[cols].values



        # vstack

        data = np.vstack([X_train, X_test])

        

        # PCA

        data = KernelPCA(n_components=len(cols), kernel='cosine', random_state=random_state).fit_transform(data)

        

        # Bad

        '''

        gmm_pred = np.zeros((len(data), 5))

        for j in range(5):

            gmm = GMM(n_components=4, random_state=random_state + j, max_iter=1000).fit(data)

            gmm_pred[:, j] += gmm.predict(data)

        '''

          

        # original

        gmm = GMM(n_components=5, random_state=random_state, max_iter=1000).fit(data)

        gmm_pred = gmm.predict_proba(data)

        gmm_score = gmm.score_samples(data)

        gmm_label = gmm.predict(data)

        

        hist = hist_model(); hist.fit(data)

        hist_pred = hist.predict(data).reshape(-1, 1)



        data = np.hstack([data, gmm_pred])



        # HOXI

        data = np.hstack([data, gmm_pred])

        data = np.hstack([data, gmm_pred])

        data = np.hstack([data, gmm_pred])

        

        # Add Some Features

        data = np.hstack([data, gmm_pred])

        data = np.hstack([data, hist_pred, gmm_score.reshape(-1, 1)])

        data = np.hstack([data, gmm_score.reshape(-1, 1)])

        data = np.hstack([data, gmm_score.reshape(-1, 1)])



        # STANDARD SCALER

        data = StandardScaler().fit_transform(data)



        # new train/test

        X_train = data[:X_train.shape[0]]

        X_test = data[X_train.shape[0]:]



        fold = StratifiedKFold(n_splits=5, random_state=random_state)

        for tr_idx, val_idx in fold.split(X_train, gmm_label[:X_train.shape[0]]):

            

            # NuSVC 1

            clf = svm.NuSVC(**svnu_params)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_nusvc[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_nusvc[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits



            # NuSVC 2

            clf = svm.NuSVC(**svnu2_params)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_nusvc2[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_nusvc2[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits





            # qda 3

            clf = QuadraticDiscriminantAnalysis(reg_param=0.111)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_qda[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_qda[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits



            # SVC 4

            clf = svm.SVC(**svc_params)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_svc[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_svc[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits

            

            # knn 8

            clf = KNeighborsClassifier(n_neighbors=16)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_knn[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_knn[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits   

            

            # LR 5

            clf = linear_model.LogisticRegression(**lr_params)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_lr[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_lr[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits

            

            # GNB

            #clf = GaussianNB(**gnb_params)

            #clf.fit(X_train[tr_idx], y_train[tr_idx])

            #oof_gnb[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            #preds_gnb[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits

            

    oof_train = pd.DataFrame()



    oof_train['nusvc'] = oof_nusvc

    oof_train['nusvc2'] = oof_nusvc2

    oof_train['qda'] = oof_qda

    oof_train['svc'] = oof_svc

    oof_train['knn'] = oof_knn

    oof_train['lr'] = oof_lr

    #oof_train['gnb'] = oof_gnb

    

    oof_test = pd.DataFrame()



    oof_test['nusvc'] = preds_nusvc

    oof_test['nusvc2'] = preds_nusvc2

    oof_test['qda'] = preds_qda

    oof_test['svc'] = preds_svc

    oof_test['knn'] = preds_knn

    oof_test['lr'] = preds_lr

    #oof_test['gnb'] = preds_gnb



    print('nusvc', roc_auc_score(train_df['target'], oof_nusvc))

    print('nusvc2', roc_auc_score(train_df['target'], oof_nusvc2))

    print('qda', roc_auc_score(train_df['target'], oof_qda))

    print('svc', roc_auc_score(train_df['target'], oof_svc))

    print('knn', roc_auc_score(train_df['target'], oof_knn))

    print('lr', roc_auc_score(train_df['target'], oof_lr))

    #print('gnb', roc_auc_score(train_df['target'], oof_gnb))

    

    return oof_train, oof_test
def get_oofs_2(random_state):

    oof_nusvc = np.zeros(len(train_df))

    preds_nusvc = np.zeros(len(test_df))



    oof_nusvc2 = np.zeros(len(train_df))

    preds_nusvc2 = np.zeros(len(test_df))



    oof_qda = np.zeros(len(train_df))

    preds_qda = np.zeros(len(test_df))



    oof_svc = np.zeros(len(train_df))

    preds_svc = np.zeros(len(test_df))

    

    oof_knn = np.zeros(len(train_df))

    preds_knn = np.zeros(len(test_df))



    oof_lr = np.zeros(len(train_df))

    preds_lr = np.zeros(len(test_df))

 

    oof_gnb = np.zeros(len(train_df))

    preds_gnb = np.zeros(len(test_df))

    

    cols = [c for c in train_df.columns if c not in ['id', 'target', 'wheezy-copper-turtle-magic']]



    for i in tqdm_notebook(range(512)):



        # each magic

        train = train_df[train_df['wheezy-copper-turtle-magic'] == i]

        test = test_df[test_df['wheezy-copper-turtle-magic'] == i]



        # for oof

        train_idx_origin = train.index

        test_idx_origin = test.index





        # start point



        # new cols

        cols = magic_variance_over2[i]



        X_train = train.reset_index(drop=True)[cols].values

        y_train = train.reset_index(drop=True).target



        X_test = test[cols].values



        # vstack

        data = np.vstack([X_train, X_test])



        # PCA

        data = KernelPCA(n_components=len(cols), kernel='cosine', random_state=random_state).fit_transform(data)



        # Bad

        '''

        gmm_pred = np.zeros((len(data), 5))

        for j in range(5):

            gmm = GMM(n_components=4, random_state=random_state + j, max_iter=1000).fit(data)

            gmm_pred[:, j] += gmm.predict(data)

        '''

            

        # original

        gmm = GMM(n_components=5, random_state=random_state, max_iter=1000, init_params='random').fit(data)

        gmm_pred = gmm.predict_proba(data)

        gmm_score = gmm.score_samples(data)

        gmm_label = gmm.predict(data)

        

        hist = hist_model(); hist.fit(data)

        hist_pred = hist.predict(data).reshape(-1, 1)



        data = np.hstack([data, gmm_pred])

        

        # HOXI

        data = np.hstack([data, gmm_pred])

        data = np.hstack([data, gmm_pred])

        data = np.hstack([data, gmm_pred])

        

        # Add Some Features

        data = np.hstack([data, gmm_pred])

        data = np.hstack([data, hist_pred, gmm_score.reshape(-1, 1)])

        data = np.hstack([data, gmm_score.reshape(-1, 1)])

        data = np.hstack([data, gmm_score.reshape(-1, 1)])



        # STANDARD SCALER

        data = StandardScaler().fit_transform(data)



        # new train/test

        X_train = data[:X_train.shape[0]]

        X_test = data[X_train.shape[0]:]



        fold = StratifiedKFold(n_splits=5, random_state=random_state)

        for tr_idx, val_idx in fold.split(X_train, gmm_label[:X_train.shape[0]]):

            

            # NuSVC 1

            clf = svm.NuSVC(**svnu_params)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_nusvc[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_nusvc[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits



            # NuSVC 2

            clf = svm.NuSVC(**svnu2_params)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_nusvc2[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_nusvc2[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits





            # qda 3

            clf = QuadraticDiscriminantAnalysis(reg_param=0.111)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_qda[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_qda[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits



            # SVC 4

            clf = svm.SVC(**svc_params)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_svc[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_svc[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits

            

            # knn 8

            clf = KNeighborsClassifier(n_neighbors=16)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_knn[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_knn[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits   

            

            # LR 5

            clf = linear_model.LogisticRegression(**lr_params)

            clf.fit(X_train[tr_idx], y_train[tr_idx])

            oof_lr[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            preds_lr[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits

            

            # GNB

            #clf = GaussianNB(**gnb_params)

            #clf.fit(X_train[tr_idx], y_train[tr_idx])

            #oof_gnb[train_idx_origin[val_idx]] = clf.predict_proba(X_train[val_idx])[:,1]

            #preds_gnb[test_idx_origin] += clf.predict_proba(X_test)[:,1] / fold.n_splits

            

    oof_train = pd.DataFrame()



    oof_train['nusvc'] = oof_nusvc

    oof_train['nusvc2'] = oof_nusvc2

    oof_train['qda'] = oof_qda

    oof_train['svc'] = oof_svc

    oof_train['knn'] = oof_knn

    oof_train['lr'] = oof_lr

    #oof_train['gnb'] = oof_gnb

    

    oof_test = pd.DataFrame()



    oof_test['nusvc'] = preds_nusvc

    oof_test['nusvc2'] = preds_nusvc2

    oof_test['qda'] = preds_qda

    oof_test['svc'] = preds_svc

    oof_test['knn'] = preds_knn

    oof_test['lr'] = preds_lr

    #oof_test['gnb'] = preds_gnb



    print('nusvc', roc_auc_score(train_df['target'], oof_nusvc))

    print('nusvc2', roc_auc_score(train_df['target'], oof_nusvc2))

    print('qda', roc_auc_score(train_df['target'], oof_qda))

    print('svc', roc_auc_score(train_df['target'], oof_svc))

    print('knn', roc_auc_score(train_df['target'], oof_knn))

    print('lr', roc_auc_score(train_df['target'], oof_lr))

    print('gnb', roc_auc_score(train_df['target'], oof_gnb))

    

    return oof_train, oof_test
oof_train_1, oof_test_1 = get_oofs(1)

oof_train_2, oof_test_2 = get_oofs(2)

oof_train_3, oof_test_3 = get_oofs_2(1)

oof_train_4, oof_test_4 = get_oofs_2(2)
x_train_second_layer = oof_train_1 + oof_train_2 + oof_train_3 + oof_train_4

x_test_second_layer = oof_test_1 + oof_test_2 + oof_test_3 + oof_test_4

print('Ensemble', roc_auc_score(train_df['target'], x_train_second_layer.mean(1)))
submit = pd.read_csv('../input/sample_submission.csv')

submit["target"] = x_test_second_layer.mean(1)

submit.to_csv("submission0.csv", index=False)
def time_decorator(func):

    

    @wraps(func)

    def wrapper(*args, **kwargs):

        print("\nStartTime: ", datetime.now() + timedelta(hours=9))

        start_time = time.time()

        

        df = func(*args, **kwargs)

        

        print("EndTime: ", datetime.now() + timedelta(hours=9))  

        print("TotalTime: ", time.time() - start_time)

        return df

        

    return wrapper



class SklearnWrapper(object):

    def __init__(self, clf, params=None, **kwargs):

        """

        params['random_state'] = kwargs.get('seed', 0)

        self.clf = clf(**params)

        self.is_classification_problem = True

        """

        if 'seed' in kwargs:

            params['random_state'] = kwargs.get('seed', 0)

        self.clf = clf(**params)

        self.is_classification_problem = True

    #@time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        if len(np.unique(y_train)) > 30:

            self.is_classification_problem = False

            

        self.clf.fit(x_train, y_train)



    def predict(self, x):

        if self.is_classification_problem is True:

            return self.clf.predict_proba(x)[:,1]

        else:

            return self.clf.predict(x)

    

class LgbmWrapper(object):

    def __init__(self, params=None, **kwargs):

        self.param = params

        if 'seed' in kwargs:

            self.param['seed'] = kwargs.get('seed', 0)

        self.num_rounds = kwargs.get('num_rounds', 1000)

        self.early_stopping = kwargs.get('ealry_stopping', 100)



        self.eval_function = kwargs.get('eval_function', None)

        self.verbose_eval = kwargs.get('verbose_eval', 100)

        self.best_round = 0

        self.feature_importance = pd.DataFrame()

        

    #@time_decorator

    def train(self, x_train, y_train, x_cross=None, y_cross=None):

        """

        x_cross or y_cross is None

        -> model train limted num_rounds

        

        x_cross and y_cross is Not None

        -> model train using validation set

        """

        if isinstance(y_train, pd.DataFrame) is True:

            y_train = y_train[y_train.columns[0]]

            if y_cross is not None:

                y_cross = y_cross[y_cross.columns[0]]



        if x_cross is None:

            dtrain = lgb.Dataset(x_train, label=y_train, silent= True)

            train_round = self.best_round

            if self.best_round == 0:

                train_round = self.num_rounds

                

            self.clf = lgb.train(self.param, train_set=dtrain, num_boost_round=train_round)

            del dtrain   

        else:

            dtrain = lgb.Dataset(x_train, label=y_train, silent=True)

            dvalid = lgb.Dataset(x_cross, label=y_cross, silent=True)

            self.clf = lgb.train(self.param, train_set=dtrain, num_boost_round=self.num_rounds, valid_sets=[dtrain, dvalid],

                                  feval=self.eval_function, early_stopping_rounds=self.early_stopping,

                                  verbose_eval=self.verbose_eval)

            

            try:

                self.feature_importance = pd.DataFrame()

                self.feature_importance["Feature"] = x_train.columns

                self.feature_importance["Importance"] = self.clf.feature_importance()

            except:

                pass

            

            self.best_round = max(self.best_round, self.clf.best_iteration)

            del dtrain, dvalid

            

        gc.collect()

    

    def get_importance_df(self):

        return self.feature_importance

    

    def predict(self, x):

        return self.clf.predict(x, num_iteration=self.clf.best_iteration)

    

    def plot_importance(self):

        lgb.plot_importance(self.clf, max_num_features=50, height=0.7, figsize=(10,30))

        plt.show()

        

    def get_params(self):

        return self.param
#@time_decorator

def get_oof(clf, x_train, y_train, x_test, eval_func, **kwargs):

    nfolds = kwargs.get('NFOLDS', 5)

    kfold_shuffle = kwargs.get('kfold_shuffle', True)

    kfold_random_state = kwargs.get('kfold_random_state', 0)

    stratified_kfold_ytrain = kwargs.get('stratifed_kfold_y_value', None)

    inner_predict = kwargs.get('inner_predict', True)

    export_feature_importance = kwargs.get('export_feature_importance', True)

    ntrain = x_train.shape[0]

    ntest = x_test.shape[0]

    

    kf_split = None

    if stratified_kfold_ytrain is None:

        kf = KFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)

        kf_split = kf.split(x_train)

    else:

        kf = StratifiedKFold(n_splits=nfolds, shuffle=kfold_shuffle, random_state=kfold_random_state)

        kf_split = kf.split(x_train, stratified_kfold_ytrain)

        

    oof_train = np.zeros((ntrain,))

    oof_test = np.zeros((ntest,))



    cv_sum = 0

    

    # before running model, print model param

    # lightgbm model and xgboost model use get_params()

    """

    try:

        if clf.clf is not None:

            print(clf.clf)

    except:

        print(clf)

        print(clf.get_params())

    """

    feature_importance_df = pd.DataFrame()

    for i, (train_index, cross_index) in enumerate(kf_split):

        x_tr, x_cr = None, None

        y_tr, y_cr = None, None

        if isinstance(x_train, pd.DataFrame):

            x_tr, x_cr = x_train.iloc[train_index], x_train.iloc[cross_index]

            y_tr, y_cr = y_train.iloc[train_index], y_train.iloc[cross_index]

        else:

            x_tr, x_cr = x_train[train_index], x_train[cross_index]

            y_tr, y_cr = y_train[train_index], y_train[cross_index]



        clf.train(x_tr, y_tr, x_cr, y_cr)

        

        if isinstance(clf, LgbmWrapper) is True:

            feature_importance_df = pd.concat([feature_importance_df, clf.get_importance_df()], axis=0)

    

        oof_train[cross_index] = clf.predict(x_cr)

        if inner_predict is True:

            oof_test += clf.predict(x_test)

        

        cv_score = eval_func(y_cr, oof_train[cross_index])

        

        #print('Fold %d / ' % (i+1), 'CV-Score: %.6f' % cv_score)

        cv_sum = cv_sum + cv_score

        

        del x_tr, x_cr, y_tr, y_cr

        

    gc.collect()

    

    score = cv_sum / nfolds

    #print("Average CV-Score: ", score)

    #print("OOF CV-Score: ", eval_func(y_train, oof_train))

    

    if export_feature_importance is True:

        print("Export Feature Importance")

        filename = '{}_cv{:.6f}'.format(datetime.now().strftime('%Y%m%d_%H%M%S'), score)

        if os.path.isdir("importance/") is True:

            feature_importance_df.to_csv('importance/importance_{}.csv'.format(filename),index=False)

        else:

            feature_importance_df.to_csv('importance_{}.csv'.format(filename),index=False)

            

    if inner_predict is True:

        oof_test = oof_test/nfolds

    else:

        # Using All Dataset, retrain

        clf.train(x_train, y_train)

        oof_test = clf.predict(x_test)



    return oof_train, oof_test, score
x_train_second_layer1 = pd.DataFrame(x_train_second_layer)

x_test_second_layer1 = pd.DataFrame(x_test_second_layer)
import time

from datetime import datetime, timedelta,date

import warnings

import itertools

from functools import wraps

import functools



import seaborn as sns

import matplotlib.pyplot as plt



import lightgbm as lgb

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import StratifiedKFold, KFold

from sklearn import preprocessing

from sklearn import svm, neighbors, linear_model

import gc

warnings.filterwarnings('ignore')



from sklearn.mixture import GaussianMixture as GMM
SEED = 0



param = {

        #'bagging_freq': 5,

        #'bagging_fraction': 0.8,

        'min_child_weight':6.790,

        "subsample_for_bin":50000,

        'bagging_seed': 0,

        'boost_from_average':'true',

        'boost': 'gbdt',

        'feature_fraction': 0.450,

        'bagging_fraction': 0.343,

        'learning_rate': 0.025,

        'max_depth': 10,

        'metric':'auc',

        'min_data_in_leaf': 78,

        'min_sum_hessian_in_leaf': 8, 

        'num_leaves': 18,

        'num_threads': 8,

        'tree_learner': 'serial',

        'objective': 'binary', 

        'verbosity': 1,

        'lambda_l1': 7.961,

        'lambda_l2': 7.781

        #'reg_lambda': 0.3,

    }



mlp16_params = {'activation':'relu','solver':'lbfgs','tol':1e-06, 'hidden_layer_sizes':(16, ), 'random_state':42}

knn_params ={'n_neighbors':17, 'p':2.9,'n_jobs':-1}



lgbm_meta_model = LgbmWrapper(params=param, num_rounds = 2000, ealry_stopping=100)

mlp_meta_model = SklearnWrapper(neural_network.MLPClassifier,mlp16_params)

knn_meta_model = SklearnWrapper(neighbors.KNeighborsClassifier,knn_params)





third_number = 4

oof_train_5 = pd.DataFrame()

oof_test_5 = pd.DataFrame()



# lgbm

third_oof = np.zeros(len(train_df))

third_pred = np.zeros(len(test_df))



# mlp

third_oof1 = np.zeros(len(train_df))

third_pred1 = np.zeros(len(test_df))



# knn

third_oof2 = np.zeros(len(train_df))

third_pred2 = np.zeros(len(test_df))



third_oof3 = np.zeros(len(train_df))

third_pred3 = np.zeros(len(test_df))



#for SEED in np.arange(third_number):

second_oof, second_pred, second_score = get_oof(lgbm_meta_model, x_train_second_layer1, train_df['target'], x_test_second_layer1, eval_func=roc_auc_score, NFOLDS=5, kfold_random_sate= SEED )

second_oof1, second_pred1, second_score1 = get_oof(mlp_meta_model, x_train_second_layer1, train_df['target'], x_test_second_layer1, eval_func=roc_auc_score, NFOLDS=5, kfold_random_sate= SEED )

second_oof2, second_pred2, second_score2 = get_oof(knn_meta_model, x_train_second_layer1, train_df['target'], x_test_second_layer1, eval_func=roc_auc_score, NFOLDS=5, kfold_random_sate= SEED )



third_oof += second_oof

third_pred += second_pred

print(second_score)

third_oof1 += second_oof1

third_pred1 += second_pred1

print(second_score1)

third_oof2 += second_oof2

third_pred2 += second_pred2

print(second_score2)

print("")

    

oof_train_5['lgb'] = third_oof * 4

oof_test_5['lgb'] = third_pred * 4

oof_train_5['mlp'] = third_oof1 * 4

oof_test_5['mlp'] = third_pred1 * 4

oof_train_5['knn'] = third_oof2 * 4

oof_test_5['knn'] = third_pred2 * 4
x_train_second_layer = oof_train_1 + oof_train_2 + oof_train_3 + oof_train_4  

x_test_second_layer = oof_test_1 + oof_test_2 + oof_test_3 + oof_test_4 



x_train_second_layer = pd.concat([x_train_second_layer,oof_train_5],axis=1)

x_test_second_layer = pd.concat([x_test_second_layer,oof_test_5],axis=1)

                                     

print('Ensemble', roc_auc_score(train_df['target'], x_train_second_layer.mean(1)))
x_train_second_layer.corr()
submit = pd.read_csv('../input/sample_submission.csv')

submit["target"] = x_test_second_layer.mean(1)

submit.to_csv("submission.csv", index=False)
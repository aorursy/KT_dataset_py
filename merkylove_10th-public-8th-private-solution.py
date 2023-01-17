import numpy as np, pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split

from sklearn.metrics import roc_auc_score

from sklearn import svm, neighbors, linear_model, neural_network

from sklearn.svm import NuSVC

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

from sklearn.decomposition import PCA

from tqdm import tqdm_notebook

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_auc_score

from sklearn.feature_selection import VarianceThreshold

from sklearn.preprocessing import OneHotEncoder

from sklearn import mixture

from scipy.stats.mstats import gmean

from bayes_opt import BayesianOptimization

from sklearn.naive_bayes import GaussianNB

import pickle

import lightgbm as lgb

from sklearn.ensemble import RandomForestClassifier

from sklearn.semi_supervised import LabelPropagation



from sklearn.covariance import GraphicalLasso



import warnings

warnings.filterwarnings('ignore')



train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
RANDOM_SEED = 4123



cols = [

    c for c in train.columns 

    if c not in ['id', 'target', 'wheezy-copper-turtle-magic']

]



def get_mean_cov(x,y):

    model = GraphicalLasso()

    ones = (y==1).astype(bool)

    x2 = x[ones]

    model.fit(x2)

    p1 = model.precision_

    m1 = model.location_

    

    onesb = (y==0).astype(bool)

    x2b = x[onesb]

    model.fit(x2b)

    p2 = model.precision_

    m2 = model.location_

    

    ms = np.stack([m1,m2])

    ps = np.stack([p1,p2])

    return ms,ps
%%time



SKIP_COMMIT = True



if SKIP_COMMIT:

    

    # to not waste time on commit

    

    sub = pd.read_csv('../input/sample_submission.csv')



    if sub.shape[0] == 131073:

        sub = pd.read_csv('../input/sample_submission.csv')

        sub.to_csv('submission.csv', index=False)



        raise ValueError('Stop!!!')





oof_nusvc = np.zeros(len(train)) 

preds_nusvc = np.zeros(len(test))



oof_nb= np.zeros(len(train)) 

preds_nb = np.zeros(len(test))



oof_lr = np.zeros(len(train)) 

preds_lr = np.zeros(len(test))



oof_qda = np.zeros(len(train)) 

preds_qda = np.zeros(len(test))



oof_lp = np.zeros(len(train))

preds_lp = np.zeros(len(test))



oof_lgbm = np.zeros(len(train)) 

preds_lgbm = np.zeros(len(test))



oof_gm = np.zeros(len(train)) 

preds_gm = np.zeros(len(test))



oof_rf = np.zeros(len(train)) 

preds_rf = np.zeros(len(test))





params_lgbm_1 = {

    'boosting_type': 'goss',

    'objective': 'xentropy',

    'metric': ['auc'],

    'num_leaves': 31,

    'learning_rate': 0.1212,

    'feature_fraction': 0.4138,

    'bagging_fraction': 0.2317,

    'num_threads': -1,

    'lambda_l2': 6.221,

    'max_bin': 29

}





for i in range(512):

    

    print(i, end=' ')

    

    train2 = train[train['wheezy-copper-turtle-magic']==i] 

    idx1 = train2.index 

    test2 = test[test['wheezy-copper-turtle-magic']==i]

    idx2 = test2.index

    

    data = pd.concat(

        [

            train2,

            test2

        ],

        axis=0

    )

    train2.reset_index(drop=True, inplace=True)

    

    train_size = train2.shape[0]

    

    

    # remove unnecessary fields

    sel = VarianceThreshold(threshold=1.5)

    tmp = sel.fit_transform(

        data[cols]

    )

    train3 = tmp[:train_size, :]

    test3 = tmp[train_size:, :]

    

    # scale data for non-QDA methods

    ss = StandardScaler()

    tmp_scaled = ss.fit_transform(tmp)

    

    train3_scaled = tmp_scaled[:train_size, :]

    test3_scaled = tmp_scaled[train_size:, :]

    

    # Polynomial features for LogReg

    poly = PolynomialFeatures(degree=2)

    tmp_poly = poly.fit_transform(tmp_scaled)

    

    train3_poly = tmp_poly[:train_size, :]

    test3_poly = tmp_poly[train_size:, :]

    

    # GM features 4

    gm_clf_4 = mixture.GaussianMixture(

        n_components=4, 

        random_state=RANDOM_SEED

    )

    #gm_tmp_4 = gm_clf_4.fit_predict(tmp).reshape(-1, 1)

    

    gm_clf_4.fit(tmp)

    #gm_tmp_6 = gm_clf_6.fit_predict(tmp).reshape(-1, 1)

    gm_tmp_4 = gm_clf_4.predict_proba(tmp)

    

    #le_4 = OneHotEncoder()

    #gm_tmp_4 = le_4.fit_transform(gm_tmp_4).todense()

    

    gm_train3_4 = gm_tmp_4[:train_size, :]

    gm_test3_4 = gm_tmp_4[train_size:, :]

    

    # GM features 6

    gm_clf_6 = mixture.GaussianMixture(

        n_components=6, 

        random_state=RANDOM_SEED

    )

    gm_clf_6.fit(tmp)

    #gm_tmp_6 = gm_clf_6.fit_predict(tmp).reshape(-1, 1)

    gm_tmp_6 = gm_clf_6.predict_proba(tmp)

    

    #le_6 = OneHotEncoder()

    #gm_tmp_6 = le_6.fit_transform(gm_tmp_6).todense()

    

    gm_train3_6 = gm_tmp_6[:train_size, :]

    gm_test3_6 = gm_tmp_6[train_size:, :]

    

    

    skf = StratifiedKFold(

        n_splits=11, 

        random_state=RANDOM_SEED, 

        shuffle=True

    )

    

    for train_index, test_index in skf.split(train3, train2['target']):

        

        train_train_index, train_val_index = train_test_split(

            train_index, 

            test_size=0.3, 

            random_state=RANDOM_SEED

        )

        

        # LGBM

        

        train_dataset = lgb.Dataset(

            np.hstack(

                (

                    train3_scaled[train_train_index,:], 

                    gm_tmp_4[:train_size, :][train_train_index, :].tolist(),

                    gm_tmp_6[:train_size, :][train_train_index, :].tolist(),

                    train3_poly[train_train_index,:]

                )

            ),

            train2.loc[train_train_index]['target'],

            free_raw_data=False

        )

        

        valid_dataset = lgb.Dataset(

            np.hstack(

                (

                    train3_scaled[train_val_index,:],

                    gm_tmp_4[:train_size, :][train_val_index, :].tolist(),

                    gm_tmp_6[:train_size, :][train_val_index, :].tolist(),

                    train3_poly[train_val_index,:]

                )

            ),

            train2.loc[train_val_index]['target'],

            free_raw_data=False

        )

        

        gm = lgb.train(

            params_lgbm_1,

            train_dataset,

            num_boost_round=5000,

            early_stopping_rounds=100,

            valid_sets=(train_dataset, valid_dataset),

            valid_names=('train', 'valid'),

            verbose_eval=0

        )

        

        oof_lgbm[idx1[test_index]] = gm.predict(

            np.hstack(

                (

                    train3_scaled[test_index,:],

                    gm_tmp_4[:train_size, :][test_index, :].tolist(),

                    gm_tmp_6[:train_size, :][test_index, :].tolist(),

                    train3_poly[test_index,:]

                )

            )

        )

        preds_lgbm[idx2] += gm.predict(

            np.hstack(

                (

                    test3_scaled,

                    gm_test3_4.tolist(),

                    gm_test3_6.tolist(),

                    test3_poly

                )

            )

        ) / skf.n_splits

        

        # GMM

        

        ms, ps = get_mean_cov(

            train3[train_index, :],

            train2.loc[train_index]['target'].values

        )

        

        gm = mixture.GaussianMixture(

            n_components=2, 

            init_params='random', 

            covariance_type='full', 

            tol=0.001,

            reg_covar=0.001,

            max_iter=100,

            n_init=1,

            means_init=ms,

            precisions_init=ps,

            random_state=RANDOM_SEED

        )

        gm.fit(tmp)

        oof_gm[idx1[test_index]] = gm.predict_proba(

            train3[test_index,:]

        )[:, 0]

        preds_gm[idx2] += gm.predict_proba(

            test3

        )[:, 0] / skf.n_splits

        

        # LabelProp



        lp = LabelPropagation(

            kernel='rbf', 

            gamma=0.15301581563198507, 

            n_jobs=-1

        )

        lp.fit(

            train3_scaled[train_index,:],

            train2.loc[train_index]['target']

        )

        oof_lp[idx1[test_index]] = lp.predict_proba(

            train3_scaled[test_index, :]

        )[:,1]

        preds_lp[idx2] += lp.predict_proba(

            test3_scaled

        )[:,1] / skf.n_splits

        

        # nuSVC

        

        clf = NuSVC(

            probability=True, 

            kernel='poly', 

            degree=2,

            gamma='auto', 

            random_state=RANDOM_SEED, 

            nu=0.27312143533915767, 

            coef0=0.4690615598786931

        )

        

        clf.fit(

            np.hstack(

                (

                    train3_scaled[train_index,:], 

                    gm_train3_4[train_index, :],

                    gm_train3_6[train_index, :]

                )

            ),

            train2.loc[train_index]['target']

        )

        oof_nusvc[idx1[test_index]] = clf.predict_proba(

            np.hstack(

                (

                    train3_scaled[test_index,:],

                    gm_train3_4[test_index, :],

                    gm_train3_6[test_index, :]

                )

            )

        )[:,1]

        

        preds_nusvc[idx2] += clf.predict_proba(

            np.hstack(

                (

                    test3_scaled, 

                    gm_test3_4,

                    gm_test3_6

                )

            )

        )[:,1] / skf.n_splits

        

        # RF

        

        clf = RandomForestClassifier(

            max_depth=4, 

            n_jobs=-1, 

            n_estimators=20,

            random_state=RANDOM_SEED

        )

        

        clf.fit(

            np.hstack(

                (

                    train3_scaled[train_index,:], 

                    gm_train3_4[train_index, :],

                    gm_train3_6[train_index, :]

                )

            ),

            train2.loc[train_index]['target']

        )

        oof_rf[idx1[test_index]] = clf.predict_proba(

            np.hstack(

                (

                    train3_scaled[test_index,:],

                    gm_train3_4[test_index, :],

                    gm_train3_6[test_index, :]

                )

            )

        )[:,1]

        

        preds_rf[idx2] += clf.predict_proba(

            np.hstack(

                (

                    test3_scaled, 

                    gm_test3_4,

                    gm_test3_6

                )

            )

        )[:,1] / skf.n_splits



        # QDA

        clf = QuadraticDiscriminantAnalysis(

            reg_param=0.5674164995882528

        )

        clf.fit(

            train3[train_index,:],

            train2.loc[train_index]['target']

        )

        oof_qda[idx1[test_index]] += clf.predict_proba(

            train3[test_index, :]

        )[:,1]

        preds_qda[idx2] += clf.predict_proba(

            test3

        )[:,1] / skf.n_splits

        

        # LogReg Poly

        

        clf = linear_model.LogisticRegression(

            solver='saga',

            penalty='l2',

            C=0.01,

            tol=0.001,

            random_state=RANDOM_SEED

        )

        clf.fit(

            train3_poly[train_index,:],

            train2.loc[train_index]['target']

        )

        oof_lr[idx1[test_index]] = clf.predict_proba(

            train3_poly[test_index,:]

        )[:,1]

        preds_lr[idx2] += clf.predict_proba(

            test3_poly

        )[:,1] / skf.n_splits

        

        # GaussianNB with GM 6

        

        clf = GaussianNB()

        clf.fit(

            np.hstack(

                (

                    train3_scaled[train_index,:], 

                    gm_train3_6[train_index, :],

                    gm_train3_4[train_index, :]

                )

            ),

            train2.loc[train_index]['target']

        )

        oof_nb[idx1[test_index]] = clf.predict_proba(

            np.hstack(

                (

                    train3_scaled[test_index,:],

                    gm_train3_6[test_index, :],

                    gm_train3_4[test_index, :]

                )

            )

        )[:,1]

        

        preds_nb[idx2] += clf.predict_proba(

            np.hstack(

                (

                    test3_scaled,

                    gm_test3_6,

                    gm_test3_4

                )

            )

        )[:,1] / skf.n_splits



        

print('\nsvcnu', roc_auc_score(train['target'], oof_nusvc))

print('gm', roc_auc_score(train['target'], oof_gm))

print('qda', roc_auc_score(train['target'], oof_qda))

print('log reg poly', roc_auc_score(train['target'], oof_lr))

print('gnb', roc_auc_score(train['target'], oof_nb))

print('lp', roc_auc_score(train['target'], oof_lp))

print('lgbm', roc_auc_score(train['target'], oof_lgbm))

print('rf', roc_auc_score(train['target'], oof_rf))



oof_qda = oof_qda.reshape(-1, 1)

preds_qda = preds_qda.reshape(-1, 1)



oof_lr = oof_lr.reshape(-1, 1)

preds_lr = preds_lr.reshape(-1, 1)



oof_nusvc = oof_nusvc.reshape(-1, 1)

preds_nusvc = preds_nusvc.reshape(-1, 1)



oof_nb = oof_nb.reshape(-1, 1)

preds_nb = preds_nb.reshape(-1, 1)



oof_lp = oof_lp.reshape(-1, 1)

preds_lp = preds_lp.reshape(-1, 1)



oof_gm = oof_gm.reshape(-1, 1)

preds_gm = preds_gm.reshape(-1, 1)



oof_lgbm = oof_lgbm.reshape(-1, 1)

preds_lgbm = preds_lgbm.reshape(-1, 1)



oof_rf = oof_rf.reshape(-1, 1)

preds_rf = preds_rf.reshape(-1, 1)



tr_2 = np.concatenate(

    (

        oof_qda,

        oof_nusvc,

        oof_lr,

        oof_nb,

        oof_lp,

        oof_gm,

        oof_lgbm,

        oof_rf

    ), 

    axis=1

)

te_2 = np.concatenate(

    (

        preds_qda, 

        preds_nusvc, 

        preds_lr, 

        preds_nb,

        preds_lp,

        preds_gm,

        preds_lgbm,

        preds_rf

    ), 

    axis=1

)



print(np.corrcoef(tr_2, rowvar=False))



params = {

        'boosting_type': 'goss',

        'objective': 'xentropy',

        'metric': ['auc'],

        'num_leaves': 3,

        'learning_rate': 0.2,

        'feature_fraction': 0.4,

        'bagging_fraction': 0.4,

        #'bagging_freq': 5,

        'num_threads': -1

    }





params.update(

    {

        'bagging_fraction': 0.4, 

        'feature_fraction': 0.4, 

        'lambda_l2': 5.0, 

        'learning_rate': 0.2, 

        'max_bin': 10, 

        'num_leaves': 7

    }

)



POWER_1 = 0.5

POWER_2 = 1.4



oof_boosting_2_bad_cv = np.zeros(train.shape[0])

pred_te_boosting_2_bad_cv = np.zeros(test.shape[0])





train2 = train.copy()

train2.reset_index(drop=True,inplace=True)



skf = StratifiedKFold(

    n_splits=11, 

    random_state=RANDOM_SEED, 

    shuffle=True

)



for train_index, test_index in skf.split(tr_2, train2['target']):

    

    train_dataset = lgb.Dataset(

        np.hstack(

            (

                tr_2[train_index, :],

                tr_2[train_index, :]  ** POWER_1,

                tr_2[train_index, :]  ** POWER_2

            )

        ),

        train2['target'][train_index],

        free_raw_data=False

    )

    valid_dataset = lgb.Dataset(

        np.hstack(

            (

                tr_2[test_index, :] ,

                tr_2[test_index, :]  ** POWER_1,

                tr_2[test_index, :]  ** POWER_2

            )

        ),

        train2['target'][test_index],

        free_raw_data=False

    )



    gbm = lgb.train(

        params,

        train_dataset,

        num_boost_round=1000,

        early_stopping_rounds=100,

        valid_sets=(train_dataset, valid_dataset),

        valid_names=('train', 'valid'),

        verbose_eval=100

    )



    oof_boosting_2_bad_cv[test_index] = gbm.predict(

        np.hstack(

            (

                tr_2[test_index, :],

                tr_2[test_index, :] ** POWER_1,

                tr_2[test_index, :] ** POWER_2

            )

        )

    )

    

    pred_te_boosting_2_bad_cv += gbm.predict(

        np.hstack(

            (

                te_2,

                te_2 ** POWER_1,

                te_2 ** POWER_2

            )

        )

    ) / skf.n_splits



    

print('gnb', roc_auc_score(train['target'], oof_boosting_2_bad_cv))

##############

# SAVE RESULTS

##############

sub = pd.read_csv('../input/sample_submission.csv')

sub['target'] = pred_te_boosting_2_bad_cv

sub.to_csv('submission.csv', index=False)

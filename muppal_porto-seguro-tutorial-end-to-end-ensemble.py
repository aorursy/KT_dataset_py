# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import roc_auc_score



from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import StratifiedKFold



import xgboost as xgb

import lightgbm as lgb

import time



pd.set_option('display.max_columns', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.max_rows', 1000)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train=pd.read_csv('../input/train.csv')

test=pd.read_csv('../input/test.csv')

sample_submission=pd.read_csv('../input/sample_submission.csv')
# This function late in a list of features 'cols' from train and test dataset, 

# and performing frequency encoding. 

def freq_encoding(cols, train_df, test_df):

    # we are going to store our new dataset in these two resulting datasets

    result_train_df=pd.DataFrame()

    result_test_df=pd.DataFrame()

    

    # loop through each feature column to do this

    for col in cols:

        

        # capture the frequency of a feature in the training set in the form of a dataframe

        col_freq=col+'_freq'

        freq=train_df[col].value_counts()

        freq=pd.DataFrame(freq)

        freq.reset_index(inplace=True)

        freq.columns=[[col,col_freq]]



        # merge ths 'freq' datafarme with the train data

        temp_train_df=pd.merge(train_df[[col]], freq, how='left', on=col)

        temp_train_df.drop([col], axis=1, inplace=True)



        # merge this 'freq' dataframe with the test data

        temp_test_df=pd.merge(test_df[[col]], freq, how='left', on=col)

        temp_test_df.drop([col], axis=1, inplace=True)



        # if certain levels in the test dataset is not observed in the train dataset, 

        # we assign frequency of zero to them

        temp_test_df.fillna(0, inplace=True)

        temp_test_df[col_freq]=temp_test_df[col_freq].astype(np.int32)



        if result_train_df.shape[0]==0:

            result_train_df=temp_train_df

            result_test_df=temp_test_df

        else:

            result_train_df=pd.concat([result_train_df, temp_train_df],axis=1)

            result_test_df=pd.concat([result_test_df, temp_test_df],axis=1)

    

    return result_train_df, result_test_df
cat_cols=['ps_ind_02_cat','ps_car_04_cat', 'ps_car_09_cat',

          'ps_ind_05_cat', 'ps_car_01_cat', 'ps_car_11_cat']



# generate dataframe for frequency features for the train and test dataset

train_freq, test_freq=freq_encoding(cat_cols, train, test)



# merge them into the original train and test dataset

train=pd.concat([train, train_freq], axis=1)

test=pd.concat([test,test_freq], axis=1)
# perform binary encoding for categorical variable

# this function take in a pair of train and test data set, and the feature that need to be encode.

# it returns the two dataset with input feature encoded in binary representation

# this function assumpt that the feature to be encoded is already been encoded in a numeric manner 

# ranging from 0 to n-1 (n = number of levels in the feature). 



def binary_encoding(train_df, test_df, feat):

    # calculate the highest numerical value used for numeric encoding

    train_feat_max = train_df[feat].max()

    test_feat_max = test_df[feat].max()

    if train_feat_max > test_feat_max:

        feat_max = train_feat_max

    else:

        feat_max = test_feat_max

        

    # use the value of feat_max+1 to represent missing value

    train_df.loc[train_df[feat] == -1, feat] = feat_max + 1

    test_df.loc[test_df[feat] == -1, feat] = feat_max + 1

    

    # create a union set of all possible values of the feature

    union_val = np.union1d(train_df[feat].unique(), test_df[feat].unique())



    # extract the highest value from from the feature in decimal format.

    max_dec = union_val.max()

    

    # work out how the ammount of digtis required to be represent max_dev in binary representation

    max_bin_len = len("{0:b}".format(max_dec))

    index = np.arange(len(union_val))

    columns = list([feat])

    

    # create a binary encoding feature dataframe to capture all the levels for the feature

    bin_df = pd.DataFrame(index=index, columns=columns)

    bin_df[feat] = union_val

    

    # capture the binary representation for each level of the feature 

    feat_bin = bin_df[feat].apply(lambda x: "{0:b}".format(x).zfill(max_bin_len))

    

    # split the binary representation into different bit of digits 

    splitted = feat_bin.apply(lambda x: pd.Series(list(x)).astype(np.uint8))

    splitted.columns = [feat + '_bin_' + str(x) for x in splitted.columns]

    bin_df = bin_df.join(splitted)

    

    # merge the binary feature encoding dataframe with the train and test dataset - Done! 

    train_df = pd.merge(train_df, bin_df, how='left', on=[feat])

    test_df = pd.merge(test_df, bin_df, how='left', on=[feat])

    return train_df, test_df

cat_cols=['ps_ind_02_cat','ps_car_04_cat', 'ps_car_09_cat',

          'ps_ind_05_cat', 'ps_car_01_cat']



train, test=binary_encoding(train, test, 'ps_ind_02_cat')

train, test=binary_encoding(train, test, 'ps_car_04_cat')

train, test=binary_encoding(train, test, 'ps_car_09_cat')

train, test=binary_encoding(train, test, 'ps_ind_05_cat')

train, test=binary_encoding(train, test, 'ps_car_01_cat')
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]

train.drop(col_to_drop, axis=1, inplace=True)  

test.drop(col_to_drop, axis=1, inplace=True)  
col_to_drop = train.columns[train.columns.str.startswith('ps_calc_')]

train.drop(col_to_drop, axis=1, inplace=True)  

test.drop(col_to_drop, axis=1, inplace=True)
train.head(5)
def auc_to_gini_norm(auc_score):

    return 2*auc_score-1
def cross_validate_sklearn(clf, x_train, y_train , x_test, kf,scale=False, verbose=True):

    start_time=time.time()

    

    # initialise the size of out-of-fold train an test prediction

    train_pred = np.zeros((x_train.shape[0]))

    test_pred = np.zeros((x_test.shape[0]))



    # use the kfold object to generate the required folds

    for i, (train_index, test_index) in enumerate(kf.split(x_train, y_train)):

        # generate training folds and validation fold

        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[test_index, :]

        y_train_kf, y_val_kf = y_train[train_index], y_train[test_index]



        # perform scaling if required i.e. for linear algorithms

        if scale:

            scaler = StandardScaler().fit(x_train_kf.values)

            x_train_kf_values = scaler.transform(x_train_kf.values)

            x_val_kf_values = scaler.transform(x_val_kf.values)

            x_test_values = scaler.transform(x_test.values)

        else:

            x_train_kf_values = x_train_kf.values

            x_val_kf_values = x_val_kf.values

            x_test_values = x_test.values

        

        # fit the input classifier and perform prediction.

        clf.fit(x_train_kf_values, y_train_kf.values)

        val_pred=clf.predict_proba(x_val_kf_values)[:,1]

        train_pred[test_index] += val_pred



        y_test_preds = clf.predict_proba(x_test_values)[:,1]

        test_pred += y_test_preds



        fold_auc = roc_auc_score(y_val_kf.values, val_pred)

        fold_gini_norm = auc_to_gini_norm(fold_auc)



        if verbose:

            print('fold cv {} AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(i, fold_auc, fold_gini_norm))



    test_pred /= kf.n_splits



    cv_auc = roc_auc_score(y_train, train_pred)

    cv_gini_norm = auc_to_gini_norm(cv_auc)

    cv_score = [cv_auc, cv_gini_norm]

    if verbose:

        print('cv AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(cv_auc, cv_gini_norm))

        end_time = time.time()

        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))

    return cv_score, train_pred,test_pred

def probability_to_rank(prediction, scaler=1):

    pred_df=pd.DataFrame(columns=['probability'])

    pred_df['probability']=prediction

    pred_df['rank']=pred_df['probability'].rank()/len(prediction)*scaler

    return pred_df['rank'].values
def cross_validate_xgb(params, x_train, y_train, x_test, kf, cat_cols=[], verbose=True, 

                       verbose_eval=50, num_boost_round=4000, use_rank=True):

    start_time=time.time()



    train_pred = np.zeros((x_train.shape[0]))

    test_pred = np.zeros((x_test.shape[0]))



    # use the k-fold object to enumerate indexes for each training and validation fold

    for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train)): # folds 1, 2 ,3 ,4, 5

        # example: training from 1,2,3,4; validation from 5

        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[val_index, :]

        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]

        x_test_kf=x_test.copy()



        d_train_kf = xgb.DMatrix(x_train_kf, label=y_train_kf)

        d_val_kf = xgb.DMatrix(x_val_kf, label=y_val_kf)

        d_test = xgb.DMatrix(x_test_kf)



        bst = xgb.train(params, d_train_kf, num_boost_round=num_boost_round,

                        evals=[(d_train_kf, 'train'), (d_val_kf, 'val')], verbose_eval=verbose_eval,

                        early_stopping_rounds=50)



        val_pred = bst.predict(d_val_kf, ntree_limit=bst.best_ntree_limit)

        if use_rank:

            train_pred[val_index] += probability_to_rank(val_pred)

            test_pred+=probability_to_rank(bst.predict(d_test))

        else:

            train_pred[val_index] += val_pred

            test_pred+=bst.predict(d_test)



        fold_auc = roc_auc_score(y_val_kf.values, val_pred)

        fold_gini_norm = auc_to_gini_norm(fold_auc)



        if verbose:

            print('fold cv {} AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(i, fold_auc, 

                                                                                     fold_gini_norm))



    test_pred /= kf.n_splits



    cv_auc = roc_auc_score(y_train, train_pred)

    cv_gini_norm = auc_to_gini_norm(cv_auc)

    cv_score = [cv_auc, cv_gini_norm]

    if verbose:

        print('cv AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(cv_auc, cv_gini_norm))

        end_time = time.time()

        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))



        return cv_score, train_pred,test_pred

def cross_validate_lgb(params, x_train, y_train, x_test, kf, cat_cols=[],

                       verbose=True, verbose_eval=50, use_cat=True, use_rank=True):

    start_time = time.time()

    train_pred = np.zeros((x_train.shape[0]))

    test_pred = np.zeros((x_test.shape[0]))



    if len(cat_cols)==0: use_cat=False



    # use the k-fold object to enumerate indexes for each training and validation fold

    for i, (train_index, val_index) in enumerate(kf.split(x_train, y_train)): # folds 1, 2 ,3 ,4, 5

        # example: training from 1,2,3,4; validation from 5

        x_train_kf, x_val_kf = x_train.loc[train_index, :], x_train.loc[val_index, :]

        y_train_kf, y_val_kf = y_train[train_index], y_train[val_index]



        if use_cat:

            lgb_train = lgb.Dataset(x_train_kf, y_train_kf, categorical_feature=cat_cols)

            lgb_val = lgb.Dataset(x_val_kf, y_val_kf, reference=lgb_train, categorical_feature=cat_cols)

        else:

            lgb_train = lgb.Dataset(x_train_kf, y_train_kf)

            lgb_val = lgb.Dataset(x_val_kf, y_val_kf, reference=lgb_train)



        gbm = lgb.train(params,

                        lgb_train,

                        num_boost_round=4000,

                        valid_sets=lgb_val,

                        early_stopping_rounds=30,

                        verbose_eval=verbose_eval)



        val_pred = gbm.predict(x_val_kf)



        if use_rank:

            train_pred[val_index] += probability_to_rank(val_pred)

            test_pred += probability_to_rank(gbm.predict(x_test))

            # test_pred += gbm.predict(x_test)

        else:

            train_pred[val_index] += val_pred

            test_pred += gbm.predict(x_test)



        # test_pred += gbm.predict(x_test)

        fold_auc = roc_auc_score(y_val_kf.values, val_pred)

        fold_gini_norm = auc_to_gini_norm(fold_auc)

        if verbose:

            print('fold cv {} AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(i, fold_auc, fold_gini_norm))



    test_pred /= kf.n_splits



    cv_auc = roc_auc_score(y_train, train_pred)

    cv_gini_norm = auc_to_gini_norm(cv_auc)

    cv_score = [cv_auc, cv_gini_norm]

    if verbose:

        print('cv AUC score is {:.6f}, Gini_Norm score is {:.6f}'.format(cv_auc, cv_gini_norm))

        end_time = time.time()

        print("it takes %.3f seconds to perform cross validation" % (end_time - start_time))

    return cv_score, train_pred,test_pred

drop_cols=['id','target']

y_train=train['target']

x_train=train.drop(drop_cols, axis=1)

x_test=test.drop(['id'], axis=1)
kf=StratifiedKFold(n_splits=5, shuffle=True, random_state=2017)
rf=RandomForestClassifier(n_estimators=200, n_jobs=6, min_samples_split=5, max_depth=7,

                          criterion='gini', random_state=0)



outcomes =cross_validate_sklearn(rf, x_train, y_train ,x_test, kf, scale=False, verbose=True)



rf_cv=outcomes[0]

rf_train_pred=outcomes[1]

rf_test_pred=outcomes[2]



rf_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=rf_train_pred)

rf_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=rf_test_pred)

et=RandomForestClassifier(n_estimators=100, n_jobs=6, min_samples_split=5, max_depth=5,

                          criterion='gini', random_state=0)



outcomes =cross_validate_sklearn(et, x_train, y_train ,x_test, kf, scale=False, verbose=True)



et_cv=outcomes[0]

et_train_pred=outcomes[1]

et_test_pred=outcomes[2]



et_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=et_train_pred)

et_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=et_test_pred)
logit=LogisticRegression(random_state=0, C=0.5)



outcomes = cross_validate_sklearn(logit, x_train, y_train ,x_test, kf, scale=True, verbose=True)



logit_cv=outcomes[0]

logit_train_pred=outcomes[1]

logit_test_pred=outcomes[2]



logit_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=logit_train_pred)

logit_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=logit_test_pred)
nb=BernoulliNB()



outcomes =cross_validate_sklearn(nb, x_train, y_train ,x_test, kf, scale=True, verbose=True)



nb_cv=outcomes[0]

nb_train_pred=outcomes[1]

nb_test_pred=outcomes[2]



nb_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=nb_train_pred)

nb_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=nb_test_pred)
xgb_params = {

    "booster"  :  "gbtree", 

    "objective"         :  "binary:logistic",

    "tree_method": "hist",

    "eval_metric": "auc",

    "eta": 0.1,

    "max_depth": 5,

    "min_child_weight": 10,

    "gamma": 0.70,

    "subsample": 0.76,

    "colsample_bytree": 0.95,

    "nthread": 6,

    "seed": 0,

    'silent': 1

}



outcomes=cross_validate_xgb(xgb_params, x_train, y_train, x_test, kf, use_rank=False, verbose_eval=False)



xgb_cv=outcomes[0]

xgb_train_pred=outcomes[1]

xgb_test_pred=outcomes[2]



xgb_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=xgb_train_pred)

xgb_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=xgb_test_pred)
lgb_params = {

    'task': 'train',

    'boosting_type': 'dart',

    'objective': 'binary',

    'metric': {'auc'},

    'num_leaves': 22,

    'min_sum_hessian_in_leaf': 20,

    'max_depth': 5,

    'learning_rate': 0.1,  # 0.618580

    'num_threads': 6,

    'feature_fraction': 0.6894,

    'bagging_fraction': 0.4218,

    'max_drop': 5,

    'drop_rate': 0.0123,

    'min_data_in_leaf': 10,

    'bagging_freq': 1,

    'lambda_l1': 1,

    'lambda_l2': 0.01,

    'verbose': 1

}





cat_cols=['ps_ind_02_cat','ps_car_04_cat', 'ps_car_09_cat','ps_ind_05_cat', 'ps_car_01_cat']

outcomes=cross_validate_lgb(lgb_params,x_train, y_train ,x_test,kf, cat_cols, use_cat=True, 

                            verbose_eval=False, use_rank=False)



lgb_cv=outcomes[0]

lgb_train_pred=outcomes[1]

lgb_test_pred=outcomes[2]



lgb_train_pred_df=pd.DataFrame(columns=['prediction_probability'], data=lgb_train_pred)

lgb_test_pred_df=pd.DataFrame(columns=['prediction_probability'], data=lgb_test_pred)
columns=['rf','et','logit','nb','xgb','lgb']

train_pred_df_list=[rf_train_pred_df, et_train_pred_df, logit_train_pred_df, nb_train_pred_df,

                    xgb_train_pred_df, lgb_train_pred_df]



test_pred_df_list=[rf_test_pred_df, et_test_pred_df, logit_test_pred_df, nb_test_pred_df,

                    xgb_test_pred_df, lgb_test_pred_df]



lv1_train_df=pd.DataFrame(columns=columns)

lv1_test_df=pd.DataFrame(columns=columns)



for i in range(0,len(columns)):

    lv1_train_df[columns[i]]=train_pred_df_list[i]['prediction_probability']

    lv1_test_df[columns[i]]=test_pred_df_list[i]['prediction_probability']



xgb_lv2_outcomes=cross_validate_xgb(xgb_params, lv1_train_df, y_train, lv1_test_df, kf, 

                                          verbose=True, verbose_eval=False, use_rank=False)



xgb_lv2_cv=xgb_lv2_outcomes[0]

xgb_lv2_train_pred=xgb_lv2_outcomes[1]

xgb_lv2_test_pred=xgb_lv2_outcomes[2]
xgb_lv2_outcomes=cross_validate_xgb(xgb_params, lv1_train_df, y_train, lv1_test_df, kf, 

                                          verbose=True, verbose_eval=False, use_rank=True)



xgb_lv2_cv=xgb_lv2_outcomes[0]

xgb_lv2_train_pred=xgb_lv2_outcomes[1]

xgb_lv2_test_pred=xgb_lv2_outcomes[2]
lgb_lv2_outcomes=cross_validate_lgb(lgb_params,lv1_train_df, y_train ,lv1_test_df,kf, [], use_cat=False, 

                                    verbose_eval=False, use_rank=True)



lgb_lv2_cv=xgb_lv2_outcomes[0]

lgb_lv2_train_pred=lgb_lv2_outcomes[1]

lgb_lv2_test_pred=lgb_lv2_outcomes[2]
rf_lv2=RandomForestClassifier(n_estimators=200, n_jobs=6, min_samples_split=5, max_depth=7,

                          criterion='gini', random_state=0)

rf_lv2_outcomes = cross_validate_sklearn(rf_lv2, lv1_train_df, y_train ,lv1_test_df, kf, 

                                            scale=True, verbose=True)

rf_lv2_cv=rf_lv2_outcomes[0]

rf_lv2_train_pred=rf_lv2_outcomes[1]

rf_lv2_test_pred=rf_lv2_outcomes[2]
logit_lv2=LogisticRegression(random_state=0, C=0.5)

logit_lv2_outcomes = cross_validate_sklearn(logit_lv2, lv1_train_df, y_train ,lv1_test_df, kf, 

                                            scale=True, verbose=True)

logit_lv2_cv=logit_lv2_outcomes[0]

logit_lv2_train_pred=logit_lv2_outcomes[1]

logit_lv2_test_pred=logit_lv2_outcomes[2]
lv2_columns=['rf_lf2', 'logit_lv2', 'xgb_lv2','lgb_lv2']

train_lv2_pred_list=[rf_lv2_train_pred, logit_lv2_train_pred, xgb_lv2_train_pred, lgb_lv2_train_pred]



test_lv2_pred_list=[rf_lv2_test_pred, logit_lv2_test_pred, xgb_lv2_test_pred, lgb_lv2_test_pred]



lv2_train=pd.DataFrame(columns=lv2_columns)

lv2_test=pd.DataFrame(columns=lv2_columns)



for i in range(0,len(lv2_columns)):

    lv2_train[lv2_columns[i]]=train_lv2_pred_list[i]

    lv2_test[lv2_columns[i]]=test_lv2_pred_list[i]
xgb_lv3_params = {

    "booster"  :  "gbtree", 

    "objective"         :  "binary:logistic",

    "tree_method": "hist",

    "eval_metric": "auc",

    "eta": 0.1,

    "max_depth": 2,

    "min_child_weight": 10,

    "gamma": 0.70,

    "subsample": 0.76,

    "colsample_bytree": 0.95,

    "nthread": 6,

    "seed": 0,

    'silent': 1

}







xgb_lv3_outcomes=cross_validate_xgb(xgb_lv3_params, lv2_train, y_train, lv2_test, kf, 

                                          verbose=True, verbose_eval=False, use_rank=True)



xgb_lv3_cv=xgb_lv3_outcomes[0]

xgb_lv3_train_pred=xgb_lv3_outcomes[1]

xgb_lv3_test_pred=xgb_lv3_outcomes[2]
logit_lv3=LogisticRegression(random_state=0, C=0.5)

logit_lv3_outcomes = cross_validate_sklearn(logit_lv3, lv2_train, y_train ,lv2_test, kf, 

                                            scale=True, verbose=True)

logit_lv3_cv=logit_lv3_outcomes[0]

logit_lv3_train_pred=logit_lv3_outcomes[1]

logit_lv3_test_pred=logit_lv3_outcomes[2]
weight_avg=logit_lv3_train_pred*0.5+ xgb_lv3_train_pred*0.5

print(auc_to_gini_norm(roc_auc_score(y_train, weight_avg)))
submission=sample_submission.copy()

submission['target']=logit_lv3_test_pred*0.5+ xgb_lv3_test_pred*0.5

filename='stacking_demonstration.csv.gz'

submission.to_csv(filename,compression='gzip', index=False)
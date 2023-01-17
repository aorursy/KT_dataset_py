# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Importing core libraries

import numpy as np

import pandas as pd

from time import time

import pprint

import joblib



# Classifiers

from catboost import CatBoostClassifier, Pool



# Model selection

from sklearn.model_selection import StratifiedKFold



# Metrics

from sklearn.metrics import roc_auc_score, average_precision_score

from sklearn.metrics import make_scorer
# Setting a 12-fold stratified cross-validation (note: shuffle=True)

SEED = 42

FOLDS = 10



skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
# Reading the data

X = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/train.csv")

Xt = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/test.csv")



# Separating target and ids

y = X.target.values

id_train = X.id

id_test = Xt.id



X.drop(['id', 'target'], axis=1, inplace=True)

Xt.drop(['id'], axis=1, inplace=True)



# Classifying variables in binary, high and low cardinality nominal, ordinal and dates

binary_vars = [c for c in X.columns if 'bin_' in c]



nominal_vars = [c for c in X.columns if 'nom_' in c]

high_cardinality = [c for c in nominal_vars if len(X[c].unique()) > 16]

low_cardinality = [c for c in nominal_vars if len(X[c].unique()) <= 16]



ordinal_vars = [c for c in X.columns if 'ord_' in c]



time_vars = ['day', 'month']
# Some feature engineering

X['ord_5_1'] = X['ord_5'].apply(lambda x: x[0] if type(x) == str else np.nan)

X['ord_5_2'] = X['ord_5'].apply(lambda x: x[1] if type(x) == str else np.nan)

Xt['ord_5_1'] = Xt['ord_5'].apply(lambda x: x[0] if type(x) == str else np.nan)

Xt['ord_5_2'] = Xt['ord_5'].apply(lambda x: x[1] if type(x) == str else np.nan)



ordinal_vars += ['ord_5_1', 'ord_5_2']
# Converting ordinal labels into ordered values

ordinals = {

    'ord_1' : {

        'Novice' : 0,

        'Contributor' : 1,

        'Expert' : 2,

        'Master' : 3,

        'Grandmaster' : 4

    },

    'ord_2' : {

        'Freezing' : 0,

        'Cold' : 1,

        'Warm' : 2,

        'Hot' : 3,

        'Boiling Hot' : 4,

        'Lava Hot' : 5

    }

}



def return_order(X, Xt, var_name):

    mode = X[var_name].mode()[0]

    el = sorted(set(X[var_name].fillna(mode).unique())|set(Xt[var_name].fillna(mode).unique()))

    return {v:e for e, v in enumerate(el)}



for mapped_var in ordinal_vars:

    if mapped_var not in ordinals:

        mapped_values = return_order(X, Xt, mapped_var)

        X[mapped_var + '_num'] = X[mapped_var].replace(mapped_values)

        Xt[mapped_var + '_num'] = Xt[mapped_var].replace(mapped_values)

    else:

        X[mapped_var + '_num'] = X[mapped_var].replace(ordinals[mapped_var])

        Xt[mapped_var + '_num'] = Xt[mapped_var].replace(ordinals[mapped_var])
# Transforming all the labels of all variables

from sklearn.preprocessing import LabelEncoder



label_encoders = [LabelEncoder() for _ in range(X.shape[1])]



for col, column in enumerate(X.columns):

    unique_values = pd.Series(X[column].append(Xt[column]).unique())

    unique_values = unique_values[unique_values.notnull()]

    label_encoders[col].fit(unique_values)

    X.loc[X[column].notnull(), column] = label_encoders[col].transform(X.loc[X[column].notnull(), column])

    Xt.loc[Xt[column].notnull(), column] = label_encoders[col].transform(Xt.loc[Xt[column].notnull(), column])
# Dealing with any residual missing value

X = X.fillna(-1)

Xt = Xt.fillna(-1)
# Enconding frequencies instead of labels (so we have some numeric variables)

def frequency_encoding(column, df, df_test=None):

    frequencies = df[column].value_counts().reset_index()

    df_values = df[[column]].merge(frequencies, how='left', 

                                   left_on=column, right_on='index').iloc[:,-1].values

    if df_test is not None:

        df_test_values = df_test[[column]].merge(frequencies, how='left', 

                                                 left_on=column, right_on='index').fillna(1).iloc[:,-1].values

    else:

        df_test_values = None

    return df_values, df_test_values



for column in X.columns:

    train_values, test_values = frequency_encoding(column, X, Xt)

    X[column+'_counts'] = train_values

    Xt[column+'_counts'] = test_values
# Target encoding of selected variables

import category_encoders as cat_encs



cat_feat_to_encode = binary_vars + ordinal_vars + nominal_vars + time_vars

smoothing = 0.3



enc_x = np.zeros(X[cat_feat_to_encode].shape)



for tr_idx, oof_idx in skf.split(X, y):

    encoder = cat_encs.TargetEncoder(cols=cat_feat_to_encode, smoothing=smoothing)

    

    encoder.fit(X[cat_feat_to_encode].iloc[tr_idx], y[tr_idx])

    enc_x[oof_idx, :] = encoder.transform(X[cat_feat_to_encode].iloc[oof_idx], y[oof_idx])

    

encoder.fit(X[cat_feat_to_encode], y)

enc_xt = encoder.transform(Xt[cat_feat_to_encode]).values



for idx, new_var in enumerate(cat_feat_to_encode):

    new_var = new_var + '_enc'

    X[new_var] = enc_x[:,idx]

    Xt[new_var] = enc_xt[:, idx]
# Setting all to dtype float32

X = X.astype(np.float32)

Xt = Xt.astype(np.float32)



# Defining categorical variables

cat_features = nominal_vars + ordinal_vars



# Setting categorical variables to int64

X[cat_features] = X[cat_features].astype(np.int64)

Xt[cat_features] = Xt[cat_features].astype(np.int64)
# Initializing a CatBoostClassifier with best parameters

best_params = {'bagging_temperature': 0.8,

               'depth': 5,

               'iterations': 1000,

               'l2_leaf_reg': 30,

               'learning_rate': 0.05,

               'random_strength': 0.8}
# CV interations

roc_auc = list()

average_precision = list()

oof = np.zeros(len(X))

cv_test_preds = np.zeros(len(Xt))

best_iteration = list()



for train_idx, test_idx in skf.split(X, y):

    X_train, y_train = X.iloc[train_idx, :], y[train_idx]

    X_test, y_test = X.iloc[test_idx, :], y[test_idx]

    

    train = Pool(data=X_train, 

             label=y_train,            

             feature_names=list(X_train.columns),

             cat_features=cat_features)



    val = Pool(data=X_test, 

               label=y_test,

               feature_names=list(X_test.columns),

               cat_features=cat_features)



    catb = CatBoostClassifier(**best_params,

                          loss_function='Logloss',

                          eval_metric = 'AUC',

                          nan_mode='Min',

                          thread_count=2,

                          verbose = False)

    

    catb.fit(train,

             verbose_eval=100, 

             early_stopping_rounds=50,

             eval_set=val,

             use_best_model=True,

             #task_type = "GPU",

             plot=False)

    

    best_iteration.append(catb.best_iteration_)

    preds = catb.predict_proba(X_test)

    oof[test_idx] = preds[:,1]

    

    # CV test prediction

    Xt_pool = Pool(data=Xt[list(X_train.columns)],

               feature_names=list(X_train.columns),

               cat_features=cat_features)

    

    cv_test_preds += catb.predict_proba(Xt_pool)[:,1] / FOLDS

    

    roc_auc.append(roc_auc_score(y_true=y_test, y_score=preds[:,1]))

    average_precision.append(average_precision_score(y_true=y_test, y_score=preds[:,1]))
# Storing results to disk

oof = pd.DataFrame({'id':id_train, 'catboost_oof': oof})

oof.to_csv("oof.csv", index=False)



cv_submission = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")

cv_submission.target = cv_test_preds

cv_submission.to_csv("./catboost_cv_submission.csv", index=False)
print("Average cv roc auc score %0.3f ± %0.3f" % (np.mean(roc_auc), np.std(roc_auc)))

print("Average cv roc average precision %0.3f ± %0.3f" % (np.mean(average_precision), np.std(average_precision)))



print("Roc auc score OOF %0.3f" % roc_auc_score(y_true=y, y_score=oof.catboost_oof))

print("Average precision OOF %0.3f" % average_precision_score(y_true=y, y_score=oof.catboost_oof))
# Using catboost on all the data for predictions

catb = CatBoostClassifier(**best_params,

                          loss_function='Logloss',

                          eval_metric = 'AUC',

                          nan_mode='Min',

                          thread_count=2,

                          verbose = False)



train = Pool(data=X, 

             label=y,            

             feature_names=list(X_train.columns),

             cat_features=cat_features)



catb.fit(train,

         verbose_eval=100,

         #task_type = "GPU",

         plot=False)



Xt_pool = Pool(data=Xt[list(X_train.columns)],

               feature_names=list(X_train.columns),

               cat_features=cat_features)



submission = pd.read_csv("/kaggle/input/cat-in-the-dat-ii/sample_submission.csv")

submission.target = catb.predict_proba(Xt_pool)[:,1]

submission.to_csv("./submission.csv", index=False)
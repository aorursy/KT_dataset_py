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

import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns
train = pd.read_csv('/kaggle/input/cat-in-the-dat/train.csv')

test = pd.read_csv('/kaggle/input/cat-in-the-dat/test.csv')
train.shape
train.head()
columns = [x for x in train.columns if x != 'target']
sns.countplot(x='target',data=train)

#train['target'].value_counts().values
for col in train.columns.tolist():

    print(f'number of unique values for {col} and column data dype : {train[col].dtype}')

    print(f'{train[col].nunique()}')

    print('===========================')

    if train[col].nunique()<8:

        print(train[col].unique())

        print('--------------------------')

    
object_cols = train.select_dtypes('object').columns.tolist()

object_df = train[object_cols]
print(f'is there missing data: {np.any(object_df.isnull().sum()>0)}')

print(f'is there missing data in test set: {np.any(test.isnull().sum()>0)}')
# def split_dataframe(val_size=0.2):

    

#     t_index = len(train.index)

#     val_index = int(t_index*0.4)

#     X_train = train[:-val_index]

#     X_val = train[-val_index:-val_index//2]

#     X_test = train[-val_index//2:]

#     return X_train, X_val, X_test

    

    
def print_value_counts(df):

    

    for col in df.columns:

        print(f'value count for {col} is')

        print(f'{df[col].value_counts()}')

    

            
cols=print_value_counts(object_df)
#object_col_freq = ['nom_5','nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_3', 'ord_4', 'ord_5']

object_col_label = ['bin_0','bin_1','bin_2','bin_3','bin_4']
# Best score 72.**

# one_hot_encode = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

# target_encode = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9', 'ord_0' ]

# cyclic_encode = ['day', 'month']

# ordinal_encode = ['ord_1','ord_2','ord_3','ord_4', 'ord_5']
one_hot_encode = ['nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4']

target_encode = ['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9']

weight_encode = target_encode + ['ord_4', 'ord_5' ,'ord_3'] + one_hot_encode + object_col_label

cyclic_encode = ['day', 'month']

ordinal_encode = ['ord_0','ord_1','ord_2']
train['bin_4'].value_counts()
# def freq_encoding(df, col):

#     temp_dict = dict(df[col].value_counts())

#     return df[col].map(temp_dict)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
#X_train, X_val, X_test = split_dataframe()

# for col in object_col_freq:

#     train[col] = freq_encoding(train,col)

#     test[col] = freq_encoding(test, col)

# # Label Encoding

# Label_train= train[object_col_label]

# Label_test= test[object_col_label]

# encoder = LabelEncoder()

# for col in object_col_label:

#     Label_train[col] = encoder.fit_transform(Label_train[col])

#     Label_test[col] = encoder.fit_transform(Label_test[col])
# one_hot = OneHotEncoder(sparse=False)

# One_hot_train = pd.DataFrame(one_hot.fit_transform(train[one_hot_encode]))

# One_hot_test = pd.DataFrame(one_hot.fit_transform(test[one_hot_encode]))



# One_hot_train.index = train[one_hot_encode].index

# One_hot_test.index = test[one_hot_encode].index

import category_encoders as ce



# # Create the encoder itself

# target_enc = ce.TargetEncoder(cols=target_encode)

# # Fit the encoder using the categorical features and target

# target_enc.fit(train[target_encode], train['target'])



# target_train = target_enc.transform(train[target_encode])

# target_test = target_enc.transform(test[target_encode])
weight_enc = ce.woe.WOEEncoder(cols=weight_encode)

# Fit the encoder using the categorical features and target

weight_enc.fit(train[weight_encode], train['target'])



weight_train = weight_enc.transform(train[weight_encode])

weight_test = weight_enc.transform(test[weight_encode])
# backDiff_enc = ce.backward_difference.BackwardDifferenceEncoder(cols=weight_encode)

# # Fit the encoder using the categorical features and target

# backDiff_enc.fit(train[weight_encode])

# backDiff_train = backDiff_enc.transform(train[weight_encode])

# backDiff_test = backDiff_enc.transform(test[weight_encode])
import gc

gc.collect()
ordinal_train= train[ordinal_encode]

ordinal_test= test[ordinal_encode]

ordinal = OrdinalEncoder()



ordinal_train = pd.DataFrame(ordinal.fit_transform(ordinal_train), columns=ordinal_encode)

ordinal_test = pd.DataFrame(ordinal.fit_transform(ordinal_test), columns=ordinal_encode)
ordinal_train.head()
train[ordinal_encode].head()
# lets try scaling here 

# from sklearn.preprocessing import StandardScaler

# min_max = StandardScaler()

# cols = ordinal_train.columns

# min_max.fit(ordinal_train)

# ordinal_train = pd.DataFrame(min_max.transform(ordinal_train), columns=cols)

# ordinal_test = pd.DataFrame(min_max.transform(ordinal_test), columns=cols)
cyclic_df_train = train[cyclic_encode]

cyclic_df_test = test[cyclic_encode]

for col in cyclic_encode:

#     cyclic_df_train[col] = cyclic_df_train[col].apply(lambda x: np.sin((2*np.pi*x)/max(cyclic_df_train[col])))

#     cyclic_df_test[col] = cyclic_df_test[col].apply(lambda x: np.sin((2*np.pi*x)/max(cyclic_df_test[col])))

    cyclic_df_train[col] = np.sin(2*np.pi*cyclic_df_train[col]/max(cyclic_df_train[col]))

    cyclic_df_test[col] = np.sin(2*np.pi*cyclic_df_test[col]/max(cyclic_df_test[col]))
to_drop = object_col_label+one_hot_encode+target_encode + cyclic_encode + ordinal_encode + ['ord_4','ord_5', 'ord_3']

train.drop(columns=to_drop, inplace=True)

test.drop(columns=to_drop, inplace=True)

# train = pd.concat([train,Label_train,One_hot_train,cyclic_df_train, ordinal_train], axis=1)

# test = pd.concat([test,Label_test,One_hot_test,cyclic_df_test, ordinal_test], axis=1)



train = pd.concat([train,weight_train,cyclic_df_train, ordinal_train], axis=1)

test = pd.concat([test,weight_test,cyclic_df_test, ordinal_test], axis=1)
train.head()
# from imblearn.over_sampling import SMOTE

# target = train.pop('target')

# columns = train.columns
# sm = SMOTE(sampling_strategy='minority',random_state=42)

# X, y = sm.fit_sample(train, target)

# train = pd.concat((pd.DataFrame(X, columns=columns), pd.DataFrame(y,columns=['target'])), axis=1)
train.head()
from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV, KFold

from sklearn.preprocessing import StandardScaler
test_id = test['id']

test.drop(columns=['id'], inplace=True)

train.drop(columns=['id'], inplace=True)
# std_scaler = StandardScaler()

# std_scaler.fit(train)

# train = pd.DataFrame(std_scaler.transform(train),columns=train.columns)

# test = pd.DataFrame(std_scaler.transform(test),columns=test.columns)
from sklearn.metrics import roc_curve, roc_auc_score
Y_train=train.pop('target')
# grid_param={

#     'n_estimators':np.arange(10,250,30),

#     'max_depth': [20,30,10,50]

# }
#rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1)

#grid_rf = GridSearchCV(rf, param_grid=grid_param, cv=5, scoring='roc_auc')
#grid_rf.fit(train, Y_train)
#grid_rf.best_estimator_
# lr = LogisticRegression(C=1.0, class_weight='balanced', dual=False,

#                    fit_intercept=True, intercept_scaling=1, l1_ratio=None,

#                    max_iter=100, multi_class='warn', n_jobs=-1, penalty='l2',

#                    random_state=None, solver='warn', tol=0.0001, verbose=0,

#                    warm_start=False)

# lr.fit(train, Y_train)

# lr_pred = lr.predict_proba(test)
# ada_boost = AdaBoostClassifier(n_estimators=100, learning_rate=0.05, random_state=2021)

# ada_boost.fit(train, Y_train)

# ada_pred = ada_boost.predict_proba(test)
# ada_pred[1,1]
# svc = SVC(kernel='rbf')

# svc.fit(train, Y_train)

# svc_pred = svc.predict_proba(test)
# r_forest = RandomForestClassifier(bootstrap=True, class_weight='balanced',

#                        criterion='gini', max_depth=10, max_features='auto',

#                        max_leaf_nodes=None, min_impurity_decrease=0.0,

#                        min_impurity_split=None, min_samples_leaf=1,

#                        min_samples_split=2, min_weight_fraction_leaf=0.0,

#                        n_estimators=220, n_jobs=-1, oob_score=False,

#                        random_state=None, verbose=0, warm_start=False)

# r_forest.fit(train, Y_train)

# forest_pred = r_forest.predict_proba(test)
# grid_params = {

#     'learning_rate':np.arange(0.01, 0.9, 0.02),

#     'n_estimators':np.arange(100,500,100),

#     'max_depth':np.arange(3,12,3),

#     'max_features':['sqrt','log2'],

#     'min_samples_split':np.arange(0.1, 1.0,0.4),

#     'loss':['exponential']

# }
# gb = GradientBoostingClassifier(validation_fraction=0.2)

# grid_gb = GridSearchCV(gb, param_grid=grid_params, cv=5, scoring='roc_auc')
# grid_gb.fit(train, Y_train)
# grid_gb.best_estimator_
# grd_tree = GradientBoostingClassifier(learning_rate=0.02,

#                                       n_estimators=350,

#                                       max_depth=12,

#                                       max_features='sqrt',

#                                       validation_fraction=0.2,

#                                       random_state=2021)

# grd_tree.fit(train, Y_train)

# grd_pred = grd_tree.predict_proba(test)
from xgboost import XGBClassifier
xgb_params = {

    'max_depth': 6,

    'learning_rate':0.03,

    'n_estimators':400,

    'booster':'gbtree',

    'random_state':2021,

    'subsample':0.5,

    'objective':'binary:logistic',

    'colsample_bytree':0.5, 'colsample_bylevel':0.5, 'colsample_bynode':0.5

#     'sample_type ':'weighted',

#     'normalize_type':'forest',

#     'rate_drop':0.2

}

#kfold = KFold(n_splits=5, shuffle=True, random_state=2021)

#for train_index, test_index in kfold.split(train):

xgb_model = XGBClassifier(**xgb_params)

xgb_model.fit(train, Y_train)

#     predictions = xgb_model.predict(train.iloc[test_index])

#     actuals = Y_train.iloc[test_index]

#     print('auc_score: ',roc_auc_score(actuals,predictions))

xgb_pred = xgb_model.predict_proba(test)
submission_xgb = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')

submission_xgb['id'] = test_id

submission_xgb['target']= xgb_pred[:,1]

submission_xgb.head()
# submission_ada = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')

# submission_ada['id'] = test_id

# submission_ada['target']= grd_pred[:,1]

# submission_ada.head()
# submission_lr = pd.read_csv('/kaggle/input/cat-in-the-dat/sample_submission.csv')

# submission_lr['id'] = test_id

# submission_lr['target']= lr_pred[:,1]

# submission_lr.head()
#submission_ada.to_csv('sample_submission_grad.csv', index=False)

#submission_lr.to_csv('sample_submission_lr.csv', index=False)

submission_xgb.to_csv('sample_submission_xgb.csv', index=False)
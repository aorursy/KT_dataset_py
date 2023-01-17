import os

from scipy.stats import kurtosis, iqr, skew, gmean, hmean, kurtosistest, mode, normaltest, skewtest, shapiro

from functools import partial

import numpy as np

import pandas as pd

from scipy.sparse import hstack

from sklearn import preprocessing

from sklearn.model_selection import KFold

from sklearn.metrics import roc_auc_score as auc

from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

import matplotlib.pylab as plt

from category_encoders import *

import warnings

warnings.simplefilter('ignore')





for dirname, _, filenames in os.walk('/kaggle/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))





def get_func_with_params(func, kwargs_dict):

    return partial(func, **kwargs_dict)





def get_vals(series):

    values = series.values

    values = values[~np.isnan(values)]

    return values





def hmean_safe(series):

    values = get_vals(series)

    return hmean(values)

train = pd.read_csv("/kaggle/input/widsdatathon2020/training_v2.csv")

test = pd.read_csv("/kaggle/input/widsdatathon2020/unlabeled.csv")
train.loc[train['apache_4a_hospital_death_prob'] == -1, 'apache_4a_hospital_death_prob'] = np.nan

test.loc[test['apache_4a_hospital_death_prob'] == -1, 'apache_4a_hospital_death_prob'] = np.nan



train.loc[train['apache_4a_icu_death_prob'] == -1, 'apache_4a_icu_death_prob'] = np.nan

test.loc[test['apache_4a_icu_death_prob'] == -1, 'apache_4a_icu_death_prob'] = np.nan





test.drop("hospital_death",inplace=True,axis=1)

y = train["hospital_death"]

train.drop("hospital_death",inplace=True,axis=1)



del train['patient_id'], test['patient_id']

del train['readmission_status'], test['readmission_status']

del train['encounter_id'], test['encounter_id']





df = pd.concat([train['icu_id'], test['icu_id']])



agg = df.value_counts().to_dict()

train['icu_id_counts'] = np.log1p(train['icu_id'].map(agg))

test['icu_id_counts'] = np.log1p(test['icu_id'].map(agg))



df = pd.concat([train['hospital_id'], test['hospital_id']])



agg = df.value_counts().to_dict()

train['hospital_id_counts'] = np.log1p(train['hospital_id'].map(agg))

test['hospital_id_counts'] = np.log1p(test['hospital_id'].map(agg))





train_only_icu_ids = list(set(train['icu_id'].unique()) - set(test['icu_id'].unique()))

test_only_icu_ids = list(set(test['icu_id'].unique()) - set(train['icu_id'].unique()))



train.loc[train['icu_id'].isin(train_only_icu_ids), 'icu_id'] = np.nan

test.loc[test['icu_id'].isin(test_only_icu_ids), 'icu_id'] = np.nan





train_only_hospital_ids = list(set(train['hospital_id'].unique()) - set(test['hospital_id'].unique()))

test_only_hospital_ids = list(set(test['hospital_id'].unique()) - set(train['hospital_id'].unique()))



train.loc[train['hospital_id'].isin(train_only_hospital_ids), 'hospital_id'] = np.nan

test.loc[test['hospital_id'].isin(test_only_hospital_ids), 'hospital_id'] = np.nan





train['apache_3j_diagnosis'] = train['apache_3j_diagnosis'].astype('str')

test['apache_3j_diagnosis'] = test['apache_3j_diagnosis'].astype('str')



train_only = list(set(train['apache_3j_diagnosis'].unique()) - set(test['apache_3j_diagnosis'].unique()))

test_only = list(set(test['apache_3j_diagnosis'].unique()) - set(train['apache_3j_diagnosis'].unique()))



train.loc[train['apache_3j_diagnosis'].isin(train_only), 'apache_3j_diagnosis'] = np.nan

test.loc[test['apache_3j_diagnosis'].isin(test_only), 'apache_3j_diagnosis'] = np.nan



categoricals  =  [

 'icu_type',

 'apache_3j_bodysystem',

 'apache_2_bodysystem',

 'elective_surgery',

 'apache_post_operative',

 'arf_apache',

 'gcs_eyes_apache',

 'gcs_motor_apache',

 'gcs_unable_apache',

 'gcs_verbal_apache',

 'intubated_apache',

 'ventilated_apache',

 'apache_2_diagnosis',

    'icu_stay_type']

for f in categoricals:

    train[f] = train[f].astype('str')

    test[f] = test[f].astype('str')



    train_only = list(set(train[f].unique()) - set(test[f].unique()))

    test_only = list(set(test[f].unique()) - set(train[f].unique()))



    train.loc[train[f].isin(train_only), f] = np.nan

    test.loc[test[f].isin(test_only), f] = np.nan

    

    print(f, len(train_only), len(test_only))
df = pd.concat([train['age'], test['age']])



agg = df.value_counts().to_dict()

train['age_counts'] = np.log1p(train['age'].map(agg))

test['age_counts'] = np.log1p(test['age'].map(agg))





train['agex'] = train['age'].astype('str')

test['agex'] = test['age'].astype('str')



train_only = list(set(train['agex'].unique()) - set(test['agex'].unique()))

test_only = list(set(test['agex'].unique()) - set(train['agex'].unique()))



train.loc[train['agex'].isin(train_only), 'age'] = np.nan

test.loc[test['agex'].isin(test_only), 'age'] = np.nan



del train['agex'], test['agex']
train['hospital_admit_source'] = train['hospital_admit_source'].replace({'Other ICU': 'ICU','ICU to SDU':'SDU', 'Step-Down Unit (SDU)': 'SDU',

                                                                                               'Other Hospital':'Other','Observation': 'Recovery Room','Acute Care/Floor': 'Acute Care'})



train['apache_2_bodysystem'] = train['apache_2_bodysystem'].replace({'Undefined diagnoses': 'Undefined Diagnoses'})

test['hospital_admit_source'] = test['hospital_admit_source'].replace({'Other ICU': 'ICU','ICU to SDU':'SDU', 'Step-Down Unit (SDU)': 'SDU', 'Other Hospital':'Other','Observation': 'Recovery Room','Acute Care/Floor': 'Acute Care'})

test['apache_2_bodysystem'] = test['apache_2_bodysystem'].replace({'Undefined diagnoses': 'Undefined Diagnoses'})



for f in test.columns:

    if test[f].dtype == 'object':

        test[f] = test[f].astype('str')

        train[f] = train[f].astype('str')

        le = preprocessing.LabelEncoder()

        le.fit(pd.concat([train[f], test[f]]))

        

        train[f] = le.transform(train[f])

        test[f] = le.transform(test[f])

        

    print(f, test[f].dtype, np.unique(test[f]).shape)
df = pd.concat([train, test])



train_4a = df.loc[~pd.isnull(df['apache_4a_hospital_death_prob'])].reset_index()

test_4a = df.loc[pd.isnull(df['apache_4a_hospital_death_prob'])].reset_index()



y_a4 = train_4a['apache_4a_hospital_death_prob'].reset_index()

del train_4a['index'], test_4a['index'], y_a4['index']



target = y_a4

kf = KFold(n_splits=5)

oofs = np.zeros(train_4a.shape[0])

preds = np.zeros([test_4a.shape[0], 5])

for i, (trn_index, val_index) in enumerate(kf.split(target)):

    

    lgb_params = {

        'boosting_type': 'gbdt',

        'objective': 'regression',

        'metric': 'auc',

        'learning_rate': 0.01,

        'subsample': 1,

        'colsample_bytree': 0.25,

        'reg_alpha': 3,

        'reg_lambda': 1,

        'min_child_weight': 5,

        'n_estimators': 3000,

        'silent': -1,

        'verbose': -1,

        'max_depth': -1

    }

    features = [f for f in test.columns if f not in ['apache_4a_hospital_death_prob']]

    

    x_train = train_4a.loc[trn_index, features]

    x_val = train_4a.loc[val_index, features]

    

    y_train = target.loc[trn_index]

    y_val = target.loc[val_index]

    

    clf = lgb.LGBMRegressor(**lgb_params)

    clf.fit(

        x_train, y_train,

        eval_set=[(x_train, y_train), (x_val, y_val)],

        verbose=100,

        early_stopping_rounds=100

    )

    

    oofs[val_index] = clf.predict(x_val, num_iteration=clf.best_iteration_).clip(0, 1)

    preds[:, i] = clf.predict(test_4a[features], num_iteration=clf.best_iteration_).clip(0, 1)

    

    

y_preds = np.power(np.prod(preds, axis=1), 1/5)



df.loc[~pd.isnull(df['apache_4a_hospital_death_prob']), 'new_a4_prob'] =  oofs

df.loc[pd.isnull(df['apache_4a_hospital_death_prob']), 'new_a4_prob'] =  y_preds



test = df.iloc[train.shape[0]:].reset_index()

train = df.iloc[0:train.shape[0]].reset_index()

del train['index'], test['index']





del train['apache_4a_hospital_death_prob'], test['apache_4a_hospital_death_prob']

# del train['apache_4a_icu_death_prob'], test['apache_4a_icu_death_prob']
df = pd.concat([train, test])

agg = df.groupby('apache_3j_diagnosis')['new_a4_prob'].agg(['max', 'var', 'mean', 'min'])

agg.columns = ['apache_3j_diagnosis_max_0', 'apache_3j_diagnosis_var_0',

               'apache_3j_diagnosis_mean_0', 'apache_3j_diagnosis_min_0'] 



train = train.merge(agg, on=['apache_3j_diagnosis'], how='left')

test = test.merge(agg, on=['apache_3j_diagnosis'], how='left')





df = pd.concat([train, test])

agg = df.groupby('apache_3j_diagnosis')['apache_4a_icu_death_prob'].agg(['max', 'var',

                                                                         'mean', 'min'])

agg.columns = ['apache_3j_diagnosis_max_1', 'apache_3j_diagnosis_var_1',

               'apache_3j_diagnosis_mean_1', 'apache_3j_diagnosis_min_1'] 



train = train.merge(agg, on=['apache_3j_diagnosis'], how='left')

test = test.merge(agg, on=['apache_3j_diagnosis'], how='left')



del train['apache_4a_icu_death_prob'], test['apache_4a_icu_death_prob']
cats = ['hospital_admit_source', 'ethnicity', 'icu_id', 'hospital_id',

 'icu_admit_source',

 'icu_type',

 'apache_3j_bodysystem',

 'apache_2_bodysystem',

 'elective_surgery',

 'apache_post_operative',

 'arf_apache',

 'gcs_eyes_apache',

 'gcs_motor_apache',

 'gcs_unable_apache',

 'gcs_verbal_apache',

 'intubated_apache',

 'ventilated_apache',

 'aids',

 'diabetes_mellitus',

 'apache_3j_diagnosis',

 'apache_2_diagnosis', 'icu_stay_type'] + ['new_a4_prob',

 'apache_3j_diagnosis_max_0',

 'apache_3j_diagnosis_var_0',

 'apache_3j_diagnosis_mean_0',

 'apache_3j_diagnosis_min_0',

 'apache_3j_diagnosis_max_1',

 'apache_3j_diagnosis_var_1',

 'apache_3j_diagnosis_mean_1',

 'apache_3j_diagnosis_min_1'] #+ ['cirrhosis', 'hepatic_failure']

nums = [f for f in train.columns if f not in cats] 







df = pd.concat([train, test])



g = 'apache_3j_diagnosis'

for f in nums:

    print(f)

    try:

        agg = df.groupby(g)[f].agg(['var'])

        agg.columns = [f + '_v0'] 



        train = train.merge(agg, on=[g], how='left')

        test = test.merge(agg, on=[g], how='left')



    except:

        print('oh..no...')



for f in nums:

    if train[f].min() > 0:

        print(f)

        try:

            agg = df.groupby(g)[f].agg([hmean_safe])

            agg.columns = [f + '_h0'] 



            train = train.merge(agg, on=[g], how='left')

            test = test.merge(agg, on=[g], how='left')



        except:

            print('oh..no...')    

            



for f in nums:

    print(f)

    try:

        agg = df.groupby(g)[f].agg(['mean'])

        agg.columns = [f + '_x0'] 



        train = train.merge(agg, on=[g], how='left')

        test = test.merge(agg, on=[g], how='left')



        train[f + '_x1'] = train[f] - train[f + '_x0'] 

        test[f + '_x1'] = test[f] - test[f + '_x0'] 

        

    except:

        print('oh..no...')



    
rank_features = ['age', 'bmi', 'd1_heartrate_min', 'weight']

for f in rank_features:



    train[f + '_rank'] = np.log1p(df[f].rank()).iloc[0:train.shape[0]].values

    test[f + '_rank'] = np.log1p(df[f].rank()).iloc[train.shape[0]:].values

cats = ['hospital_admit_source', 

 'icu_admit_source',

 'icu_type',

 'apache_3j_bodysystem',

 'apache_2_bodysystem',

 'elective_surgery',

 'apache_post_operative',

 'arf_apache',

 'gcs_eyes_apache',

 'gcs_motor_apache',

 'gcs_unable_apache',

 'gcs_verbal_apache',

 'intubated_apache',

 'ventilated_apache',

 'aids',

 'cirrhosis',

 'diabetes_mellitus',

 'hepatic_failure',

 'apache_2_diagnosis',

        'icu_stay_type']

for f in cats:

        print(np.unique(df[f]).shape, f)

        if np.unique(df[f]).shape[0] > 100:

            agg = df.groupby(f)[f].agg(['count'])

            agg.columns = [f + '_c0'] 



            train = train.merge(agg, on=[f], how='left')

            test = test.merge(agg, on=[f], how='left') 

       



    
from scipy import stats

res = []

for col in test.columns:

    if '_v0' in col or '_x0' in col or  '_x1' in col or '_c0' in col or '_h0' in col  or  '_x2' in col or  '_m0' in col:

        r = stats.ks_2samp(train[col].dropna(), test[col].dropna())

        print(col, r)

        if r[0] >= 0.05:

            del train[col], test[col]

categoricals  =  [

 'icu_type',

 'apache_3j_bodysystem',

 'apache_2_bodysystem',

 'elective_surgery',

 'apache_post_operative',

 'arf_apache',

 'gcs_eyes_apache',

 'gcs_motor_apache',

 'gcs_unable_apache',

 'gcs_verbal_apache',

 'intubated_apache',

 'ventilated_apache',

 'apache_2_diagnosis',

 'apache_3j_diagnosis'] + ['icu_id']
inters = []

cats = categoricals

met = []

for cat_0 in cats:

    for cat_1 in cats:

        if (cat_0, cat_1) not in met and (cat_1, cat_0) not in met and cat_0 != cat_1:

            train[cat_0 + '-' + cat_1] = train[cat_0].astype('str') + '-' + train[cat_1].astype('str')

            test[cat_0 + '-' + cat_1] = test[cat_0].astype('str') + '-' + test[cat_1].astype('str')

            

            met.append((cat_0, cat_1))

            met.append((cat_1, cat_0))

            print((cat_0, cat_1))

            

            inters.append(cat_0 + '-' + cat_1)


for f in inters:

    train[f] = train[f].astype('str')

    test[f] = test[f].astype('str')



    train_only = list(set(train[f].unique()) - set(test[f].unique()))

    test_only = list(set(test[f].unique()) - set(train[f].unique()))



    train.loc[train[f].isin(train_only), f] = np.nan

    test.loc[test[f].isin(test_only), f] = np.nan

    

    print(f, len(train_only), len(test_only))
for f in inters:

    if test[f].dtype == 'object':

        test[f] = test[f].astype('str')

        train[f] = train[f].astype('str')

        le = preprocessing.LabelEncoder()

        le.fit(pd.concat([train[f], test[f]]))

        

        train[f] = le.transform(train[f])

        test[f] = le.transform(test[f])

        

    print(f, test[f].dtype, np.unique(test[f]).shape)
from scipy import stats

res = []

for col in inters:

   

        r = stats.ks_2samp(train[col].dropna(), test[col].dropna())

        print(col, r)

        if r[0] >= 0.1:

            del train[col], test[col]

from scipy.sparse import hstack





categoricals  =  ['hospital_admit_source', 'ethnicity',

 'icu_admit_source',

 'icu_type',

 'apache_3j_bodysystem',

 'apache_2_bodysystem',

 'elective_surgery',

 'apache_post_operative',

 'arf_apache',

 'gcs_eyes_apache',

 'gcs_motor_apache',

 'gcs_unable_apache',

 'gcs_verbal_apache',

 'intubated_apache',

 'ventilated_apache',

 'aids',

 'cirrhosis',

 'diabetes_mellitus',

 'hepatic_failure',

 'apache_2_diagnosis',

 'apache_3j_diagnosis',

        'icu_stay_type'] + inters + ['icu_id', 'hospital_id'] + []



numericals = []

for f in train.columns:

        if f not in categoricals:

            numericals.append(f)

            

xs = []

for f in categoricals:

        test[f] = test[f].astype('str')

        train[f] = train[f].astype('str')

        le = preprocessing.LabelEncoder()

        le.fit(pd.concat([train[f], test[f]]))

        

        train[f] = le.transform(train[f])

        test[f] = le.transform(test[f])

    

        print(f, test[f].dtype, np.unique(test[f]).shape)

        xs.append(pd.get_dummies(pd.concat([train[f], test[f]])))

for f in numericals:

        xs.append(np.concatenate([train[f].values, test[f].values]).reshape(-1, 1))



xs = hstack(xs).tocsr().astype('float32')

xs_train = xs[0:train.shape[0]]

xs_test = xs[train.shape[0]:]

del xs

xs_train.shape
params = {

        'boosting_type': 'goss',

        'objective': 'binary',

        'metric': 'auc',

        'learning_rate': 0.01,

        'subsample': 1,

        'colsample_bytree': 0.1,

        'reg_alpha': 3,

        'reg_lambda': 1,

        'scale_pos_weight': 1,

        'n_estimators': 30000,

        'silent': -1,

        'verbose': -1,

        'max_depth': -1

    }



dtrain = lgb.Dataset(xs_train, y)

evals = lgb.cv(params,

             dtrain,

             nfold=5,

             stratified=True,

             num_boost_round=20000,

             early_stopping_rounds=200,

             verbose_eval=100,

             seed = 666,

             show_stdv=True)

max(evals['auc-mean'])
n = int(1.1 * len(evals['auc-mean']))

n
target = y

preds = np.zeros([test.shape[0], 5])

for i in range(5):

    

    lgb_params = {

        'boosting_type': 'goss',

        'objective': 'binary',

        'metric': 'auc',

        'learning_rate': 0.01,

        'subsample': 1,

        'colsample_bytree': 0.1,

        'reg_alpha': 3,

        'reg_lambda': 1,

        'scale_pos_weight': 1,

        'n_estimators': n,

        'silent': -1,

        'verbose': -1,

        'max_depth': -1,

        'seed':i + 666,

    }

    

    x_train = xs_train

    x_val = xs_train

    

   

    y_train = target

    y_val = target

    

    clf = lgb.LGBMClassifier(**lgb_params)

    clf.fit(

        x_train, y_train,

        eval_set=[(x_train, y_train), (x_val, y_val)],

        verbose=100,

        early_stopping_rounds=None

    )

    

    preds[:, i] = clf.predict_proba(xs_test, num_iteration=clf.best_iteration_)[:, 1]



y_preds = np.power(np.prod(preds, axis=1), 1/5)
sample_submission = pd.read_csv('/kaggle/input/widsdatathon2020/unlabeled.csv', usecols=['encounter_id', 'hospital_death'])

sample_submission['hospital_death'] = y_preds

sample_submission.to_csv("lgbm_submission.csv", header=True, index=False)

sample_submission.head()
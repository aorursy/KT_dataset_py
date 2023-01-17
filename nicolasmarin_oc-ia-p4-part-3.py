# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

from numpy import savetxt

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nmpy_corr as nmc

#import nmpy_df as nmd

#import nmpy_plot as nmp

import missingno as mno

import matplotlib.pyplot as plt

import seaborn as sns

import datetime

from ast import literal_eval



from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer, make_column_selector

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, PolynomialFeatures

from sklearn.experimental import enable_iterative_imputer 

from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer



from sklearn.model_selection import cross_val_score, train_test_split, cross_validate, StratifiedKFold

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import roc_auc_score, roc_curve, f1_score, auc, precision_recall_curve, confusion_matrix, log_loss, matthews_corrcoef, precision_score, recall_score, fbeta_score, cohen_kappa_score, brier_score_loss

from scikitplot.helpers import binary_ks_curve



from sklearn.inspection import partial_dependence

from sklearn.inspection import plot_partial_dependence



import xgboost

from xgboost import XGBClassifier



import lightgbm

from lightgbm import LGBMClassifier



import imblearn

from imblearn.under_sampling import RandomUnderSampler, TomekLinks, ClusterCentroids

from imblearn.over_sampling import RandomOverSampler, SMOTE

from imblearn.combine import SMOTETomek





from functools import partial

from hyperopt import fmin, hp, tpe, Trials, space_eval



import shap

from lime.lime_tabular import LimeTabularExplainer

from pdpbox import pdp



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import gc

import multiprocessing

core_nb = multiprocessing.cpu_count()

print('nombre de threads : {}'.format(core_nb))        



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
apptrain = pd.read_csv('../input/w1-home-credit/application_train.csv', sep=',', encoding='utf-8')

apptest = pd.read_csv('../input/w1-home-credit/application_test.csv', sep=',', encoding='utf-8')

xfeat = pd.read_csv('../input/w1-home-credit/w1_extra_features.csv', sep=',', encoding='utf-8')
model_eval_100 = pd.read_json('../input/w1-home-credit/binClfEval_100---20200512-034959.json')

model_eval_1000 = pd.read_json('../input/w1-home-credit/binClfEval_1000---20200511-042341.json')

hparam_eval_p1 = pd.read_json('../input/w1-home-credit/HParamEval-lGBMC-gbdt-init---2020050605.json')

hparam_eval_p2 = pd.read_json('../input/w1-home-credit/HParamEval-lGBMC-gbdt-init--20200512-111039.json')

hparam_eval_rus = pd.read_json('../input/w1-home-credit/HParamEval---RandomUnderSampler--20200507-152951.json')

hparam_eval_smo = pd.read_json('../input/w1-home-credit/HParamEval---SMOTE--20200507-115631.json')
apptrain_xfeat = apptrain.join(xfeat.set_index('SK_ID_CURR'), on='SK_ID_CURR')

apptrain_xfeat
x = apptrain.select_dtypes(exclude='object').values

col = apptrain.select_dtypes(exclude='object').columns

min_max_scaler = MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

xfeat_normalized = pd.DataFrame(x_scaled, columns=col)

xfeat_normalized.boxplot(figsize=(25,10), rot=90)

plt.show()
x = xfeat.select_dtypes(exclude='object').values

col = xfeat.select_dtypes(exclude='object').columns

min_max_scaler = MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

xfeat_normalized = pd.DataFrame(x_scaled, columns=col)

xfeat_normalized.boxplot(figsize=(25,10), rot=90)

plt.show()
mno.matrix(apptrain_xfeat)

plt.show()
apptrain_xfeat_part1 = apptrain_xfeat.iloc[:,122:164]

apptrain_xfeat_part2 = apptrain_xfeat.iloc[:,164:]

apptrain_xfeat_part1 = pd.concat([apptrain_xfeat_part1,apptrain_xfeat['TARGET']], axis=1, sort=False)

apptrain_xfeat_part2 = pd.concat([apptrain_xfeat_part2,apptrain_xfeat['TARGET']], axis=1, sort=False)
nmc.associations(apptrain_xfeat_part1,figsize=(30,30),theil_u=True, mark_columns=True, clustering=True)
nmc.associations(apptrain_xfeat_part2,figsize=(30,30),theil_u=True, mark_columns=True, clustering=True)
num_si_nml_list = []

num_ii_nml_list = ['CREDIT_TERM','EXT_SOURCE_2','EXT_SOURCE_3']

norm_only_list = []

std_only_list = ['AGE']

#ohe_list = ['OCCUPATION_TYPE','ORGANIZATION_TYPE','NAME_EDUCATION_TYPE','NAME_HOUSING_TYPE']

#ohe_only_list = ['NAME_HOUSING_TYPE']

ohe_nan_list = ['OCCUPATION_TYPE','NAME_HOUSING_TYPE']

#pass_list = []

xfeat_col_list = xfeat.columns.tolist()[1:]

#xfeat_col_list = []

poly_feat_list = ['AGE','CREDIT_TERM','EXT_SOURCE_2','EXT_SOURCE_3']

poly_feat_degree = 3

poly_feat_impstrat = 'median'

poly_feat_output_list = ['1', 'AGE', 'CREDIT_TERM', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 'AGE^2', 'AGE CREDIT_TERM', 'AGE EXT_SOURCE_2', 'AGE EXT_SOURCE_3', 'CREDIT_TERM^2', 'CREDIT_TERM EXT_SOURCE_2', 'CREDIT_TERM EXT_SOURCE_3', 'EXT_SOURCE_2^2', 'EXT_SOURCE_2 EXT_SOURCE_3', 'EXT_SOURCE_3^2', 'AGE^3', 'AGE^2 CREDIT_TERM', 'AGE^2 EXT_SOURCE_2', 'AGE^2 EXT_SOURCE_3', 'AGE CREDIT_TERM^2', 'AGE CREDIT_TERM EXT_SOURCE_2', 'AGE CREDIT_TERM EXT_SOURCE_3', 'AGE EXT_SOURCE_2^2', 'AGE EXT_SOURCE_2 EXT_SOURCE_3', 'AGE EXT_SOURCE_3^2', 'CREDIT_TERM^3', 'CREDIT_TERM^2 EXT_SOURCE_2', 'CREDIT_TERM^2 EXT_SOURCE_3', 'CREDIT_TERM EXT_SOURCE_2^2', 'CREDIT_TERM EXT_SOURCE_2 EXT_SOURCE_3', 'CREDIT_TERM EXT_SOURCE_3^2', 'EXT_SOURCE_2^3', 'EXT_SOURCE_2^2 EXT_SOURCE_3', 'EXT_SOURCE_2 EXT_SOURCE_3^2', 'EXT_SOURCE_3^3']
class MetaFeatureBuilder(BaseEstimator, TransformerMixin):

    def __init__(self):

        return

    def fit(self, df_src):

        return self

    def transform(self, df_input):

        df_src = df_input.copy()

        # Convert DAYS_BIRTH into positive age

        df_src['AGE'] = (df_src['DAYS_BIRTH']/(-1*365)).map('{:,.0f}'.format)

        #df_src.drop('DAYS_BIRTH', axis=1, inplace=True)

        # Combine AMT_CREDIT and AMT_ANNUITY by dividing the first one by the second one > last of the credit at the application time

        df_src['CREDIT_TERM'] = (df_src['AMT_CREDIT'] / df_src['AMT_ANNUITY']).map('{:,.1f}'.format)

        #df_src.drop('AMT_ANNUITY', axis=1, inplace=True)

        return df_src
class PolyFeatureBuilder(BaseEstimator, TransformerMixin):

    def __init__(self, feat2poly='', degrees = 3, imputer_strat='median'):

        self.feat2poly = feat2poly

        self.degrees = degrees

        self.imputer_strat = imputer_strat

        self.imputer_proc = SimpleImputer(missing_values=np.nan, strategy=self.imputer_strat)

        return

    def fit(self, df_src):

        df_src = df_src[self.feat2poly]

        self.imputer_proc.fit(df_src)

        return self

    def transform(self, df_input):

        pts = PolynomialFeatures(degree = self.degrees)

        df_temp = df_input[self.feat2poly]

        df_temp = self.imputer_proc.transform(df_temp)

        #print(self.feat2poly)

        #print(df_input)

        #df_temp = df_temp[self.feat2poly]

        poly_feat = pts.fit_transform(df_temp)

        feat_names = pts.get_feature_names(input_features=self.feat2poly)[:]

        print(len(feat_names), '\n', feat_names)

        self.feat_names = pts.get_feature_names(input_features=self.feat2poly)[:]

        df_output_temp = pd.DataFrame(poly_feat, columns=feat_names)

        df_output = pd.concat([df_input,df_output_temp.iloc[:,len(self.feat2poly)+1:]], axis=1)

        #print(df_output_temp.iloc[:,len(self.feat2poly)+1:])

        return df_output
ppl_cat_nan_ohe = Pipeline([

    ('Cat_SImputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='XNA')),

    ('OHE_after_SImputer', OneHotEncoder(sparse=False))

])

ppl_num_si_nml_core = Pipeline([

    ('Num_SImputer', SimpleImputer(missing_values=np.nan, strategy='median')),

    ('Norm_after_SImputer', MinMaxScaler(feature_range=(0,1)))

])

ppl_num_si_nml_xfeat = Pipeline([

    ('Num_SImputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)),

    ('Norm_after_SImputer', MinMaxScaler(feature_range=(0,1)))

])

ppl_num_ii_nml = Pipeline([

    ('Num_IImputer', IterativeImputer(missing_values=np.nan, initial_strategy='median', max_iter=10, random_state=0)),

    ('Norm_after_IImputer', MinMaxScaler(feature_range=(0,1)))

])

w1_hc_pipeline = Pipeline([

    ('MetaFeatureBuilder',MetaFeatureBuilder()),

    ('PolyFeatureBuilder',PolyFeatureBuilder(poly_feat_list, poly_feat_degree, poly_feat_impstrat)),

    ('Transformers',ColumnTransformer([

        ('Num_SImputer_core', ppl_num_si_nml_core, num_si_nml_list),

        ('Num_IImputer_core', ppl_num_ii_nml, num_ii_nml_list),

        ('Num_SImputer_xfeat', ppl_num_si_nml_xfeat, xfeat_col_list),

        ##('Cat_SImputer', SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='XNA'), make_column_selector(dtype_include=np.object)),     

        ('Normalizer_core', MinMaxScaler(feature_range=(0,1)), norm_only_list),

        ('Standardizer_core',StandardScaler(),std_only_list),

        ('Normalizer_poly', MinMaxScaler(feature_range=(0,1)), poly_feat_output_list[len(poly_feat_list)+1:]),

        #('OHE_only', OneHotEncoder(sparse=False), ohe_only_list),

        ('OHE_NaN', ppl_cat_nan_ohe, ohe_nan_list)

    ],

        remainder='drop'

    )),

])
appxf_full_train_init = apptrain_xfeat.drop('TARGET', axis=1)

appxf_full_train_init.shape
hc_ppl_1 = w1_hc_pipeline

appxf_full_train_tf = hc_ppl_1.fit_transform(appxf_full_train_init)

appxf_full_train_tf.shape
#ohe_1 = hc_ppl_1['Transformers'].transformers_[6][1].get_feature_names()

ohe_2 = hc_ppl_1['Transformers'].transformers_[6][1]['OHE_after_SImputer'].get_feature_names()

#feature_names = num_si_nml_list + num_ii_nml_list + xfeat_col_list + norm_only_list + std_only_list + poly_feat_output_list[len(poly_feat_list)+1:] + ohe_1.tolist() + ohe_2.tolist()

feature_names = num_si_nml_list + num_ii_nml_list + xfeat_col_list + norm_only_list + std_only_list + poly_feat_output_list[len(poly_feat_list)+1:] + ohe_2.tolist()

feature_names
appxf_full_train_lbl = apptrain_xfeat['TARGET']

appxf_full_train_lbl.shape
#savetxt('../working/appxf_full_train_features.csv', appxf_full_train_tf, delimiter=',', encoding='utf-8')

#savetxt('../working/appxf_full_train_lbl.csv', appxf_full_train_lbl, delimiter=',', encoding='utf-8')
apptest_xfeat = apptest.join(xfeat.set_index('SK_ID_CURR'), on='SK_ID_CURR')

appxf_full_test_tf = hc_ppl_1.transform(apptest_xfeat)

appxf_full_test_tf.shape
appxf_split_train_init, appxf_split_test_init, appxf_split_train_lbl, appxf_split_test_lbl = train_test_split(apptrain_xfeat.drop('TARGET', axis=1), apptrain_xfeat['TARGET'], test_size=0.2, random_state=3, shuffle=True, stratify=apptrain_xfeat['TARGET'])

appxf_split_train_init.reset_index(drop=True, inplace=True)

appxf_split_test_init.reset_index(drop=True, inplace=True)

appxf_split_train_lbl.reset_index(drop=True, inplace=True)

appxf_split_test_lbl.reset_index(drop=True, inplace=True)

appxf_split_train_init.shape
hc_ppl_2 = w1_hc_pipeline

appxf_split_train_tf = hc_ppl_2.fit_transform(appxf_split_train_init)

appxf_split_train_tf.shape
#ohe_1 = hc_ppl_0['Transformers'].transformers_[6][1].get_feature_names()

ohe_2 = hc_ppl_2['Transformers'].transformers_[6][1]['OHE_after_SImputer'].get_feature_names()

#feature_names = num_si_nml_list + num_ii_nml_list + xfeat_col_list + norm_only_list + std_only_list + poly_feat_output_list[len(poly_feat_list)+1:] + ohe_1.tolist() + ohe_2.tolist()

feature_names = num_si_nml_list + num_ii_nml_list + xfeat_col_list + norm_only_list + std_only_list + poly_feat_output_list[len(poly_feat_list)+1:] + ohe_2.tolist()

feature_names
appxf_split_test_tf = hc_ppl_2.transform(appxf_split_test_init)

appxf_split_test_tf.shape
#savetxt('../working/appxf_split_train_tf.csv', appxf_split_train_tf, delimiter=',', encoding='utf-8')

#savetxt('../working/appxf_split_train_lbl.csv', appxf_split_train_lbl, delimiter=',', encoding='utf-8')

#savetxt('../working/appxf_split_test_tf.csv', appxf_split_test_tf, delimiter=',', encoding='utf-8')

#savetxt('../working/appxf_split_test_lbl.csv', appxf_split_test_lbl, delimiter=',', encoding='utf-8')
dummy_pred = [0 for _ in range(len(appxf_full_train_lbl))]

len(dummy_pred)
round(roc_auc_score(appxf_full_train_lbl, dummy_pred),6)
dummy_pred_2 = [0 for _ in range(len(apptest_xfeat))]

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':dummy_pred_2})

result_.to_csv('../working/pred_dummy.csv', sep=',', encoding='utf-8', index=False)
model_eval_100.sort_values(by='ROC_AUC', ascending=False)
model_eval_1000.sort_values(by='ROC_AUC', ascending=False)
clf_xgb_1 = XGBClassifier(n_estimators=100, nthread=core_nb, random_state=3)

clf_xgb_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_xgb_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_xgb_1b = XGBClassifier(n_estimators=100, nthread=core_nb, random_state=3)

clf_xgb_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_xgb_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_xgb_base-init.csv', sep=',', encoding='utf-8', index=False)
ros_1 = RandomOverSampler(random_state=3)

appxf_split_ros_train_tf, appxf_split_ros_train_lbl = ros_1.fit_sample(appxf_split_train_tf, appxf_split_train_lbl)

appxf_split_ros_train_lbl.value_counts()
clf_xgb_2 = XGBClassifier(n_estimators=100, nthread=core_nb, random_state=3)

clf_xgb_2.fit(appxf_split_ros_train_tf, appxf_split_ros_train_lbl)

pred_prob_2 = clf_xgb_2.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_2[:,1]),6)
ros_2 = RandomOverSampler(random_state=3)

appxf_full_ros_train_tf, appxf_full_ros_train_lbl = ros_2.fit_sample(appxf_full_train_tf,appxf_full_train_lbl)

appxf_full_ros_train_lbl.value_counts()
clf_xgb_2b = XGBClassifier(n_estimators=100, nthread=core_nb, random_state=3)

clf_xgb_2b.fit(appxf_full_ros_train_tf, appxf_full_ros_train_lbl)

pred_prob_2b = clf_xgb_2b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_2b[:,1]})

result_.to_csv('../working/pred_xgb_base-ros.csv', sep=',', encoding='utf-8', index=False)
del appxf_split_ros_train_tf

del appxf_full_ros_train_tf

gc.collect()
smo_1 = SMOTE(sampling_strategy='minority', n_jobs=core_nb, random_state=3)

appxf_split_smo_train_tf, appxf_split_smo_train_lbl = smo_1.fit_sample(appxf_split_train_tf, appxf_split_train_lbl)

appxf_split_smo_train_lbl.value_counts()
clf_xgb_3 = XGBClassifier(n_estimators=100, nthread=core_nb, random_state=3)

clf_xgb_3.fit(appxf_split_smo_train_tf, appxf_split_smo_train_lbl)

pred_prob_3 = clf_xgb_3.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_3[:,1]),6)
smo_2 = SMOTE(sampling_strategy='minority', n_jobs=core_nb, random_state=3)

appxf_full_smo_train_tf, appxf_full_smo_train_lbl = smo_2.fit_sample(appxf_full_train_tf,appxf_full_train_lbl)

appxf_full_smo_train_lbl.value_counts()
clf_xgb_3b = XGBClassifier(n_estimators=100, nthread=core_nb, random_state=3)

clf_xgb_3b.fit(appxf_full_smo_train_tf, appxf_full_smo_train_lbl)

pred_prob_3b = clf_xgb_3b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_3b[:,1]})

result_.to_csv('../working/pred_xgb_base-smo.csv', sep=',', encoding='utf-8', index=False)
del appxf_split_smo_train_tf

del appxf_full_smo_train_tf

gc.collect()
rus_1 = RandomUnderSampler(random_state=3)

appxf_split_rus_train_tf, appxf_split_rus_train_lbl = rus_1.fit_sample(appxf_split_train_tf, appxf_split_train_lbl)

appxf_split_rus_train_lbl.value_counts()
clf_xgb_4 = XGBClassifier(n_estimators=100, nthread=core_nb, random_state=3)

clf_xgb_4.fit(appxf_split_rus_train_tf, appxf_split_rus_train_lbl)

pred_prob_4 = clf_xgb_4.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_4[:,1]),6)
rus_2 = RandomUnderSampler(random_state=3)

appxf_full_rus_train_tf, appxf_full_rus_train_lbl = rus_2.fit_sample(appxf_full_train_tf,appxf_full_train_lbl)

appxf_full_rus_train_lbl.value_counts()
clf_xgb_4b = XGBClassifier(n_estimators=100, nthread=core_nb, random_state=3)

clf_xgb_4b.fit(appxf_full_rus_train_tf, appxf_full_rus_train_lbl)

pred_prob_4b = clf_xgb_4b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_4b[:,1]})

result_.to_csv('../working/pred_xgb_base-rus.csv', sep=',', encoding='utf-8', index=False)
del appxf_split_rus_train_tf

del appxf_full_rus_train_tf

gc.collect()
clf_lgbm_1 = LGBMClassifier(n_estimators=100, n_jobs=core_nb, random_state=3)

clf_lgbm_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_lgbm_1b = LGBMClassifier(n_estimators=100, n_jobs=core_nb, random_state=3)

clf_lgbm_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_base-init.csv', sep=',', encoding='utf-8', index=False)
ros_1 = RandomOverSampler(random_state=3)

appxf_split_ros_train_tf, appxf_split_ros_train_lbl = ros_1.fit_sample(appxf_split_train_tf, appxf_split_train_lbl)

appxf_split_ros_train_lbl.value_counts()
clf_lgbm_2 = LGBMClassifier(n_estimators=1000, n_jobs=core_nb, random_state=3)

clf_lgbm_2.fit(appxf_split_ros_train_tf, appxf_split_ros_train_lbl)

pred_prob_2 = clf_lgbm_2.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_2[:,1]),6)
ros_2 = RandomOverSampler(random_state=3)

appxf_full_ros_train_tf, appxf_full_ros_train_lbl = ros_2.fit_sample(appxf_full_train_tf,appxf_full_train_lbl)

appxf_full_ros_train_lbl.value_counts()
clf_lgbm_2b = LGBMClassifier(n_estimators=1000, n_jobs=core_nb, random_state=3)

clf_lgbm_2b.fit(appxf_full_ros_train_tf, appxf_full_ros_train_lbl)

pred_prob_2b = clf_lgbm_2b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_2b[:,1]})

result_.to_csv('../working/pred_lgbm_base-ros.csv', sep=',', encoding='utf-8', index=False)
del appxf_split_ros_train_tf

del appxf_full_ros_train_tf

gc.collect()
rus_1 = RandomUnderSampler(random_state=3)

appxf_split_rus_train_tf, appxf_split_rus_train_lbl = rus_1.fit_sample(appxf_split_train_tf, appxf_split_train_lbl)

appxf_split_rus_train_lbl.value_counts()
clf_lgbm_3 = LGBMClassifier(n_estimators=1000, n_jobs=core_nb, random_state=3)

clf_lgbm_3.fit(appxf_split_rus_train_tf, appxf_split_rus_train_lbl)

pred_prob_3 = clf_lgbm_3.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_3[:,1]),6)
rus_2 = RandomUnderSampler(random_state=3)

appxf_full_rus_train_tf, appxf_full_rus_train_lbl = rus_2.fit_sample(appxf_full_train_tf,appxf_full_train_lbl)

appxf_full_rus_train_lbl.value_counts()
clf_lgbm_3b = LGBMClassifier(n_estimators=1000, n_jobs=core_nb, random_state=3)

clf_lgbm_3b.fit(appxf_full_rus_train_tf, appxf_full_rus_train_lbl)

pred_prob_3b = clf_lgbm_3b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_3b[:,1]})

result_.to_csv('../working/pred_lgbm_base-rus.csv', sep=',', encoding='utf-8', index=False)
del appxf_split_rus_train_tf

del appxf_full_rus_train_tf

gc.collect()
smo_1 = SMOTE(sampling_strategy='minority', n_jobs=core_nb, random_state=3)

appxf_split_smo_train_tf, appxf_split_smo_train_lbl = smo_1.fit_sample(appxf_split_train_tf, appxf_split_train_lbl)

appxf_split_smo_train_lbl.value_counts()
clf_lgbm_4 = LGBMClassifier(n_estimators=1000, n_jobs=core_nb, random_state=3)

clf_lgbm_4.fit(appxf_split_smo_train_tf, appxf_split_smo_train_lbl)

pred_prob_4 = clf_lgbm_4.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_4[:,1]),6)
smo_2 = SMOTE(sampling_strategy='minority', n_jobs=core_nb, random_state=3)

appxf_full_smo_train_tf, appxf_full_smo_train_lbl = smo_2.fit_sample(appxf_full_train_tf,appxf_full_train_lbl)

appxf_full_smo_train_lbl.value_counts()
clf_lgbm_4b = LGBMClassifier(n_estimators=1000, n_jobs=core_nb, random_state=3)

clf_lgbm_4b.fit(appxf_full_smo_train_tf, appxf_full_smo_train_lbl)

pred_prob_4b = clf_lgbm_4b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_4b[:,1]})

result_.to_csv('../working/pred_lgbm_base-smo.csv', sep=',', encoding='utf-8', index=False)
del appxf_split_smo_train_tf

del appxf_full_smo_train_tf

gc.collect()
clf_skgb_1 = GradientBoostingClassifier(n_estimators=100, random_state=3)

clf_skgb_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_skgb_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_skgb_1b = GradientBoostingClassifier(n_estimators=100, random_state=3)

clf_skgb_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_skgb_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_skgb_base-init.csv', sep=',', encoding='utf-8', index=False)
hparam_eval_p1.sort_values(by='ROC_AUC', ascending=False)
hparam_eval_p2.sort_values(by='ROC_AUC', ascending=False)
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.10,

    n_estimators=800,

    max_depth=4,

    num_leaves=12,

    scale_pos_weight=1.258513,

    lambda_l1=0.0,

    lambda_l2=6.248898e-03,

    min_data_in_leaf=136,

    bagging_fraction=0.988737,

    feature_fraction=0.737824,

    min_child_weight=154.477441,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.10,

    n_estimators=800,

    max_depth=4,

    num_leaves=12,

    scale_pos_weight=1.258513,

    lambda_l1=0.0,

    lambda_l2=6.248898e-03,

    min_data_in_leaf=136,

    bagging_fraction=0.988737,

    feature_fraction=0.737824,

    min_child_weight=154.477441,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opt1-init.csv', sep=',', encoding='utf-8', index=False)
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.05,

    n_estimators=800,

    max_depth=4,

    num_leaves=12,

    scale_pos_weight=1.960861,

    lambda_l1=4.899498,

    lambda_l2=3.850000e-07,

    min_data_in_leaf=119,

    bagging_fraction=0.853731,

    feature_fraction=0.984112,

    min_child_weight=3.339487,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.05,

    n_estimators=800,

    max_depth=4,

    num_leaves=12,

    scale_pos_weight=1.960861,

    lambda_l1=4.899498,

    lambda_l2=3.850000e-07,

    min_data_in_leaf=119,

    bagging_fraction=0.853731,

    feature_fraction=0.984112,

    min_child_weight=3.339487,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opt2-init.csv', sep=',', encoding='utf-8', index=False)
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.10,

    n_estimators=800,

    max_depth=4,

    num_leaves=6,

    scale_pos_weight=1.005525,

    lambda_l1=0.0,

    lambda_l2=1.177467e-02,

    min_data_in_leaf=346,

    bagging_fraction=0.876592,

    feature_fraction=0.682101,

    min_child_weight=1.210909,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.10,

    n_estimators=800,

    max_depth=4,

    num_leaves=6,

    scale_pos_weight=1.005525,

    lambda_l1=0.0,

    lambda_l2=1.177467e-02,

    min_data_in_leaf=346,

    bagging_fraction=0.876592,

    feature_fraction=0.682101,

    min_child_weight=1.210909,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opt3-init.csv', sep=',', encoding='utf-8', index=False)
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.05,

    n_estimators=400,

    max_depth=8,

    num_leaves=128,

    scale_pos_weight=1.850452,

    lambda_l1=17.345933,

    lambda_l2=3.453537e-01,

    min_data_in_leaf=80,

    bagging_fraction=0.644553,

    feature_fraction=0.605880,

    min_child_weight=0.003471,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.05,

    n_estimators=400,

    max_depth=8,

    num_leaves=128,

    scale_pos_weight=1.850452,

    lambda_l1=17.345933,

    lambda_l2=3.453537e-01,

    min_data_in_leaf=80,

    bagging_fraction=0.644553,

    feature_fraction=0.605880,

    min_child_weight=0.003471,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opt4-init.csv', sep=',', encoding='utf-8', index=False)
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.10,

    n_estimators=2410,

    max_depth=4,

    num_leaves=3,

    scale_pos_weight=1.0,

    lambda_l1=7.080496,

    lambda_l2=0.0,

    min_data_in_leaf=1,

    bagging_fraction=0.729976,

    feature_fraction=0.890131,

    min_child_weight=2.045646e-04,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.10,

    n_estimators=2410,

    max_depth=4,

    num_leaves=3,

    scale_pos_weight=1.0,

    lambda_l1=7.080496,

    lambda_l2=0.0,

    min_data_in_leaf=1,

    bagging_fraction=0.729976,

    feature_fraction=0.890131,

    min_child_weight=2.045646e-04,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opt5-init.csv', sep=',', encoding='utf-8', index=False)
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=2156,

    max_depth=8,

    num_leaves=78,

    scale_pos_weight=2.967285,

    lambda_l1=11.413391,

    lambda_l2=0.0,

    min_data_in_leaf=33,

    bagging_fraction=0.910421,

    feature_fraction=0.711578,

    min_child_weight=1.982752e+02,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=2156,

    max_depth=8,

    num_leaves=78,

    scale_pos_weight=2.967285,

    lambda_l1=11.413391,

    lambda_l2=0.0,

    min_data_in_leaf=33,

    bagging_fraction=0.910421,

    feature_fraction=0.711578,

    min_child_weight=1.982752e+02,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opt6-init.csv', sep=',', encoding='utf-8', index=False)
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=1262,

    max_depth=12,

    num_leaves=3798,

    scale_pos_weight=4.530422,

    lambda_l1=0.002142,

    lambda_l2=0.0,

    min_data_in_leaf=18,

    bagging_fraction=0.701903,

    feature_fraction=0.766920,

    min_child_weight=3.379132e+02,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=1262,

    max_depth=12,

    num_leaves=3798,

    scale_pos_weight=4.530422,

    lambda_l1=0.002142,

    lambda_l2=0.0,

    min_data_in_leaf=18,

    bagging_fraction=0.701903,

    feature_fraction=0.766920,

    min_child_weight=3.379132e+02,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opt7-init.csv', sep=',', encoding='utf-8', index=False)
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=2977,

    max_depth=4,

    num_leaves=14,

    scale_pos_weight=7.995956,

    lambda_l1=16.966428,

    lambda_l2=2.261177,

    min_data_in_leaf=214,

    bagging_fraction=0.707032,

    feature_fraction=0.655838,

    min_child_weight=1.981114e+02,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=2977,

    max_depth=4,

    num_leaves=14,

    scale_pos_weight=7.995956,

    lambda_l1=16.966428,

    lambda_l2=2.261177,

    min_data_in_leaf=214,

    bagging_fraction=0.707032,

    feature_fraction=0.655838,

    min_child_weight=1.981114e+02,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opt8-init.csv', sep=',', encoding='utf-8', index=False)
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.10,

    n_estimators=1807,

    max_depth=8,

    num_leaves=3,

    scale_pos_weight=0.872894,

    lambda_l1=0.037629,

    lambda_l2=0.0,

    min_data_in_leaf=303,

    bagging_fraction=0.951395,

    feature_fraction=0.882096,

    min_child_weight=5.590000e-08,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.10,

    n_estimators=1807,

    max_depth=8,

    num_leaves=3,

    scale_pos_weight=0.872894,

    lambda_l1=0.037629,

    lambda_l2=0.0,

    min_data_in_leaf=303,

    bagging_fraction=0.951395,

    feature_fraction=0.882096,

    min_child_weight=5.590000e-08,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opt9-init.csv', sep=',', encoding='utf-8', index=False)
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.05,

    n_estimators=400,

    max_depth=8,

    num_leaves=128,

    scale_pos_weight=1.850452,

    lambda_l1=17.345933,

    lambda_l2=3.453537e-01,

    min_data_in_leaf=80,

    bagging_fraction=0.644553,

    feature_fraction=0.605880,

    min_child_weight=0.003471,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_train_tf, appxf_split_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.05,

    n_estimators=400,

    max_depth=8,

    num_leaves=128,

    scale_pos_weight=1.850452,

    lambda_l1=17.345933,

    lambda_l2=3.453537e-01,

    min_data_in_leaf=80,

    bagging_fraction=0.644553,

    feature_fraction=0.605880,

    min_child_weight=0.003471,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_train_tf,appxf_full_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opt10-init.csv', sep=',', encoding='utf-8', index=False)
hparam_eval_rus.sort_values(by='ROC_AUC', ascending=False).head(10)
rus_1 = RandomUnderSampler(random_state=3)

appxf_split_rus_train_tf, appxf_split_rus_train_lbl = rus_1.fit_sample(appxf_split_train_tf, appxf_split_train_lbl)

appxf_split_rus_train_lbl.value_counts()
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=3706,

    max_depth=4,

    num_leaves=12,

    scale_pos_weight=1.359928,

    lambda_l1=0.0,

    lambda_l2=6.300000e-09,

    min_data_in_leaf=1,

    bagging_fraction=0.505582,

    feature_fraction=0.537263,

    min_child_weight=9.291917,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_rus_train_tf, appxf_split_rus_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
rus_2 = RandomUnderSampler(random_state=3)

appxf_full_rus_train_tf, appxf_full_rus_train_lbl = rus_2.fit_sample(appxf_full_train_tf,appxf_full_train_lbl)

appxf_full_rus_train_lbl.value_counts()
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=3706,

    max_depth=4,

    num_leaves=12,

    scale_pos_weight=1.359928,

    lambda_l1=0.0,

    lambda_l2=6.300000e-09,

    min_data_in_leaf=1,

    bagging_fraction=0.505582,

    feature_fraction=0.537263,

    min_child_weight=9.291917,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_rus_train_tf, appxf_full_rus_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_optr1-rus.csv', sep=',', encoding='utf-8', index=False)
del appxf_split_rus_train_tf

del appxf_full_rus_train_tf

gc.collect()
rus_1 = RandomUnderSampler(random_state=3)

appxf_split_rus_train_tf, appxf_split_rus_train_lbl = rus_1.fit_sample(appxf_split_train_tf, appxf_split_train_lbl)

appxf_split_rus_train_lbl.value_counts()
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=3590,

    max_depth=4,

    num_leaves=12,

    scale_pos_weight=0.830589,

    lambda_l1=0.0,

    lambda_l2=1.320000e-08,

    min_data_in_leaf=12,

    bagging_fraction=0.527756,

    feature_fraction=0.577792,

    min_child_weight=0.058829,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_rus_train_tf, appxf_split_rus_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
rus_2 = RandomUnderSampler(random_state=3)

appxf_full_rus_train_tf, appxf_full_rus_train_lbl = rus_2.fit_sample(appxf_full_train_tf,appxf_full_train_lbl)

appxf_full_rus_train_lbl.value_counts()
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=3590,

    max_depth=4,

    num_leaves=12,

    scale_pos_weight=0.830589,

    lambda_l1=0.0,

    lambda_l2=1.320000e-08,

    min_data_in_leaf=12,

    bagging_fraction=0.527756,

    feature_fraction=0.577792,

    min_child_weight=0.058829,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_rus_train_tf, appxf_full_rus_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_optr2-rus.csv', sep=',', encoding='utf-8', index=False)
del appxf_split_rus_train_tf

del appxf_full_rus_train_tf

gc.collect()
hparam_eval_smo.sort_values(by='ROC_AUC', ascending=False).head(10)
smo_1 = SMOTE(sampling_strategy='minority', n_jobs=core_nb, random_state=3)

appxf_split_smo_train_tf, appxf_split_smo_train_lbl = smo_1.fit_sample(appxf_split_train_tf, appxf_split_train_lbl)

appxf_split_smo_train_lbl.value_counts()
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.10,

    n_estimators=3056,

    max_depth=4,

    num_leaves=4,

    scale_pos_weight=1.097013,

    lambda_l1=2.274953e+00,

    lambda_l2=0.0,

    min_data_in_leaf=240,

    bagging_fraction=0.758552,

    feature_fraction=0.798497,

    min_child_weight=6.493000e-07,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_smo_train_tf, appxf_split_smo_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
smo_2 = SMOTE(sampling_strategy='minority', n_jobs=core_nb, random_state=3)

appxf_full_smo_train_tf, appxf_full_smo_train_lbl = smo_2.fit_sample(appxf_full_train_tf,appxf_full_train_lbl)

appxf_full_smo_train_lbl.value_counts()
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.10,

    n_estimators=3056,

    max_depth=4,

    num_leaves=4,

    scale_pos_weight=1.097013,

    lambda_l1=2.274953e+00,

    lambda_l2=0.0,

    min_data_in_leaf=240,

    bagging_fraction=0.758552,

    feature_fraction=0.798497,

    min_child_weight=6.493000e-07,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_smo_train_tf, appxf_full_smo_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opts1-smo.csv', sep=',', encoding='utf-8', index=False)
del appxf_split_smo_train_tf

del appxf_full_smo_train_tf

gc.collect()
smo_1 = SMOTE(sampling_strategy='minority', n_jobs=core_nb, random_state=3)

appxf_split_smo_train_tf, appxf_split_smo_train_lbl = smo_1.fit_sample(appxf_split_train_tf, appxf_split_train_lbl)

appxf_split_smo_train_lbl.value_counts()
clf_lgbm_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=3244,

    max_depth=12,

    num_leaves=48,

    scale_pos_weight=0.667928,

    lambda_l1=2.500000e-09,

    lambda_l2=0.0,

    min_data_in_leaf=103,

    bagging_fraction=0.742400,

    feature_fraction=0.609537,

    min_child_weight=2.520000e-08,

    importance_type='split'

)

clf_lgbm_1.fit(appxf_split_smo_train_tf, appxf_split_smo_train_lbl)

pred_prob_1 = clf_lgbm_1.predict_proba(appxf_split_test_tf)

round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
smo_2 = SMOTE(sampling_strategy='minority', n_jobs=core_nb, random_state=3)

appxf_full_smo_train_tf, appxf_full_smo_train_lbl = smo_2.fit_sample(appxf_full_train_tf,appxf_full_train_lbl)

appxf_full_smo_train_lbl.value_counts()
clf_lgbm_1b = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=3244,

    max_depth=12,

    num_leaves=48,

    scale_pos_weight=0.667928,

    lambda_l1=2.500000e-09,

    lambda_l2=0.0,

    min_data_in_leaf=103,

    bagging_fraction=0.742400,

    feature_fraction=0.609537,

    min_child_weight=2.520000e-08,

    importance_type='split'

)

clf_lgbm_1b.fit(appxf_full_smo_train_tf, appxf_full_smo_train_lbl)

pred_prob_1b = clf_lgbm_1b.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1b[:,1]})

result_.to_csv('../working/pred_lgbm_opts2-smo.csv', sep=',', encoding='utf-8', index=False)
del appxf_split_smo_train_tf

del appxf_full_smo_train_tf

gc.collect()


clf_lgbm_best_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=2156,

    max_depth=8,

    num_leaves=78,

    scale_pos_weight=2.967285,

    lambda_l1=11.413391,

    lambda_l2=0.0,

    min_data_in_leaf=33,

    bagging_fraction=0.910421,

    feature_fraction=0.711578,

    min_child_weight=1.982752e+02,

    importance_type='split',

    metric='auc'

)

'''

clf_lgbm_best_1 = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    n_estimators=100

)

'''

clf_lgbm_best_1.fit(appxf_split_train_tf, appxf_split_train_lbl, early_stopping_rounds=20, eval_metric='auc', eval_set=[(appxf_split_test_tf, appxf_split_test_lbl)])

#pred_prob_1 = clf_lgbm_best_1.predict_proba(appxf_split_test_tf)

#round(roc_auc_score(appxf_split_test_lbl, pred_prob_1[:,1]),6)
pred_prob_1 = clf_lgbm_best_1.predict_proba(appxf_full_test_tf)

result_ = pd.DataFrame({'SK_ID_CURR':apptest_xfeat.SK_ID_CURR,'TARGET':pred_prob_1[:,1]})

result_.to_csv('../working/pred_lgbm_best_es-init.csv', sep=',', encoding='utf-8', index=False)
clf_lgbm_best_s = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=2156,

    max_depth=8,

    num_leaves=78,

    scale_pos_weight=2.967285,

    lambda_l1=11.413391,

    lambda_l2=0.0,

    min_data_in_leaf=33,

    bagging_fraction=0.910421,

    feature_fraction=0.711578,

    min_child_weight=1.982752e+02,

    importance_type='split'

)

clf_lgbm_best_s.fit(appxf_full_train_tf,appxf_full_train_lbl)
feat_imp_s = clf_lgbm_best_s.feature_importances_

feature_importances_s = pd.DataFrame({'split_importances':feat_imp_s, 'features':feature_names})

feature_importances_s.sort_values(by='split_importances', ascending=False).head(20)
feat_nb = 50



sns.set_style('darkgrid')

plt.figure(figsize=(10,20), dpi=300)

#cmap_tmp = sns.cubehelix_palette(start=2.8, rot=0, n_colors=feat_nb, reverse=True)

cmap_tmp = sns.diverging_palette(133, 20, l=60, s=99, sep=10, n=feat_nb, center="dark")

cmap = sns.set_palette(cmap_tmp,feat_nb)



sns.barplot(y='features', x='split_importances', data=feature_importances_s.sort_values(by='split_importances', ascending=False).head(feat_nb), color=cmap)



plt.xlabel('Nb of time features are involved in splitting', fontsize=15, style='italic', color='#888888', labelpad=20)

plt.ylabel('Features', fontsize=15, style='italic', color='#888888', labelpad=20)

plt.xticks(fontsize=12, fontweight='bold', rotation=0)

plt.yticks(fontsize=12, fontweight='bold')

plt.title('-- Feature Importance (split) --', fontweight='bold', fontsize=20, pad=20)

filename_FI = 'Feature_importance.png'

plt.savefig(filename_FI, dpi=150)

plt.show()
clf_lgbm_best_g = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=2156,

    max_depth=8,

    num_leaves=78,

    scale_pos_weight=2.967285,

    lambda_l1=11.413391,

    lambda_l2=0.0,

    min_data_in_leaf=33,

    bagging_fraction=0.910421,

    feature_fraction=0.711578,

    min_child_weight=1.982752e+02,

    importance_type='gain'

)

clf_lgbm_best_g.fit(appxf_full_train_tf,appxf_full_train_lbl)
feat_imp_g = clf_lgbm_best_g.feature_importances_

feature_importances_g = pd.DataFrame({'gain_importances':feat_imp_g, 'features':feature_names})

feature_importances_g.sort_values(by='gain_importances', ascending=False).head(20)
feat_nb = 50



sns.set_style('darkgrid')

plt.figure(figsize=(10,20), dpi=300)

#cmap_tmp = sns.cubehelix_palette(start=2.8, rot=0, n_colors=feat_nb, reverse=True)

cmap_tmp = sns.diverging_palette(133, 20, l=60, s=99, sep=10, n=feat_nb, center="dark")

cmap = sns.set_palette(cmap_tmp,feat_nb)



sns.barplot(y='features', x='gain_importances', data=feature_importances_g.sort_values(by='gain_importances', ascending=False).head(feat_nb), color=cmap)



plt.xlabel('Average training loss reduction gained when using a feature for splitting.', fontsize=15, style='italic', color='#888888', labelpad=20)

plt.ylabel('Features', fontsize=15, style='italic', color='#888888', labelpad=20)

plt.xticks(fontsize=12, fontweight='bold', rotation=0)

plt.yticks(fontsize=12, fontweight='bold')

plt.title('-- Feature Importance (gain) --', fontweight='bold', fontsize=20, pad=20)

filename_FI = 'Feature_importance.png'

plt.savefig(filename_FI, dpi=150)

plt.show()
appxf_split_train_tf_df = pd.DataFrame(appxf_split_train_tf, columns=feature_names)

appxf_split_test_tf_df = pd.DataFrame(appxf_split_test_tf, columns=feature_names)
params = {

    'n_jobs'           : core_nb, 

    'random_state'     : 3,

    'learning_rate'    : 0.01,

    'n_estimators'     : 2156,

    'max_depth'        : 8,

    'num_leaves'       : 78,

    'scale_pos_weight' : 2.967285,

    'lambda_l1'        : 11.413391,

    'lambda_l2'        : 0.0,

    'min_data_in_leaf' : 33,

    'bagging_fraction' : 0.910421,

    'feature_fraction' : 0.711578,

    'min_child_weight' : 1.982752e+02,

    'metric'           : 'auc',

    'objective'        : 'binary'

}
train_set=lightgbm.Dataset(appxf_split_train_tf_df, label=appxf_split_train_lbl)

valid_set=lightgbm.Dataset(appxf_split_test_tf_df, label=appxf_split_test_lbl)
# load JS visualization code to notebook

shap.initjs()
mdl = lightgbm.train(params,

                     train_set,

                     5000,

                     valid_sets=valid_set,

                     early_stopping_rounds= 50,

                     verbose_eval= 100

                    )
#shap_values = shap.TreeExplainer(mdl).shap_values(appxf_split_test_tf)

explainer = shap.TreeExplainer(mdl)

shap_values = explainer.shap_values(appxf_split_test_tf_df)
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

shap.force_plot(explainer.expected_value[0], shap_values[0][:100,:], appxf_split_test_tf_df.iloc[:100,:])
# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)

shap.force_plot(explainer.expected_value[0], shap_values[0][2,:], appxf_split_test_tf_df.iloc[2,:])
shap.force_plot(explainer.expected_value[1], shap_values[1][2,:], appxf_split_test_tf_df.iloc[2,:])
shap.summary_plot(shap_values[0], appxf_split_test_tf_df, plot_type="bar")
shap.summary_plot(shap_values[1], appxf_split_test_tf_df, plot_type="bar")
shap.summary_plot(shap_values, appxf_split_test_tf_df, plot_type="bar")
shap.summary_plot(shap_values[0], appxf_split_test_tf_df)
shap.summary_plot(shap_values[1], appxf_split_test_tf_df)
shap.dependence_plot('CREDIT_TERM', shap_values[0], appxf_split_test_tf_df)
shap.dependence_plot('AGE', shap_values[0], appxf_split_test_tf_df)
shap.dependence_plot('CREDIT_TERM', shap_values[0], appxf_split_test_tf_df, interaction_index='AGE')
shap.dependence_plot('AGE', shap_values[0], appxf_split_test_tf_df, interaction_index='CREDIT_TERM')
shap.dependence_plot('CREDIT_TERM', shap_values[0], appxf_split_test_tf_df, interaction_index='bu_s_Remaining_days')
shap.dependence_plot('CREDIT_TERM', shap_values[0], appxf_split_test_tf_df, interaction_index='x1_House / apartment')
shap.waterfall_plot(explainer.expected_value[0], shap_values[0][2,:], appxf_split_test_tf_df.iloc[2,:], max_display=20)
shap.waterfall_plot(explainer.expected_value[1], shap_values[1][2,:], appxf_split_test_tf_df.iloc[2,:], max_display=100)
shap.decision_plot(explainer.expected_value[0], shap_values[0][7,:], appxf_split_test_tf_df.iloc[7,:])
shap.decision_plot(explainer.expected_value[0], shap_values[0][10,:], appxf_split_test_tf_df.iloc[10,:])
shap.decision_plot(explainer.expected_value[0], shap_values[0][15,:], appxf_split_test_tf_df.iloc[15,:])
shap.decision_plot(explainer.expected_value[0], shap_values[0][:20,:], appxf_split_test_tf_df.iloc[:20,:])
shap.decision_plot(explainer.expected_value[0], shap_values[0][:200,:], appxf_split_test_tf_df.iloc[:200,:])
clf_lgbm_best_g = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=2156,

    max_depth=8,

    num_leaves=78,

    scale_pos_weight=2.967285,

    lambda_l1=11.413391,

    lambda_l2=0.0,

    min_data_in_leaf=33,

    bagging_fraction=0.910421,

    feature_fraction=0.711578,

    min_child_weight=1.982752e+02,

    importance_type='gain'

)

clf_lgbm_best_g.fit(appxf_split_train_tf_df,appxf_split_train_lbl)
shap.partial_dependence_plot(0, clf_lgbm_best_g.predict_proba, appxf_split_train_tf_df.iloc[:10000,:], model_expected_value=explainer.expected_value[0], shap_values=shap_values[0])
shap.partial_dependence_plot(0, clf_lgbm_best_g.predict, appxf_split_train_tf_df.iloc[:10000,:], model_expected_value=explainer.expected_value[0], shap_values=shap_values[0], ice=True, npoints=100)
# Visiblement issue suivante : https://github.com/microsoft/LightGBM/issues/3014

# En attente de correction / ajustement

'''

features = ['CREDIT_TERM', 'AGE']

plot_partial_dependence(clf_lgbm_best_g, appxf_split_train_tf_df, features, n_jobs=4, grid_resolution=20)

print("done in {:.3f}s".format(time() - tic))

fig = plt.gcf()

fig.suptitle('Partial dependence of house value on non-location features\n'

             'for the California housing dataset, with Gradient Boosting')

fig.subplots_adjust(wspace=0.4, hspace=0.3)

'''
appxf_full_train_tf_df = pd.DataFrame(appxf_full_train_tf, columns=feature_names)
clf_lgbm_best_g = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=2156,

    max_depth=8,

    num_leaves=78,

    scale_pos_weight=2.967285,

    lambda_l1=11.413391,

    lambda_l2=0.0,

    min_data_in_leaf=33,

    bagging_fraction=0.910421,

    feature_fraction=0.711578,

    min_child_weight=1.982752e+02,

    importance_type='gain'

)

clf_lgbm_best_g.fit(appxf_full_train_tf_df,appxf_full_train_lbl)
pdp2plot = pdp.pdp_isolate(dataset=appxf_full_train_tf_df, model=clf_lgbm_best_g, model_features=appxf_full_train_tf_df.columns.tolist(), feature='CREDIT_TERM')
pdp.pdp_plot(pdp2plot, 'CREDIT_TERM', plot_lines=True, frac_to_plot = 0.05, plot_pts_dist=True)
pdp2plot = pdp.pdp_isolate(dataset=appxf_full_train_tf_df, model=clf_lgbm_best_g, model_features=appxf_full_train_tf_df.columns.tolist(), feature='AGE')
pdp.pdp_plot(pdp2plot, 'AGE', plot_lines=True, frac_to_plot = 0.05, plot_pts_dist=True)
appxf_full_train_tf_df = pd.DataFrame(appxf_full_train_tf, columns=feature_names)

appxf_full_train_tf_df_all = pd.concat([appxf_full_train_tf_df, appxf_full_train_lbl], axis=1, sort=False)
clf_lgbm_best_g = LGBMClassifier(

    n_jobs=core_nb, 

    random_state=3,

    learning_rate=0.01,

    n_estimators=2156,

    max_depth=8,

    num_leaves=78,

    scale_pos_weight=2.967285,

    lambda_l1=11.413391,

    lambda_l2=0.0,

    min_data_in_leaf=33,

    bagging_fraction=0.910421,

    feature_fraction=0.711578,

    min_child_weight=1.982752e+02,

    importance_type='gain'

)

clf_lgbm_best_g.fit(appxf_full_train_tf_df,appxf_full_train_lbl)
LTE = LimeTabularExplainer(appxf_full_train_tf_df.values, feature_names=appxf_full_train_tf_df.columns, class_names=[0,1], discretize_continuous=False)
expl = LTE.explain_instance(appxf_full_train_tf_df.iloc[3], clf_lgbm_best_g.predict_proba, num_samples=1000)
expl.show_in_notebook(show_table=True, show_all=True)
expl.as_pyplot_figure()

plt.tight_layout

plt.show()
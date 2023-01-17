import os
import gc
import pandas as pd
import numpy as np
import lightgbm as lgbm
import featuretools as ft
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')
"""
I created this kernel to test the Featuretools framework to perform automated feature engineering
https://docs.featuretools.com/index.html

I got inspired by the following Towards Data Science post
https://towardsdatascience.com/automated-feature-engineering-in-python-99baf11cc219

and I borrowed some code from James Shepherd's kernel --- LightGBM with weighted averages ---
that was of big help
https://www.kaggle.com/shep312/lightgbm-with-weighted-averages-dropout-787

I'm new to Data Science and machine learning so fork this kernel at your own risk ;)
"""
""" 
Sample Size set to 25k to avoid Kaggle's Kernel execution timeout (6 hours)
Setting this variable to None will process all lines from the input files, it took
more or less 5 hours on my machine (i7 processor with 16GB RAM)
"""

#sample_size = None
sample_size = 25000 

""" Load and process inputs """
input_dir = os.path.join(os.pardir, 'input')
print('Input files:\n{}'.format(os.listdir(input_dir)))
print('Loading data sets...')
app_train_df = pd.read_csv(os.path.join(input_dir,'application_train.csv'), nrows=sample_size)
app_test_df = pd.read_csv(os.path.join(input_dir,'application_test.csv'))
prev_app_df = pd.read_csv(os.path.join(input_dir,'previous_application.csv'), nrows=sample_size)
bureau_df = pd.read_csv(os.path.join(input_dir,'bureau.csv'), nrows=sample_size)
bureau_balance_df = pd.read_csv(os.path.join(input_dir,'bureau_balance.csv'), nrows=sample_size)
installments_df = pd.read_csv(os.path.join(input_dir,'installments_payments.csv'), nrows=sample_size)
cc_balance_df = pd.read_csv(os.path.join(input_dir,'credit_card_balance.csv'), nrows=sample_size)
pos_balance_df = pd.read_csv(os.path.join(input_dir,'POS_CASH_balance.csv'), nrows=sample_size)

print('Data loaded.\nMain application training data set shape = {}'.format(app_train_df.shape))
print('Main application test data set shape = {}'.format(app_test_df.shape))
print('Positive target proportion = {:.2f}'.format(app_train_df['TARGET'].mean()))
# Merge the datasets into a single one for training
app_both = pd.concat([app_train_df, app_test_df])
# A lot of the continuous days variables have integers as missing value indicators.
prev_app_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
prev_app_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)
prev_app_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
prev_app_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
prev_app_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
#Add new features
# Amount loaned relative to salary
app_both['LOAN_INCOME_RATIO'] = app_both['AMT_CREDIT'] / app_both['AMT_INCOME_TOTAL']
app_both['ANNUITY_INCOME_RATIO'] = app_both['AMT_ANNUITY'] / app_both['AMT_INCOME_TOTAL']
    
# Number of overall payments (I think!)
app_both['ANNUITY LENGTH'] = app_both['AMT_CREDIT'] / app_both['AMT_ANNUITY']
    
# Social features
app_both['WORKING_LIFE_RATIO'] = app_both['DAYS_EMPLOYED'] / app_both['DAYS_BIRTH']
app_both['INCOME_PER_FAM'] = app_both['AMT_INCOME_TOTAL'] / app_both['CNT_FAM_MEMBERS']
app_both['CHILDREN_RATIO'] = app_both['CNT_CHILDREN'] / app_both['CNT_FAM_MEMBERS']
# Create new entityset
es = ft.EntitySet(id='home_credit_default_risk')

applications_var_types = {'FLAG_CONT_MOBILE': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_10': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_11': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_12': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_13': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_14': ft.variable_types.Boolean, 
                          'FLAG_DOCUMENT_15': ft.variable_types.Boolean, 
                          'FLAG_DOCUMENT_16': ft.variable_types.Boolean, 
                          'FLAG_DOCUMENT_17': ft.variable_types.Boolean, 
                          'FLAG_DOCUMENT_18': ft.variable_types.Boolean, 
                          'FLAG_DOCUMENT_19': ft.variable_types.Boolean, 
                          'FLAG_DOCUMENT_2': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_20': ft.variable_types.Boolean, 
                          'FLAG_DOCUMENT_21': ft.variable_types.Boolean, 
                          'FLAG_DOCUMENT_3': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_4': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_5': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_6': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_7': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_8': ft.variable_types.Boolean,
                          'FLAG_DOCUMENT_9': ft.variable_types.Boolean,
                          'FLAG_EMAIL': ft.variable_types.Boolean,
                          'FLAG_EMP_PHONE': ft.variable_types.Boolean,
                          'FLAG_MOBIL': ft.variable_types.Boolean,
                          'FLAG_PHONE': ft.variable_types.Boolean,
                          'FLAG_WORK_PHONE': ft.variable_types.Boolean,
                          'LIVE_CITY_NOT_WORK_CITY': ft.variable_types.Boolean,
                          'LIVE_REGION_NOT_WORK_REGION': ft.variable_types.Boolean,
                          'REG_CITY_NOT_LIVE_CITY': ft.variable_types.Boolean,
                          'REG_CITY_NOT_WORK_CITY': ft.variable_types.Boolean,
                          'REG_REGION_NOT_LIVE_REGION': ft.variable_types.Boolean,
                          'REG_REGION_NOT_WORK_REGION': ft.variable_types.Boolean,
                          'TARGET': ft.variable_types.Discrete}

# Create an entity from the applications (app_both) dataframe
# This dataframe already has an index
es = es.entity_from_dataframe(entity_id='applications',
                              variable_types = applications_var_types,
                              dataframe=app_both, index='SK_ID_CURR')

bureau_var_types = {'CREDIT_ACTIVE': ft.variable_types.Categorical, 
                    'CREDIT_CURRENCY': ft.variable_types.Categorical,
                    'CREDIT_TYPE': ft.variable_types.Categorical}

# Create an entity from the bureau dataframe
# This dataframe already has an index
es = es.entity_from_dataframe(entity_id='bureau', 
                              variable_types = bureau_var_types,
                              dataframe=bureau_df, index='SK_ID_BUREAU')

# Create an entity from the bureau balance dataframe
es = es.entity_from_dataframe(entity_id='bureau_balance', 
                              variable_types = {'MONTHS_BALANCE': ft.variable_types.Ordinal},
                              make_index = True,
                              dataframe=bureau_balance_df, index='bureau_balance_id')

# Create an entity from the installments dataframe
es = es.entity_from_dataframe(entity_id='installments',
                              make_index = True,
                              dataframe=installments_df, index='installment_id')

prev_app_var_types = {'NFLAG_LAST_APPL_IN_DAY': ft.variable_types.Boolean,
                      'NFLAG_INSURED_ON_APPROVAL': ft.variable_types.Boolean,
                      'SELLERPLACE_AREA': ft.variable_types.Categorical}

# Create an entity from the previous applications dataframe
es = es.entity_from_dataframe(entity_id='previous_application',
                              variable_types = prev_app_var_types,
                              make_index = True,
                              dataframe=prev_app_df, index='prev_app_id')

# Create an entity from the credit card balance dataframe
es = es.entity_from_dataframe(entity_id='cc_balance',
                              variable_types = {'MONTHS_BALANCE': ft.variable_types.Ordinal},
                              make_index = True,
                              dataframe=cc_balance_df, index='cc_balance_id')

# Create an entity from the POS Cash balance dataframe
es = es.entity_from_dataframe(entity_id='pos_balance',
                              variable_types = {'MONTHS_BALANCE': ft.variable_types.Ordinal},
                              make_index = True,
                              dataframe=pos_balance_df, index='pos_balance_id')
# Relationship between applications and credits bureau
r_applications_bureau = ft.Relationship(es['applications']['SK_ID_CURR'],
                                    es['bureau']['SK_ID_CURR'])
es = es.add_relationship(r_applications_bureau)

# Relationship between applications and credits bureau
r_applications_installment = ft.Relationship(es['applications']['SK_ID_CURR'],
                                    es['installments']['SK_ID_CURR'])
es = es.add_relationship(r_applications_installment)

# Relationship between applications and credits bureau
r_bureau_bureaubalance = ft.Relationship(es['bureau']['SK_ID_BUREAU'],
                                    es['bureau_balance']['SK_ID_BUREAU'])
es = es.add_relationship(r_bureau_bureaubalance)

# Relationship between applications and previous applications
r_applications_prev_apps = ft.Relationship(es['applications']['SK_ID_CURR'],
                                    es['previous_application']['SK_ID_CURR'])
es = es.add_relationship(r_applications_prev_apps)

# Relationship between applications and credit card balance
r_applications_cc_balance = ft.Relationship(es['applications']['SK_ID_CURR'],
                                    es['cc_balance']['SK_ID_CURR'])
es = es.add_relationship(r_applications_cc_balance)

# Relationship between applications and POS cash balance
r_applications_pos_balance = ft.Relationship(es['applications']['SK_ID_CURR'],
                                    es['pos_balance']['SK_ID_CURR'])
es = es.add_relationship(r_applications_pos_balance)

print(es)
"""
Deep Feature Synthesis (DFS) is an automated method for performing feature engineering on relational and transactional data.
https://docs.featuretools.com/automated_feature_engineering/afe.html
"""
# Create new features using specified primitives
feature_matrix, feature_defs = ft.dfs(entityset = es, target_entity = 'applications',
                                      drop_contains=['SK_ID_PREV'], max_depth=2, verbose=True)
def process_dataframe(input_df, encoder_dict=None):
    """ Process a dataframe into a form useable by LightGBM """

    # Label encode categoricals
    print('Label encoding categorical features...')
    categorical_feats = input_df.columns[input_df.dtypes == 'object']
    for feat in categorical_feats:
        encoder = LabelEncoder()
        input_df[feat] = encoder.fit_transform(input_df[feat].fillna('NULL'))
    print('Label encoding complete.')

    return input_df, categorical_feats.tolist(), encoder_dict
# Process the data set.
feature_matrix_enc, categorical_feats, encoder_dict = process_dataframe(input_df=feature_matrix)
# Separate into train and test
train_df = feature_matrix_enc[feature_matrix_enc['TARGET'].notnull()].copy()

test_df = feature_matrix_enc[feature_matrix_enc['TARGET'].isnull()].copy()
test_df.drop(['TARGET'], axis=1, inplace=True)

del feature_matrix, feature_defs, feature_matrix_enc
gc.collect()
""" Train the model """
target = train_df.pop('TARGET')

lgbm_train = lgbm.Dataset(data=train_df,
                          label=target,
                          categorical_feature=categorical_feats,
                          free_raw_data=False)
lgbm_params = {
    'boosting': 'dart',
    'application': 'binary',
    'learning_rate': 0.1,
    'min_data_in_leaf': 30,
    'num_leaves': 31,
    'max_depth': -1,
    'feature_fraction': 0.5,
    'scale_pos_weight': 2,
    'drop_rate': 0.02
}

cv_results = lgbm.cv(train_set=lgbm_train,
                     params=lgbm_params,
                     nfold=5,
                     num_boost_round=600,
                     early_stopping_rounds=50,
                     verbose_eval=20,
                     metrics=['auc'])

optimum_boost_rounds = np.argmax(cv_results['auc-mean'])
print('Optimum boost rounds = {}'.format(optimum_boost_rounds))
print('Best CV result = {}'.format(np.max(cv_results['auc-mean'])))

clf = lgbm.train(train_set=lgbm_train,
                 params=lgbm_params,
                 num_boost_round=optimum_boost_rounds)

""" Predict on test set and create submission """
y_pred = clf.predict(test_df)
out_df = pd.DataFrame({'SK_ID_CURR': test_df.index, 'TARGET': y_pred})
out_df.to_csv('submission.csv', index=False)
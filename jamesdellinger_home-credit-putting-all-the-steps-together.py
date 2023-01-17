# Imports
import numpy as np
import pandas as pd
import gc
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Display matplotlib plots inline in this notebook.
%matplotlib inline
# Make plots display well on retina displays
%config InlineBackend.figure_format = 'retina'
# Set dpi of plots displayed inline
mpl.rcParams['figure.dpi'] = 300
# Configure style of plots
plt.style.use('fivethirtyeight')
# Make plots smaller
sns.set_context('paper') 

# In order to silence a numpy deprecation warning 
# thrown when Seaborn plots a bar chart.
import warnings
warnings.filterwarnings('ignore')

# Allows the use of display() for dataframes.
from IPython.display import display
# Have all columns appear when dataframes are displayed.
pd.set_option('display.max_columns', None) 
# Have 100 rows appear when a dataframe is displayed
pd.set_option('display.max_rows', 500)
# Display dimensions whenever a dataframe is printed out.
pd.set_option('display.show_dimensions', True)

# To compute ROC AUC score
from sklearn.metrics import roc_auc_score

# For stratified K-fold cv
from sklearn.model_selection import StratifiedKFold

# LightGBM classifier
import lightgbm as lgb

# To calculate rank mean of the five sets 
# of test predictions.
from sklearn.preprocessing import MinMaxScaler

# In order to create CSV files
import csv
def replace_XNA_XAP(table):

    # Replace all values of 'XNA', 'XAP' with np.nan
    table.replace(to_replace = {'XNA': np.nan, 'XAP': np.nan}, value = None, inplace = True)
    
    return table
def preprocess_main(application_train, application_test):
    
    # Separate target data from training dataset.
    y_train = application_train['TARGET']
    X = application_train.drop('TARGET', axis = 1)

    # Combine application train and test tables for preprocessing.
    X = pd.concat([X, application_test], axis=0)

    # Several features have entries with values that can best 
    # be interpreted as np.nan:

    # Replace all entries of 365243 in 'DAYS_EMPLOYED' with nan
    X['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    
    # Replace all entries of 0 in 'DAYS_LAST_PHONE_CHANGE' with nan
    X['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)

    # Replace all entries of 'XNA' or 'XAP' in main data table with np.nan
    # (Such entries should be confined to the features 'CODE_GENDER' 
    #  and 'ORGANIZATION_TYPE'.)
    X = replace_XNA_XAP(X)

    # Two rows in training table have a value of 'Unknown' for 
    # 'NAME_FAMILY_STATUS', but no rows in test table do.
    X['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
    # Five rows in training table have a value of 'Maternity leave' for 
    # 'NAME_INCOME_TYPE', but no rows in test table do.
    X['NAME_INCOME_TYPE'].replace('Maternity leave', np.nan, inplace=True)
    # No rows in training table have -1 for 'REGION_RATING_CLIENT_W_CITY' 
    # but at least one row in test table does.
    X['REGION_RATING_CLIENT_W_CITY'].replace(-1, np.nan, inplace=True)
    
    return X, y_train
def engineer_main_features(X):
    
    # Create aggregated features. Idea to do this taken 
    # from neptune-ml's solution at: 
    # https://github.com/neptune-ml/open-solution-home-credit/blob/solution-5/notebooks/eda-application.ipynb
    
    aggregation_recipes = [
        (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], [('AMT_ANNUITY', 'max'),
                                                  ('AMT_CREDIT', 'max'),
                                                  ('EXT_SOURCE_1', 'mean'),
                                                  ('EXT_SOURCE_2', 'mean'),
                                                  ('OWN_CAR_AGE', 'max'),
                                                  ('OWN_CAR_AGE', 'sum')]),
        (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                                ('AMT_INCOME_TOTAL', 'mean'),
                                                ('DAYS_REGISTRATION', 'mean'),
                                                ('EXT_SOURCE_1', 'mean')]),
        (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                     ('CNT_CHILDREN', 'mean'),
                                                     ('DAYS_ID_PUBLISH', 'mean')]),
        (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                               ('EXT_SOURCE_2', 'mean')]),
        (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                      ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                      ('APARTMENTS_AVG', 'mean'),
                                                      ('BASEMENTAREA_AVG', 'mean'),
                                                      ('EXT_SOURCE_1', 'mean'),
                                                      ('EXT_SOURCE_2', 'mean'),
                                                      ('EXT_SOURCE_3', 'mean'),
                                                      ('NONLIVINGAREA_AVG', 'mean'),
                                                      ('OWN_CAR_AGE', 'mean'),
                                                      ('YEARS_BUILD_AVG', 'mean')]),
        (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                                ('EXT_SOURCE_1', 'mean')]),
        (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                               ('CNT_CHILDREN', 'mean'),
                               ('CNT_FAM_MEMBERS', 'mean'),
                               ('DAYS_BIRTH', 'mean'),
                               ('DAYS_EMPLOYED', 'mean'),
                               ('DAYS_ID_PUBLISH', 'mean'),
                               ('DAYS_REGISTRATION', 'mean'),
                               ('EXT_SOURCE_1', 'mean'),
                               ('EXT_SOURCE_2', 'mean'),
                               ('EXT_SOURCE_3', 'mean')]),
    ]
    
    # Groupby categorical features, calculate the mean and or max 
    # of various numerical statistics.
    for groupby_cols, specs in aggregation_recipes:
        group_object = X.groupby(groupby_cols)
        for select, agg in specs:
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg.upper(), select)
            X = X.merge(group_object[select]
                                  .agg(agg)
                                  .reset_index()
                                  .rename(index=str,
                                          columns={select: groupby_aggregate_name})
                                  [groupby_cols + [groupby_aggregate_name]],
                                  on=groupby_cols,
                                  how='left')
            
    # Get the difference and absolute difference between two 
    # categorical features' mean and or max values of various
    # numerical statistics.
    for groupby_cols, specs in aggregation_recipes:
        for select, agg in specs:
            if agg in ['mean','median','max','min']:
                groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg.upper(), select)
                diff_name = '{}_DIFF'.format(groupby_aggregate_name)
                abs_diff_name = '{}_ABS_DIFF'.format(groupby_aggregate_name)

                X[diff_name] = X[select] - X[groupby_aggregate_name] 
                X[abs_diff_name] = np.abs(X[select] - X[groupby_aggregate_name]) 
            

    # Categorical features
    cat_feat = [
        'NAME_CONTRACT_TYPE', 
        'CODE_GENDER', 
        'FLAG_OWN_CAR', 
        'FLAG_OWN_REALTY', 
        'NAME_INCOME_TYPE', 
        'NAME_EDUCATION_TYPE', 
        'NAME_FAMILY_STATUS', 
        'NAME_HOUSING_TYPE', 
        'REGION_RATING_CLIENT',
        'REGION_RATING_CLIENT_W_CITY', 
        'WEEKDAY_APPR_PROCESS_START', 
        'ORGANIZATION_TYPE',
        'NAME_TYPE_SUITE', 
        'OCCUPATION_TYPE', 
        'WALLSMATERIAL_MODE', 
        'FONDKAPREMONT_MODE',
        'OBS_30_CNT_SOCIAL_CIRCLE',
        'DEF_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR',
        'CNT_CHILDREN',
        'CNT_FAM_MEMBERS',
        'OWN_CAR_AGE',
    ]

    # Make copies of each multi-categorical feature, and set aside 
    # those copies to be target mean encoded. 
    #
    # (Some of the original categorical features will later be 
    #  used as numerical features.)
    for feature in cat_feat:
        new_name = feature + '_CAT'
        orig_idx_of_feature = X.columns.get_loc(feature)
        X.insert(orig_idx_of_feature, new_name, X[feature])
        X[[new_name]] = X[[new_name]].apply(lambda x: x.astype('category'))

    # Set aside the categorical features that will also be used as 
    # numerical features.
    cat_feat_to_keep = [
        'CNT_CHILDREN',
        'CNT_FAM_MEMBERS',
        'OWN_CAR_AGE',
        'OBS_30_CNT_SOCIAL_CIRCLE',
        'DEF_30_CNT_SOCIAL_CIRCLE',
        'OBS_60_CNT_SOCIAL_CIRCLE',
        'DEF_60_CNT_SOCIAL_CIRCLE',
        'AMT_REQ_CREDIT_BUREAU_DAY',
        'AMT_REQ_CREDIT_BUREAU_HOUR',
        'AMT_REQ_CREDIT_BUREAU_WEEK',
        'AMT_REQ_CREDIT_BUREAU_MON',
        'AMT_REQ_CREDIT_BUREAU_QRT',
        'AMT_REQ_CREDIT_BUREAU_YEAR',
    ]


    # Drop the original categorical features that won't also be used as 
    # numerical features:
    cat_feat_to_drop = list(set(cat_feat) - set(cat_feat_to_keep))
    X.drop(cat_feat_to_drop, axis=1, inplace=True)
    
    
    # Engineer some simple binary features:
    
    # 'HAS_CHILDREN': a binary feature that indicates whether or 
    # not a borrower has one or more children.
    X['HAS_CHILDREN'] = X['CNT_CHILDREN'].map(lambda x: 1 if x > 0 else 0)

    # 'HAS_JOB': a binary feature that indicates whether or not 
    # a borrower was employed when their application was submitted.
    X['HAS_JOB'] = X['DAYS_EMPLOYED'].map(lambda x: 1 if x < 0 else 0)
    
    
    # Engineer numerical features:
    
    # Sums
    X['SUM_AMT_INCOME_TOTAL_AMT_ANNUITY'] = X['AMT_INCOME_TOTAL'] + X['AMT_ANNUITY']
    X['TOTAL_ENQUIRIES_CREDIT_BUREAU'] = X[['AMT_REQ_CREDIT_BUREAU_DAY',
                                          'AMT_REQ_CREDIT_BUREAU_HOUR',
                                          'AMT_REQ_CREDIT_BUREAU_WEEK',
                                          'AMT_REQ_CREDIT_BUREAU_MON',
                                          'AMT_REQ_CREDIT_BUREAU_QRT',
                                          'AMT_REQ_CREDIT_BUREAU_YEAR']].sum(axis=1)
    
    # Differences                                                             
    X['DIFF_AMT_CREDIT_AMT_GOODS_PRICE'] = X['AMT_CREDIT'] - X['AMT_GOODS_PRICE']
    X['DIFF_AMT_ANNUITY_AMT_GOODS_PRICE'] = X['AMT_ANNUITY'] - X['AMT_GOODS_PRICE']
    X['DIFF_AMT_INCOME_TOTAL_AMT_ANNUITY'] = X['AMT_INCOME_TOTAL'] - X['AMT_ANNUITY']
    X['CNT_ADULT_FAM_MEMBER'] = X['CNT_FAM_MEMBERS'] - X['CNT_CHILDREN']
    X['DIFF_OBS_30_CNT_SOCIAL_CIRCLE_OBS_60_CNT_SOCIAL_CIRCLE'] = X['OBS_30_CNT_SOCIAL_CIRCLE'] - X['OBS_60_CNT_SOCIAL_CIRCLE']
    X['DIFF_DEF_30_CNT_SOCIAL_CIRCLE_DEF_60_CNT_SOCIAL_CIRCLE'] = X['DEF_30_CNT_SOCIAL_CIRCLE'] - X['DEF_60_CNT_SOCIAL_CIRCLE']

    # Ratios
    X['RATIO_AMT_CREDIT_TO_AMT_ANNUITY'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
    X['RATIO_AMT_CREDIT_TO_CNT_ADULT_FAM_MEMBER'] = X['AMT_CREDIT'] / X['CNT_ADULT_FAM_MEMBER']
    X['RATIO_AMT_INCOME_TOTAL_TO_AMT_ANNUITY'] = X['AMT_INCOME_TOTAL'] / X['AMT_ANNUITY']
    X['AMT_INCOME_TOTAL_PER_ADULT_FAM_MEMBER'] = X['AMT_INCOME_TOTAL'] / X['CNT_ADULT_FAM_MEMBER']
    X['RATIO_AMT_GOODS_PRICE_TO_LIVINGAREA_AVG'] = X['AMT_GOODS_PRICE'] / X['LIVINGAREA_AVG']
    X['RATIO_AMT_GOODS_PRICE_TO_LANDAREA_AVG'] = X['AMT_GOODS_PRICE'] / X['LANDAREA_AVG']
    X['RATIO_AMT_GOODS_PRICE_TO_FLOORSMAX_AVG_AVG'] = X['AMT_GOODS_PRICE'] / X['FLOORSMAX_AVG']
    X['RATIO_AMT_GOODS_PRICE_TO_LIVINGAPARTMENTS_AVG'] = X['AMT_GOODS_PRICE'] / X['LIVINGAPARTMENTS_AVG']
    X['RATIO_AMT_GOODS_PRICE_TO_YEARS_BUILD_AVG'] = X['AMT_GOODS_PRICE'] / X['YEARS_BUILD_AVG']
    X['RATIO_AMT_GOODS_PRICE_TO_DAYS_EMPLOYED'] = X['AMT_GOODS_PRICE'] / X['DAYS_EMPLOYED']
    X['RATIO_AMT_GOODS_PRICE_TO_CNT_CHILDREN'] = X['AMT_GOODS_PRICE'] / X['CNT_CHILDREN']
    X['RATIO_AMT_GOODS_PRICE_TO_SUM_AMT_INCOME_TOTAL_AMT_ANNUITY'] = X['AMT_GOODS_PRICE'] / X['SUM_AMT_INCOME_TOTAL_AMT_ANNUITY']
    X['RATIO_AMT_ANNUITY_TO_LIVINGAREA_AVG'] = X['AMT_ANNUITY'] / X['LIVINGAREA_AVG']
    X['RATIO_AMT_ANNUITY_TO_DAYS_EMPLOYED'] = X['AMT_ANNUITY'] / X['DAYS_EMPLOYED']
    X['RATIO_AMT_ANNUITY_TO_CNT_CHILDREN'] = X['AMT_ANNUITY'] / X['CNT_CHILDREN']
    X['RATIO_AMT_ANNUITY_TO_CNT_ADULT_FAM_MEMBER'] = X['AMT_ANNUITY'] / X['CNT_ADULT_FAM_MEMBER']
    X['RATIO_EXT_SOURCE_3_TO_REGION_POPULATION_RELATIVE'] = X['EXT_SOURCE_3'] / X['REGION_POPULATION_RELATIVE']
    X['RATIO_DAYS_LAST_PHONE_CHANGE_TO_DAYS_REGISTRATION'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_REGISTRATION']
    X['PCTG_FAM_CHILDREN'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
    X['PCTG_ENQUIRIES_HOUR'] = X['AMT_REQ_CREDIT_BUREAU_HOUR'] / X['TOTAL_ENQUIRIES_CREDIT_BUREAU']
    X['PCTG_ENQUIRIES_DAY'] = X['AMT_REQ_CREDIT_BUREAU_DAY'] / X['TOTAL_ENQUIRIES_CREDIT_BUREAU']
    X['PCTG_ENQUIRIES_WEEK'] = X['AMT_REQ_CREDIT_BUREAU_WEEK'] / X['TOTAL_ENQUIRIES_CREDIT_BUREAU']
    X['PCTG_ENQUIRIES_MON'] = X['AMT_REQ_CREDIT_BUREAU_MON'] / X['TOTAL_ENQUIRIES_CREDIT_BUREAU']
    X['PCTG_ENQUIRIES_QRT'] = X['AMT_REQ_CREDIT_BUREAU_QRT'] / X['TOTAL_ENQUIRIES_CREDIT_BUREAU']
    X['PCTG_ENQUIRIES_YEAR'] = X['AMT_REQ_CREDIT_BUREAU_YEAR'] / X['TOTAL_ENQUIRIES_CREDIT_BUREAU']

    
    # The idea to make the following features comes 
    # from neptune-ml's solution:
    # https://github.com/neptune-ml/open-solution-home-credit/blob/solution-5/notebooks/eda-application.ipynb
    
    # Based on EXT_SOURCES features
    X['EXT_SOURCES_WEIGHTED_SUM'] = X['EXT_SOURCE_3'] * 5 + X['EXT_SOURCE_1'] * 3 + X['EXT_SOURCE_2'] * 1
    X['EXT_SOURCES_WEIGHTED_AVG'] = (X['EXT_SOURCE_3'] * 5 + X['EXT_SOURCE_1'] * 3 + X['EXT_SOURCE_2'] * 1) / 3
    X['EXT_SOURCES_MAX'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].max(axis=1)
    X['EXT_SOURCES_SUM'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].sum(axis=1)
    X['EXT_SOURCES_MIN'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].min(axis=1)
    X['EXT_SOURCES_MEDIAN'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].median(axis=1)
    X['EXT_SOURCES_MEAN'] = X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    
    # Ratios
    X['RATIO_AMT_CREDIT_TO_AMT_GOODS_PRICE'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
    X['RATIO_AMT_CREDIT_TO_AMT_INCOME_TOTAL'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
    X['RATIO_AMT_CREDIT_TO_CNT_FAM_MEMBERS'] = X['AMT_CREDIT'] / X['CNT_FAM_MEMBERS']
    X['RATIO_AMT_CREDIT_TO_CNT_CHILDREN'] = X['AMT_CREDIT'] / (1 + X['CNT_CHILDREN'])
    X['RATIO_AMT_INCOME_TOTAL_TO_AMT_CREDIT'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
    X['RATIO_AMT_INCOME_TOTAL_TO_CNT_CHILDREN'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
    X['RATIO_AMT_ANNUITY_TO_AMT_INCOME_TOTAL'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
    X['RATIO_AMT_ANNUITY_AMT_CREDIT'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
    X['RATIO_CHILDREN_TO_ADULTS'] = X['CNT_CHILDREN'] / X['CNT_ADULT_FAM_MEMBER']
    X['RATIO_OWN_CAR_AGE_TO_DAYS_BIRTH'] = X['OWN_CAR_AGE'] / X['DAYS_BIRTH']
    X['RATIO_OWN_CAR_AGE_TO_DAYS_EMPLOYED'] = X['OWN_CAR_AGE'] / X['DAYS_EMPLOYED']
    X['RATIO_DAYS_LAST_PHONE_CHANGE_TO_DAYS_BIRTH'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_BIRTH']
    X['RATIO_DAYS_LAST_PHONE_CHANGE_TO_DAYS_EMPLOYED'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_EMPLOYED']
    X['PCTG_DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
    
    # Binary categorical features
    X['LONG_EMPLOYMENT'] = (X['DAYS_EMPLOYED'] < -2000).astype(int)
    X['RETIREMENT_AGE'] = (X['DAYS_BIRTH'] < -14000).astype(int)


    # The idea to create these features comes from Olivier's kernel:
    # https://www.kaggle.com/ogrellier/home-credit-hyperopt-optimization
    X['EXT_SOURCES_PROD'] = X['EXT_SOURCE_1'] * X['EXT_SOURCE_2'] * X['EXT_SOURCE_3']
    X['RATIO_AMT_ANNUITY_TO_AMT_INCOME_TOTAL'] = X['AMT_ANNUITY'] / (1 + X['AMT_INCOME_TOTAL'])
    X['RATIO_AMT_CREDIT_TO_AMT_GOODS_PRICE'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']

    return X
def engineer_bureau_features(X, bureau):
    
    # 'COUNT_BUREAU_LOANS_(BUREAU)': Number of bureau loans 
    # each borrower has.
    count_bureau_loans_df = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby(['SK_ID_CURR'], as_index=False).count()
    count_bureau_loans_df = count_bureau_loans_df.rename(index=str, columns = {'SK_ID_BUREAU': 'COUNT_BUREAU_LOANS_(BUREAU)'})
    # Join to main dataframe
    X = X.join(count_bureau_loans_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    del count_bureau_loans_df
    gc.collect()
    
    
    # Engineer some new numerical features
    bureau_to_agg = bureau.copy()
    
    # Days with Days
    bureau_to_agg['DIFF_DAYS_CREDIT_ENDDATE_DAYS_CREDIT'] = bureau_to_agg['DAYS_CREDIT_ENDDATE'] - bureau_to_agg['DAYS_CREDIT']
    bureau_to_agg['DIFF_DAYS_CREDIT_UPDATE_DAYS_CREDIT'] = bureau_to_agg['DAYS_CREDIT_UPDATE'] - bureau_to_agg['DAYS_CREDIT']
    bureau_to_agg['RATIO_CREDIT_DAY_OVERDUE_TO_90_DAYS'] = bureau_to_agg['CREDIT_DAY_OVERDUE'] / 90
    
    # Days with cnt prolong
    bureau_to_agg['RATIO_CREDIT_DAY_OVERDUE_TO_CNT_CREDIT_PROLONG'] = bureau_to_agg['CREDIT_DAY_OVERDUE'] / bureau_to_agg['CNT_CREDIT_PROLONG']
    
    # Credit amounts with cnt prolong
    bureau_to_agg['RATIO_AMT_CREDIT_SUM_OVERDUE_TO_CNT_CREDIT_PROLONG'] = bureau_to_agg['AMT_CREDIT_SUM_OVERDUE'] / bureau_to_agg['CNT_CREDIT_PROLONG']
    bureau_to_agg['RATIO_AMT_CREDIT_MAX_OVERDUE_TO_CNT_CREDIT_PROLONG'] = bureau_to_agg['AMT_CREDIT_MAX_OVERDUE'] / bureau_to_agg['CNT_CREDIT_PROLONG']

    # Credit amounts with credit amounts
    bureau_to_agg['DIFF_AMT_CREDIT_SUM_AMT_CREDIT_SUM_DEBT'] = bureau_to_agg['AMT_CREDIT_SUM'] - bureau_to_agg['AMT_CREDIT_SUM_DEBT']
    bureau_to_agg['RATIO_AMT_CREDIT_SUM_TO_AMT_CREDIT_SUM_DEBT'] = bureau_to_agg['AMT_CREDIT_SUM'] / bureau_to_agg['AMT_CREDIT_SUM_DEBT']
    bureau_to_agg['DIFF_AMT_CREDIT_SUM_AMT_CREDIT_SUM_OVERDUE'] = bureau_to_agg['AMT_CREDIT_SUM'] - bureau_to_agg['AMT_CREDIT_SUM_OVERDUE']
    bureau_to_agg['RATIO_AMT_CREDIT_SUM_TO_AMT_CREDIT_SUM_OVERDUE'] = bureau_to_agg['AMT_CREDIT_SUM'] / bureau_to_agg['AMT_CREDIT_SUM_OVERDUE']
    bureau_to_agg['DIFF_AMT_CREDIT_SUM_LIMIT_AMT_CREDIT_SUM_DEBT'] = bureau_to_agg['AMT_CREDIT_SUM_LIMIT'] - bureau_to_agg['AMT_CREDIT_SUM_DEBT']
    bureau_to_agg['RATIO_AMT_CREDIT_SUM_DEBT_TO_AMT_CREDIT_SUM_LIMIT'] = bureau_to_agg['AMT_CREDIT_SUM_DEBT'] / bureau_to_agg['AMT_CREDIT_SUM_LIMIT']
    
    # Build aggregations based on the numerical features.
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
        'DAYS_ENDDATE_FACT': ['min', 'max', 'mean'],
        'CREDIT_DAY_OVERDUE': ['min', 'max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['min', 'max', 'mean'],
        'AMT_CREDIT_SUM': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['min', 'max', 'mean'],
        'AMT_CREDIT_SUM_LIMIT': ['min', 'max', 'mean', 'sum'],
        'AMT_ANNUITY': ['min', 'max', 'mean', 'sum'],
        'CNT_CREDIT_PROLONG': ['min', 'max', 'mean', 'sum'],
        
        # Days with Days
        'DIFF_DAYS_CREDIT_ENDDATE_DAYS_CREDIT': ['min', 'max', 'mean'],
        'DIFF_DAYS_CREDIT_UPDATE_DAYS_CREDIT': ['min', 'max', 'mean'],
        'RATIO_CREDIT_DAY_OVERDUE_TO_90_DAYS': ['min', 'max', 'mean'],
                                    
        # Days with cnt prolong
        'RATIO_CREDIT_DAY_OVERDUE_TO_CNT_CREDIT_PROLONG': ['min', 'max', 'mean'],
                                                
        # Credit amounts with cnt prolong
        'RATIO_AMT_CREDIT_SUM_OVERDUE_TO_CNT_CREDIT_PROLONG': ['min', 'max', 'mean'],
        'RATIO_AMT_CREDIT_MAX_OVERDUE_TO_CNT_CREDIT_PROLONG': ['min', 'max', 'mean'],
                                        
        # Credit amounts with credit amounts
        'DIFF_AMT_CREDIT_SUM_AMT_CREDIT_SUM_DEBT': ['min', 'max', 'mean'],
        'RATIO_AMT_CREDIT_SUM_TO_AMT_CREDIT_SUM_DEBT': ['min', 'max', 'mean'],
        'DIFF_AMT_CREDIT_SUM_AMT_CREDIT_SUM_OVERDUE': ['min', 'max', 'mean'],
        'RATIO_AMT_CREDIT_SUM_TO_AMT_CREDIT_SUM_OVERDUE': ['min', 'max', 'mean'],                                                            
        'DIFF_AMT_CREDIT_SUM_LIMIT_AMT_CREDIT_SUM_DEBT': ['min', 'max', 'mean'],
        'RATIO_AMT_CREDIT_SUM_DEBT_TO_AMT_CREDIT_SUM_LIMIT': ['min', 'max', 'mean'],
    }
    
    
    num_agg_df = bureau_to_agg.groupby('SK_ID_CURR').agg({**num_aggregations})
    num_agg_df.columns = pd.Index([e[0] + '_' + e[1].upper() + '_(BUREAU)' for e in num_agg_df.columns.tolist()])
    # Join to main dataframe
    X = X.join(num_agg_df, how='left', on='SK_ID_CURR')
    del num_agg_df
    gc.collect()
    
    
    # Build aggregations based on the numerical features, 
    # using only loans that are active.
    # The idea to do this came from Olivier's kernel:
    # https://www.kaggle.com/ogrellier/home-credit-hyperopt-optimization
    active_agg_df = bureau_to_agg[bureau_to_agg['CREDIT_ACTIVE'] == 'Active'].groupby('SK_ID_CURR').agg(num_aggregations)
    # Drop 'DAYS_ENDDATE_FACT' from the active loans DF, since this 
    # feature only occurs for closed loans.
    active_agg_df.drop('DAYS_ENDDATE_FACT', axis=1, inplace=True)
    # The column names of aggregate numerical features for 
    # active loans. Will be used below for creating ratios 
    # of aggregated numerical features of active loans to 
    # those of closed loans.
    cols = active_agg_df.columns.tolist()
    active_agg_df.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() + '_(BUREAU)' for e in active_agg_df.columns.tolist()])
    X = X.join(active_agg_df, how='left', on='SK_ID_CURR')
    del active_agg_df
    gc.collect()
    
    
    # Build aggregations based on the numerical features, 
    # using only loans that are closed.
    # The idea to do this came from Olivier's kernel:
    # https://www.kaggle.com/ogrellier/home-credit-hyperopt-optimization
    closed_agg_df = bureau_to_agg[bureau_to_agg['CREDIT_ACTIVE'] == 'Closed'].groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg_df.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() + '_(BUREAU)' for e in closed_agg_df.columns.tolist()])
    X = X.join(closed_agg_df, how='left', on='SK_ID_CURR')
    del bureau_to_agg, closed_agg_df
    gc.collect()
    
    
    # Finally, divide the aggregated numerical features of active loans 
    # by their counterparts for closed loans, to get their ratios. 
    # The idea to do this came from Olivier's kernel:
    # https://www.kaggle.com/ogrellier/home-credit-hyperopt-optimization
    for e in cols:
        X['RATIO_ACTIVE_TO_CLOSED_' + e[0] + "_" + e[1].upper() + '_(BUREAU)'] = X['ACTIVE_' + e[0] + "_" + e[1].upper() + '_(BUREAU)'] / X['CLOSED_' + e[0] + "_" + e[1].upper() + '_(BUREAU)']
    
    
    # Next, build aggregations based of the three categorical features.
    bureau_cat_feats = [
        'CREDIT_CURRENCY', 
        'CREDIT_TYPE', 
        'CREDIT_ACTIVE'
    ]
    cat_feats_df = bureau.loc[:,['SK_ID_CURR'] + bureau_cat_feats]
    
    # Change dtype to 'category' only after creating a new df 
    # containing the bureau table's cat features for *all* loans. 
    # (Later we will also create categorical features based on value of 
    #  each cat feature for only the loan *most-recently* applied for.)
    cat_feats_df[bureau_cat_feats] = cat_feats_df[bureau_cat_feats].apply(lambda x: x.astype('category'))
    
    cat_aggregations = {}
    for cat in bureau_cat_feats: 
        # Get numerical code of each categorical feature.
        cat_feats_df[cat] = cat_feats_df.loc[:,cat].cat.codes
        # Specify type of aggregation for cat features.
        cat_aggregations[cat] = ['mean']
        
    cat_agg_df = cat_feats_df.groupby('SK_ID_CURR').agg({**cat_aggregations})
    cat_agg_df.columns = pd.Index([e[0] + "_" + e[1].upper() + '_(BUREAU)' for e in cat_agg_df.columns.tolist()])
    X = X.join(cat_agg_df, how='left', on='SK_ID_CURR')
    del cat_feats_df, cat_agg_df
    gc.collect()
    
    
    # Create three categorical features, which are the values of 
    # of each of the three bureau categorical feats for *only* the 
    # loan most-recently applied for.
    #
    # In other words, for each borrower, for each categorical 
    # feature, we will only keep the value belonging to the loan 
    # that has the highest (or least negative) value for 'DAYS_CREDIT' 
    # (how long ago the loan was applied for, where a value of 0 
    # means most recent).
    latest_cat_feats_df = (bureau[['SK_ID_CURR', 'DAYS_CREDIT'] + bureau_cat_feats].loc[bureau[['SK_ID_CURR', 'DAYS_CREDIT'] + bureau_cat_feats].sort_values(['SK_ID_CURR','DAYS_CREDIT']).drop_duplicates('SK_ID_CURR',keep='last').index])
    
    # Change the dtype to 'category' only *after* getting rid of 
    # all loans except the loan most-recently applied for.
    latest_cat_feats_df[bureau_cat_feats] = latest_cat_feats_df[bureau_cat_feats].apply(lambda x: x.astype('category'))
    
    # Update the name of the three categorical features
    cat_feats_name_dict = {}
    for feature in bureau_cat_feats:
        cat_feats_name_dict[feature] = 'LATEST_' + feature + '_CAT_(BUREAU)'
    latest_cat_feats_df = latest_cat_feats_df.rename(index=str, columns = cat_feats_name_dict)
    
    # No longer need 'DAYS_CREDIT'
    latest_cat_feats_df.drop('DAYS_CREDIT', axis=1, inplace=True)
    
    # Join the three bureau categorical features to main dataframe
    X = X.join(latest_cat_feats_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    del latest_cat_feats_df
    gc.collect()
    
    
    # Build two features based on 'CREDIT_DAY_OVERDUE':

    # 'COUNT_CREDIT_BUREAU_LOANS_OVERDUE_(BUREAU)': the total number 
    # of overdue loans belonging to each borrower
    overdue_loans_df = bureau.loc[:,['SK_ID_CURR', 'CREDIT_DAY_OVERDUE']]
    overdue_loans_df['CREDIT_DAY_OVERDUE'] = overdue_loans_df['CREDIT_DAY_OVERDUE'].map(lambda x: 1 if x > 0 else 0)
    overdue_loans_df = overdue_loans_df.groupby('SK_ID_CURR', as_index=False, sort=False).sum()
    overdue_loans_df = overdue_loans_df.rename(index=str, columns = {'CREDIT_DAY_OVERDUE': 'COUNT_CREDIT_BUREAU_LOANS_OVERDUE_(BUREAU)'})

    # 'HAS_CREDIT_BUREAU_LOANS_OVERDUE_(BUREAU)': Whether or not the
    # borrower has one or more overdue loans (a binary categorical feature).
    overdue_loans_df['HAS_CREDIT_BUREAU_LOANS_OVERDUE_(BUREAU)'] = overdue_loans_df['COUNT_CREDIT_BUREAU_LOANS_OVERDUE_(BUREAU)'].map(lambda x: 1 if x > 0 else 0)

    # Join the two new features to the main dataframe
    X = X.join(overdue_loans_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    del overdue_loans_df
    gc.collect()
    
    
    # One-hot encode the 'CREDIT_TYPE', and 'CREDIT_CURRENCY' 
    # categorical features.
    one_hot_df = pd.get_dummies(bureau[['SK_ID_CURR', 'CREDIT_TYPE', 'CREDIT_CURRENCY' ]], columns = ['CREDIT_TYPE', 'CREDIT_CURRENCY'])

    # The names of the features created by one-hot encoding
    one_hot_feature_names = one_hot_df.columns.values.tolist()
    one_hot_feature_names.remove('SK_ID_CURR')

    # Some borrowers ('SK_ID_CURR') may have multiple rows for each 
    # one-hot encoded feature if they have more than one loan in the 
    # bureau table.
    #
    # I combine these rows for each borrower, setting the value to 1 
    # if the borrower has 1 in at least one of their rows for a particular 
    # one-hot encoded feature. Otherwise, the value is set to 0.
    one_hot_df = one_hot_df.groupby(['SK_ID_CURR'], as_index=False, sort=False).sum()
    one_hot_df[one_hot_feature_names] = one_hot_df[one_hot_feature_names].applymap(lambda x: 1 if x >= 1 else 0)

    # Update the names of the one-hot encoded features
    one_hot_feats_name_dict = {}
    for feature in one_hot_feature_names:
        one_hot_feats_name_dict[feature] = feature + '_(BUREAU)'
    one_hot_df = one_hot_df.rename(index=str, columns = one_hot_feats_name_dict)

    # Join the one-hot encoded features to main dataframe
    X = X.join(one_hot_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    del one_hot_df
    gc.collect()

    return X
def preprocess_bureau_balance(bureau_balance, bureau):
    
    # Note:
    #  - There are 1716428 unique SK_ID_BUREAU values in the bureau data table. 
    #  - There are 817395 unique SK_ID_BUREAU values in the bureau balance data table.
    #  - Which means: there are 43041 SK_ID_BUREAU values in the bureau balance data table, 
    #    which are *not* in the bureau data table. As such, there are no corresponding 
    #    SK_ID_CURR ids that can ever be associated with these 43041 entries.
    #  - Therefore: we can eliminate from the bureau balance table all rows 
    #    belonging to SK_ID_CURRs that aren't also in the bureau table.
    bureau_table_borrowers = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']]
    bureau_balance = bureau_table_borrowers.join(bureau_balance.set_index('SK_ID_BUREAU'), on='SK_ID_BUREAU', how='inner')

    # Replace all values of 'XNA', 'XAP' with np.nan
    bureau_balance = replace_XNA_XAP(bureau_balance)
    
    return bureau_balance
def engineer_bureau_balance_features(X, bureau_balance):
    
    # Build aggregate features based on the numerical 
    # 'MONTHS_BALANCE' and the categorical 'STATUS' features:
    # 
    # The approach is to first group by loan ('SK_ID_BUREAU') 
    # and generate a set of aggregate features. Then, group 
    # by borrower ('SK_ID_CURR') and generate a second set 
    # of aggregate features, which aggregates the set that 
    # had initially been aggregated by loan.
    
    # In order to aggregate the 'STATUS' feature, we first 
    # change it's dtype to 'category' and then get the 
    # numerical category codes. We make a copy of the bureau 
    # balance table because later on we will make yet another 
    # feature built on the 'STATUS' feature's original values.
    bureau_balance_to_agg = bureau_balance.copy()
    bureau_balance_to_agg[['STATUS']] = bureau_balance_to_agg[['STATUS']].apply(lambda x: x.astype('category'))
    bureau_balance_to_agg['STATUS'] = bureau_balance_to_agg.loc[:,'STATUS'].cat.codes

    # Aggregate across each loan
    months_bal_agg_by_loan = {
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'STATUS': ['min', 'max', 'var']
    }
    months_bal_agg_loan_df = bureau_balance_to_agg.groupby('SK_ID_BUREAU').agg({**months_bal_agg_by_loan})
    months_bal_agg_loan_df.columns = pd.Index([e[0] + '_' + e[1].upper() + '_BY_LOAN' for e in months_bal_agg_loan_df.columns.tolist()])
    months_bal_agg_loan_df.reset_index(level=0, inplace=False)
    bureau_balance_to_agg = bureau_balance_to_agg.join(months_bal_agg_loan_df, how='left', on='SK_ID_BUREAU')

    # Now aggregate across each borrower
    months_bal_agg_by_borrower = {
        'MONTHS_BALANCE_MIN_BY_LOAN': ['min'],
        'MONTHS_BALANCE_MAX_BY_LOAN': ['max'],
        'MONTHS_BALANCE_SIZE_BY_LOAN': ['mean', 'sum'],
        'STATUS_MIN_BY_LOAN': ['mean'],
        'STATUS_MAX_BY_LOAN': ['mean'],
        'STATUS_VAR_BY_LOAN': ['mean']
    }
    months_bal_agg_borrower_df = bureau_balance_to_agg.groupby('SK_ID_CURR').agg({**months_bal_agg_by_borrower})
    months_bal_agg_borrower_df.columns = pd.Index([e[0] + '_' + e[1].upper() + '_(BUREAU_BALANCE)' for e in months_bal_agg_borrower_df.columns.tolist()])
    # Join to the main data table.
    X = X.join(months_bal_agg_borrower_df, how='left', on='SK_ID_CURR')
    del bureau_balance_to_agg, months_bal_agg_loan_df, months_bal_agg_borrower_df
    gc.collect()
    
    
    # 'COUNT_MONTHS_STATUS_WAS_DPD_(BUREAU_BALANCE)': Number of times borrower 
    # had a bureau loan where its status for a particular month was 31 or 
    # more days past due. 
    #
    # This is the number of times a borrower had a loan where the value of 
    # the 'STATUS' feature was either 2, 3, 4, or 5.
    count_status_df = bureau_balance.loc[:,['SK_ID_CURR', 'STATUS']]
    count_status_df['STATUS'] = count_status_df['STATUS'].map(lambda x: 0 if x not in ['5', '4', '3', '2'] else 1)
    count_status_df = count_status_df.groupby(['SK_ID_CURR'], as_index=False, sort=False).sum()
    count_status_df = count_status_df.rename(index=str, columns = {'STATUS': 'COUNT_MONTHS_STATUS_WAS_DPD_(BUREAU_BALANCE)'})
    
    # Join to main dataframe
    X = X.join(count_status_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    del count_status_df
    gc.collect()
    
    return X
def preprocess_previous_application(previous_application):
    
    # Replace all values of 'XNA', 'XAP' with np.nan
    previous_application = replace_XNA_XAP(previous_application)

    # Replace all entries of 365243.0 with nan
    previous_application.loc[:,[
        'DAYS_FIRST_DRAWING',
        'DAYS_FIRST_DUE',
        'DAYS_LAST_DUE_1ST_VERSION',
        'DAYS_LAST_DUE',
        'DAYS_TERMINATION',
    ]].replace(365243.0, np.nan, inplace=True)
    
    return previous_application
def engineer_previous_application_features(X, previous_application):
    
    # Create aggregations of the numerical and categorical features:
    prev_app_to_agg = previous_application.copy()
    
    # Add a new numerical feature: the percentage of the amount of 
    # money applied for in the loan application, that was actually received.
    # The idea to do this came from Olivier's kernel:
    # https://www.kaggle.com/ogrellier/home-credit-hyperopt-optimization
    prev_app_to_agg['RATIO_AMT_APPLICATION_TO_AMT_CREDIT'] = (prev_app_to_agg['AMT_APPLICATION'] / prev_app_to_agg['AMT_CREDIT'])
    
    #New aggregations
    prev_app_to_agg['RATIO_AMT_CREDIT_TO_AMT_ANNUITY'] = prev_app_to_agg['AMT_CREDIT'] / prev_app_to_agg['AMT_ANNUITY']
    prev_app_to_agg['RATIO_AMT_APPLICATION_TO_AMT_ANNUITY'] = prev_app_to_agg['AMT_APPLICATION'] / prev_app_to_agg['AMT_ANNUITY']
    prev_app_to_agg['DIFF_AMT_DOWN_PAYMENT_AMT_ANNUITY'] = prev_app_to_agg['AMT_DOWN_PAYMENT'] - prev_app_to_agg['AMT_ANNUITY']
    prev_app_to_agg['RATIO_AMT_CREDIT_TO_AMT_DOWN_PAYMENT'] = prev_app_to_agg['AMT_CREDIT'] / prev_app_to_agg['AMT_DOWN_PAYMENT']
    prev_app_to_agg['DIFF_AMT_CREDIT_AMT_GOODS_PRICE'] = prev_app_to_agg['AMT_CREDIT'] - prev_app_to_agg['AMT_GOODS_PRICE']
    prev_app_to_agg['DIFF_AMT_APPLICATION_AMT_GOODS_PRICE'] = prev_app_to_agg['AMT_APPLICATION'] - prev_app_to_agg['AMT_GOODS_PRICE']
    prev_app_to_agg['DIFF_RATE_DOWN_PAYMENT_RATE_INTEREST_PRIMARY'] = prev_app_to_agg['RATE_DOWN_PAYMENT'] - prev_app_to_agg['RATE_INTEREST_PRIMARY']
    prev_app_to_agg['DIFF_RATE_INTEREST_PRIVILEGED_RATE_INTEREST_PRIMARY'] = prev_app_to_agg['RATE_INTEREST_PRIVILEGED'] - prev_app_to_agg['RATE_INTEREST_PRIMARY']
    prev_app_to_agg['DIFF_DAYS_LAST_DUE_DAYS_FIRST_DUE'] = prev_app_to_agg['DAYS_LAST_DUE'] - prev_app_to_agg['DAYS_FIRST_DUE']
    prev_app_to_agg['DIFF_DAYS_TERMINATION_DAYS_DECISION'] = prev_app_to_agg['DAYS_TERMINATION'] - prev_app_to_agg['DAYS_DECISION']
    prev_app_to_agg['RATIO_DAYS_LAST_DUE_TO_DAYS_LAST_DUE_1ST_VERSION'] = prev_app_to_agg['DAYS_LAST_DUE'] / prev_app_to_agg['DAYS_LAST_DUE_1ST_VERSION']
    prev_app_to_agg['RATIO_DAYS_DECISION_TO_DAYS_TERMINATION'] = prev_app_to_agg['DAYS_DECISION'] / prev_app_to_agg['DAYS_TERMINATION']
    
    # The numerical feature aggregations
    num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'], 
        'AMT_CREDIT': ['min', 'max', 'mean'], 
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'], 
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'], 
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'], 
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'], 
        'RATE_INTEREST_PRIMARY': ['min', 'max', 'mean'],   
        'RATE_INTEREST_PRIVILEGED': ['min', 'max', 'mean'], 
        'SELLERPLACE_AREA': ['min', 'max', 'mean'], 
        'CNT_PAYMENT': ['mean', 'sum'],
        'DAYS_FIRST_DRAWING': ['min', 'max', 'mean'], 
        'DAYS_FIRST_DUE': ['min', 'max', 'mean'], 
        'DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'], 
        'DAYS_LAST_DUE': ['min', 'max', 'mean'],
        'DAYS_TERMINATION': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'RATIO_AMT_APPLICATION_TO_AMT_CREDIT': ['min', 'max', 'mean', 'var'],
        
        #New aggregations
        'RATIO_AMT_CREDIT_TO_AMT_ANNUITY': ['min', 'max', 'mean'],
        'RATIO_AMT_APPLICATION_TO_AMT_ANNUITY': ['min', 'max', 'mean'],
        'DIFF_AMT_DOWN_PAYMENT_AMT_ANNUITY': ['min', 'max', 'mean'],
        'RATIO_AMT_CREDIT_TO_AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DIFF_AMT_CREDIT_AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'DIFF_AMT_APPLICATION_AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'DIFF_RATE_DOWN_PAYMENT_RATE_INTEREST_PRIMARY': ['min', 'max', 'mean'],
        'DIFF_RATE_INTEREST_PRIVILEGED_RATE_INTEREST_PRIMARY': ['min', 'max', 'mean'],
        'DIFF_DAYS_LAST_DUE_DAYS_FIRST_DUE': ['min', 'max', 'mean'],
        'DIFF_DAYS_TERMINATION_DAYS_DECISION': ['min', 'max', 'mean'],
        'RATIO_DAYS_LAST_DUE_TO_DAYS_LAST_DUE_1ST_VERSION': ['min', 'max', 'mean'],
        'RATIO_DAYS_DECISION_TO_DAYS_TERMINATION': ['min', 'max', 'mean'],           
    }
    
    # The categorical features.
    cat_feats = [
        'NAME_CONTRACT_TYPE', 
        'WEEKDAY_APPR_PROCESS_START', 
        'FLAG_LAST_APPL_PER_CONTRACT', 
        'NFLAG_LAST_APPL_IN_DAY',
        'NAME_CASH_LOAN_PURPOSE', 
        'NAME_CONTRACT_STATUS',  
        'NAME_PAYMENT_TYPE', 
        'CODE_REJECT_REASON',  
        'NAME_TYPE_SUITE', 
        'NAME_CLIENT_TYPE', 
        'NAME_GOODS_CATEGORY', 
        'NAME_PORTFOLIO', 
        'NAME_PRODUCT_TYPE', 
        'CHANNEL_TYPE', 
        'NAME_SELLER_INDUSTRY', 
        'NAME_YIELD_GROUP', 
        'PRODUCT_COMBINATION', 
        'NFLAG_INSURED_ON_APPROVAL',
    ]
    
    # Change dtype of each categorical feature to 'category'.
    # So that we can get a numeric code for each category of 
    # each feature. Need a copy of the dataframe containing all the 
    # features engineered above -- we will need to leave 
    # 'NAME_CONTRACT_STATUS' as non-categorical dtype so that
    # we can later run the numerical aggregations once on loans 
    # that come from 'Approved' applications, and then again
    # on loans that come from 'Refused' applications.
    prev_app_to_agg_with_cat_feats = prev_app_to_agg.copy()
    prev_app_to_agg_with_cat_feats[cat_feats] = prev_app_to_agg_with_cat_feats[cat_feats].apply(lambda x: x.astype('category'))
    
    # Categorical feature aggregations
    cat_aggregations = {}
    for cat in cat_feats:
        # Convert each cat features values to numeric codes
        prev_app_to_agg_with_cat_feats[cat] = prev_app_to_agg_with_cat_feats.loc[:,cat].cat.codes
        # Specify type of aggregation for cat features.
        cat_aggregations[cat] = ['mean']
    
    # Aggregate both numerical and categorical features
    prev_app_agg_df = prev_app_to_agg_with_cat_feats.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    prev_app_agg_df.columns = pd.Index([e[0] + "_" + e[1].upper() + '_(PREV_APP)' for e in prev_app_agg_df.columns.tolist()])
    # Join the aggregated features to main dataframe
    X = X.join(prev_app_agg_df, how='left', on='SK_ID_CURR')
    del prev_app_to_agg_with_cat_feats, prev_app_agg_df 
    gc.collect()
    
    
    # Now, create the numerical feature aggregations for only the approved 
    # applications. And then again for only the rejected applications. 
    # Finally, create a third set of features, which will be the ratios 
    # of the approved application agg features to the rejected 
    # application agg features.
    # The idea to do this came from Olivier's kernel:
    # https://www.kaggle.com/ogrellier/home-credit-hyperopt-optimization
    
    # Aggregate numerical features for approved applications
    approved_agg_df = prev_app_to_agg[prev_app_to_agg['NAME_CONTRACT_STATUS'] == 'Approved'].groupby('SK_ID_CURR').agg(num_aggregations)
    # Keep track of column names so we can make the ratios 
    # of approved application agg feats to rejected 
    # application agg feats.
    cols = approved_agg_df.columns.tolist()
    approved_agg_df.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() + '_(PREV_APP)' for e in approved_agg_df.columns.tolist()])
    X = X.join(approved_agg_df, how='left', on='SK_ID_CURR')
    del approved_agg_df
    gc.collect()
    
    # Aggregate numerical features for rejected applications
    rejected_agg_df = prev_app_to_agg[prev_app_to_agg['NAME_CONTRACT_STATUS'] == 'Refused'].groupby('SK_ID_CURR').agg(num_aggregations)
    rejected_agg_df.columns = pd.Index(['REJECTED_' + e[0] + "_" + e[1].upper() + '_(PREV_APP)' for e in rejected_agg_df.columns.tolist()])
    X = X.join(rejected_agg_df, how='left', on='SK_ID_CURR')
    del prev_app_to_agg, rejected_agg_df
    gc.collect()
    
    
    # And create the ratios, dividing each aggregated numerical feature for 
    # approved applications, by their counterpart features for rejected 
    # applications.
    for e in cols:
        X['RATIO_APPROVED_TO_REJECTED_' + e[0] + "_" + e[1].upper() + '_(PREV_APP)'] = X['APPROVED_' + e[0] + "_" + e[1].upper() + '_(PREV_APP)'] / X['REJECTED_' + e[0] + "_" + e[1].upper() + '_(PREV_APP)']
    

    # One-hot encode the categorical features in the previous 
    # application data table.
    one_hot_df = pd.get_dummies(previous_application[['SK_ID_CURR'] + cat_feats], columns = cat_feats)

    # The names of the features created by one-hot encoding
    one_hot_feature_names = one_hot_df.columns.values.tolist()
    one_hot_feature_names.remove('SK_ID_CURR')

    # Some borrowers ('SK_ID_CURR') may have multiple rows for each 
    # one-hot encoded feature if they have more than one application 
    # in the previous application table.
    #
    # I combine these rows for each borrower, setting the value to 1 
    # if the borrower has 1 in at least one of their rows for a particular 
    # one-hot encoded feature. Otherwise, the value is set to 0.
    one_hot_df = one_hot_df.groupby(['SK_ID_CURR'], as_index=False, sort=False).sum()
    one_hot_df[one_hot_feature_names] = one_hot_df[one_hot_feature_names].applymap(lambda x: 1 if x >= 1 else 0)

    # Update the names of the one-hot encoded features
    one_hot_feats_name_dict = {}
    for feature in one_hot_feature_names:
        one_hot_feats_name_dict[feature] = feature + '_(PREV_APP)'
    one_hot_df = one_hot_df.rename(index=str, columns = one_hot_feats_name_dict)

    # Join the one-hot encoded features to main dataframe
    X = X.join(one_hot_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    del one_hot_df
    gc.collect()
    
    return X
def engineer_pos_cash_balance_features(X, pos_cash_balance):
    
    # Create aggregations of the numerical and categorical features:
    
    # Copy the dataframe. Necessary because the one categorical 
    # feature ('NAME_CONTRACT_STATUS') will first be aggregated, 
    # and then later, new features will be engineered using its 
    # original values. 
    pos_cash_to_agg = pos_cash_balance.copy()
    
    # Engineer some new features
    pos_cash_to_agg['RATIO_SK_DPD_TO_CNT_INSTALMENT_FUTURE'] = pos_cash_to_agg['SK_DPD'] / pos_cash_to_agg['CNT_INSTALMENT_FUTURE']
    pos_cash_to_agg['RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT_FUTURE'] = pos_cash_to_agg['SK_DPD_DEF'] / pos_cash_to_agg['CNT_INSTALMENT_FUTURE']
    pos_cash_to_agg['RATIO_SK_DPD_TO_CNT_INSTALMENT'] = pos_cash_to_agg['SK_DPD'] / pos_cash_to_agg['CNT_INSTALMENT']
    pos_cash_to_agg['RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT'] = pos_cash_to_agg['SK_DPD_DEF'] / pos_cash_to_agg['CNT_INSTALMENT']
    pos_cash_to_agg['RATIO_CNT_INSTALMENT_TO_CNT_INSTALMENT_FUTURE'] = pos_cash_to_agg['CNT_INSTALMENT'] / pos_cash_to_agg['CNT_INSTALMENT_FUTURE']
    pos_cash_to_agg['DIFF_CNT_INSTALMENT_FUTURE_CNT_INSTALMENT'] = pos_cash_to_agg['CNT_INSTALMENT_FUTURE'] - pos_cash_to_agg['CNT_INSTALMENT']

    
    # First define the feature aggregations
    aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean', 'sum'],
        'SK_DPD_DEF': ['max', 'mean', 'sum'],
        'CNT_INSTALMENT_FUTURE': ['mean', 'sum'],
        'CNT_INSTALMENT': ['max'],
        'NAME_CONTRACT_STATUS': ['mean'],
        
        # New
        'RATIO_SK_DPD_TO_CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'RATIO_SK_DPD_TO_CNT_INSTALMENT': ['min', 'max', 'mean'],
        'RATIO_SK_DPD_DEF_TO_CNT_INSTALMENT': ['min', 'max', 'mean'],
        'RATIO_CNT_INSTALMENT_TO_CNT_INSTALMENT_FUTURE': ['min', 'max', 'mean'],
        'DIFF_CNT_INSTALMENT_FUTURE_CNT_INSTALMENT': ['min', 'max', 'mean'],
    }
    
    # Change dtype of 'NAME_CONTRACT_STATUS' to 'category'.
    # So that we can get a numeric code for each of its entries.
    pos_cash_to_agg[['NAME_CONTRACT_STATUS']] = pos_cash_to_agg[['NAME_CONTRACT_STATUS']].apply(lambda x: x.astype('category'))
    # Convert each cat features values to numeric codes
    pos_cash_to_agg['NAME_CONTRACT_STATUS'] = pos_cash_to_agg.loc[:,'NAME_CONTRACT_STATUS'].cat.codes

    # Aggregate the features
    pos_cash_agg_df = pos_cash_to_agg.groupby('SK_ID_CURR').agg(aggregations)
    pos_cash_agg_df.columns = pd.Index([e[0] + "_" + e[1].upper() + '_(POS_CASH)' for e in pos_cash_agg_df.columns.tolist()])
    # Join the aggregated features to main dataframe
    X = X.join(pos_cash_agg_df, how='left', on='SK_ID_CURR')
    del pos_cash_to_agg, pos_cash_agg_df 
    gc.collect()
    
    
    # 'COUNT_POS_CASH_LOANS_(POS_CASH)': Number of POS cash loans each 
    # borrower has. Will eventually be aggregated with the number of 
    # credit card loans, and number of bureau loans, to give the 
    # total number of loans an applicant has
    count_pos_cash_loans_df = pos_cash_balance[['SK_ID_CURR', 'SK_ID_PREV']]
    count_pos_cash_loans_df = pd.DataFrame(data=count_pos_cash_loans_df.groupby(['SK_ID_CURR'], as_index=True)['SK_ID_PREV'].nunique()).reset_index(level=0, inplace=False)
    count_pos_cash_loans_df = count_pos_cash_loans_df.rename(index=str, columns = {'SK_ID_PREV': 'COUNT_POS_CASH_LOANS_(POS_CASH)'})
    # Join to main dataframe
    X = X.join(count_pos_cash_loans_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    del count_pos_cash_loans_df
    gc.collect()
    
    
    # Also create 'COUNT_POS_CASH_LOANS_CAT_(POS_CASH)', which 
    # is the same as 'COUNT_POS_CASH_LOANS_(POS_CASH)' above, 
    # but encoded as a categorical feature with cardinal values.
    X['COUNT_POS_CASH_LOANS_CAT_(POS_CASH)'] = X['COUNT_POS_CASH_LOANS_(POS_CASH)']
    X[['COUNT_POS_CASH_LOANS_CAT_(POS_CASH)']] = X[['COUNT_POS_CASH_LOANS_CAT_(POS_CASH)']].apply(lambda x: x.astype('category'))
    
    # Build a set of categorical features that indicate the number 
    # of previous cash loans that each borrower ('SK_ID_CURR') has 
    # that have a particular status during their most recently 
    # updated monthly balance:
    
    # The statuses for which such a feature will be built:
    recent_statuses = [
#         'Active', 
        'Completed', 
        'Signed', 
#         'Demand', 
#         'Returned to the store',
#         'Approved',
#         'Amortized debt',
#         'Canceled'
    ]
    
    # Get only the statuses for most recent balance 
    # for each loan of every borrower
    recent_status_df = pos_cash_balance[['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE','NAME_CONTRACT_STATUS']]
    recent_status_df = recent_status_df.loc[recent_status_df.sort_values(['SK_ID_PREV','MONTHS_BALANCE']).drop_duplicates('SK_ID_PREV',keep='last').index]

    # Create a categorical feature for each status type 
    # in recent_contract_statuses above:
    for status in recent_statuses:
        new_feature_name = 'NUMBER_CONTRACTS_MOST_RECENTLY_' + status.upper() + '_CAT_(POS_CASH)'
        recent_status_df[new_feature_name] = recent_status_df['NAME_CONTRACT_STATUS'].map(lambda x: 1 if x == status else 0)
        new_feature_df = recent_status_df[['SK_ID_CURR', new_feature_name]].groupby(['SK_ID_CURR'], as_index=False, sort=False).sum()
        # Join new feature to main dataframe
        X = X.join(new_feature_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
        # Change dtype to categorical
        X[[new_feature_name]] = X[[new_feature_name]].apply(lambda x: x.astype('category'))


    # Create another set of categorical features, where each feature is a sum, 
    # for each borrower, of all the number of times any of the borrower's loans 
    # had a particular status during all of their months of balance history.
    # 
    # In other words: while the previous set of categorical features looked at 
    # only the *most recent* status for each borrower's loans. Now, we will count 
    # the *total* number of times that a borrower had any of their loans at any 
    # month be designated a certain balance. 
    
    # The statuses for which such a feature will be built:
    total_statuses = [
#         'Active', 
        'Completed', 
#         'Signed', 
#         'Demand', 
#         'Returned to the store',
#         'Approved',
#         'Amortized debt',
#         'Canceled'
    ]
    
    for status in total_statuses:
        new_feature_df = pos_cash_balance.loc[:,['SK_ID_CURR', 'NAME_CONTRACT_STATUS']]
        new_feature_df['NAME_CONTRACT_STATUS'] = new_feature_df['NAME_CONTRACT_STATUS'].map(lambda x: np.nan if x != status else status)
        new_feature_df = new_feature_df.groupby(['SK_ID_CURR'], as_index=False, sort=False).count()
        new_feature_name = 'TOTAL_FREQ_CONTRACT_STATUS_' + status.upper() + '_CAT_(POS_CASH)'
        new_feature_df = new_feature_df.rename(index=str, columns = {'NAME_CONTRACT_STATUS': new_feature_name})
        # Change dtype to categorical
        new_feature_df[[new_feature_name]] = new_feature_df[[new_feature_name]].apply(lambda x: x.astype('category'))
        # Join new feature to main dataframe
        X = X.join(new_feature_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    
    del recent_status_df
    gc.collect()
    
    return X
def engineer_installments_payments_features(X, installments_payments):
    
    # Create four new numerical features, which will be 
    # aggregated along with the other numerical feats. 
    # The first two, 'PCTG_PAYMENT' and 'DIFF_PAYMENT', 
    # came from Olivier's kernel at:
    # https://www.kaggle.com/ogrellier/home-credit-hyperopt-optimization
    install_pay_to_agg = installments_payments.copy()
    
    
    # 'PCTG_PAYMENT': The percentage of the required payment value that was actually paid. 
    install_pay_to_agg['PCTG_PAYMENT'] = install_pay_to_agg['AMT_PAYMENT'] / install_pay_to_agg['AMT_INSTALMENT']
    # 'DIFF_PAYMENT': The difference between the required payment value and the and the amount that was actually paid.
    install_pay_to_agg['DIFF_PAYMENT'] = install_pay_to_agg['AMT_INSTALMENT'] - installments_payments['AMT_PAYMENT']
    # 'DAYS_PAST_DUE': How many days late did the payment finally arrive.
    install_pay_to_agg['DAYS_PAST_DUE'] = install_pay_to_agg['DAYS_ENTRY_PAYMENT'] - install_pay_to_agg['DAYS_INSTALMENT']
    # 'DAYS_BEFORE_DUE': How many days early was the payment made.
    install_pay_to_agg['DAYS_BEFORE_DUE'] = install_pay_to_agg['DAYS_INSTALMENT'] - install_pay_to_agg['DAYS_ENTRY_PAYMENT']
    # Make sure we only take into account when a payment is 
    # actually early, or actually overdue (no negative values).
    install_pay_to_agg['DAYS_PAST_DUE'] = install_pay_to_agg['DAYS_PAST_DUE'].apply(lambda x: x if x > 0 else 0)
    install_pay_to_agg['DAYS_BEFORE_DUE'] = install_pay_to_agg['DAYS_BEFORE_DUE'].apply(lambda x: x if x > 0 else 0)
    
    # Build aggregations of the features (all are numerical).
    aggregations = {
        'NUM_INSTALMENT_VERSION': ['max', 'nunique'],
        'DAYS_PAST_DUE': ['max', 'mean', 'sum'],
        'DAYS_BEFORE_DUE': ['max', 'mean', 'sum'],
        'DAYS_INSTALMENT': ['min'],
        'PCTG_PAYMENT': ['min', 'max', 'mean', 'sum', 'var'],
        'DIFF_PAYMENT': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['min','max', 'mean', 'sum'],
    }
    
    install_agg_df = install_pay_to_agg.groupby('SK_ID_CURR').agg(aggregations)
    install_agg_df.columns = pd.Index([e[0] + "_" + e[1].upper() + '_(INSTALL_PAY)' for e in install_agg_df.columns.tolist()])
    X = X.join(install_agg_df, on='SK_ID_CURR', how='left' )
    del install_pay_to_agg, install_agg_df
    gc.collect()
    
    
    # 'COUNT_INSTALLMENTS_PAID_(INSTALL_PAY)':  The total number of payment 
    # installments made by each borrower, across all their loans.
    count_installments_df = installments_payments[['SK_ID_CURR', 'NUM_INSTALMENT_NUMBER']].groupby(['SK_ID_CURR'], as_index=False, sort=False).count()
    count_installments_df = count_installments_df.rename(index=str, columns = {'NUM_INSTALMENT_NUMBER': 'COUNT_INSTALLMENTS_PAID_(INSTALL_PAY)'})
    # Join to main dataframe
    X = X.join(count_installments_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    del count_installments_df
    gc.collect()
    
    
    # 'COUNT_INSTALLMENTS_ACCTS_CAT_(INSTALL_PAY)': Number of installments accounts each 
    # borrower has. Encode as a categorical feature.
    count_install_accts_df = installments_payments[['SK_ID_CURR', 'SK_ID_PREV']]
    count_install_accts_df = pd.DataFrame(data=count_install_accts_df.groupby(['SK_ID_CURR'], as_index=True)['SK_ID_PREV'].nunique()).reset_index(level=0, inplace=False)
    count_install_accts_df = count_install_accts_df.rename(index=str, columns = {'SK_ID_PREV': 'COUNT_INSTALLMENTS_ACCTS_CAT_(INSTALL_PAY)'})
    count_install_accts_df[['COUNT_INSTALLMENTS_ACCTS_CAT_(INSTALL_PAY)']] = count_install_accts_df[['COUNT_INSTALLMENTS_ACCTS_CAT_(INSTALL_PAY)']].apply(lambda x: x.astype('category'))
    # Join to main dataframe
    X = X.join(count_install_accts_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    del count_install_accts_df
    gc.collect()
    
    
    # 'INSTALMENT_VERSION_LAST_DUE_CAT_(INSTALL_PAY)': A categorical 
    # feature which gives the installment version of the loan that was 
    # *most recently due*, for each borrower.
    last_version_df = installments_payments[['SK_ID_CURR', 'DAYS_INSTALMENT', 'NUM_INSTALMENT_VERSION']]
    last_version_df = last_version_df.loc[last_version_df.sort_values(['SK_ID_CURR','DAYS_INSTALMENT']).drop_duplicates('SK_ID_CURR',keep='last').index]
    last_version_df = last_version_df.rename(index=str, columns = {'NUM_INSTALMENT_VERSION': 'INSTALMENT_VERSION_LAST_DUE_CAT_(INSTALL_PAY)'})
    last_version_df[['INSTALMENT_VERSION_LAST_DUE_CAT_(INSTALL_PAY)']] = last_version_df[['INSTALMENT_VERSION_LAST_DUE_CAT_(INSTALL_PAY)']].apply(lambda x: x.astype('category'))
    # Join to main dataframe
    X = X.join(last_version_df[['SK_ID_CURR', 'INSTALMENT_VERSION_LAST_DUE_CAT_(INSTALL_PAY)']].set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    del last_version_df
    gc.collect()
    
    return X
def engineer_credit_card_balance_features(X, credit_card_balance):
    
    # Build aggregations of all the numerical features, along 
    # with the one categorical feature, 'NAME_CONTRACT_STATUS'.
    cc_to_agg = credit_card_balance.copy()
    
    # Engineer some new features.
    cc_to_agg['RATIO_AMT_BALANCE_TO_AMT_CREDIT_LIMIT_ACTUAL'] = cc_to_agg['AMT_BALANCE'] / cc_to_agg['AMT_CREDIT_LIMIT_ACTUAL']
    cc_to_agg['SUM_ALL_AMT_DRAWINGS'] = cc_to_agg[['AMT_DRAWINGS_ATM_CURRENT', 
                                                   'AMT_DRAWINGS_CURRENT', 
                                                   'AMT_DRAWINGS_OTHER_CURRENT', 
                                                   'AMT_DRAWINGS_POS_CURRENT']].sum(axis=1)
    cc_to_agg['RATIO_AMT_PAYMENT_TOTAL_CURRENT_TO_AMT_TOTAL_RECEIVABLE'] = cc_to_agg['AMT_PAYMENT_TOTAL_CURRENT'] / cc_to_agg['AMT_TOTAL_RECEIVABLE']
    cc_to_agg['RATIO_AMT_PAYMENT_CURRENT_TO_AMT_RECIVABLE'] = cc_to_agg['AMT_PAYMENT_CURRENT'] / cc_to_agg['AMT_RECIVABLE']
    cc_to_agg['SUM_ALL_CNT_DRAWINGS'] = cc_to_agg[['CNT_DRAWINGS_ATM_CURRENT', 
                                                   'CNT_DRAWINGS_CURRENT', 
                                                   'CNT_DRAWINGS_OTHER_CURRENT', 
                                                   'CNT_DRAWINGS_POS_CURRENT']].sum(axis=1)
    cc_to_agg['RATIO_ALL_AMT_DRAWINGS_TO_ALL_CNT_DRAWINGS'] = cc_to_agg['SUM_ALL_AMT_DRAWINGS'] / cc_to_agg['SUM_ALL_CNT_DRAWINGS']
    cc_to_agg['DIFF_AMT_TOTAL_RECEIVABLE_AMT_PAYMENT_TOTAL_CURRENT'] = cc_to_agg['AMT_TOTAL_RECEIVABLE'] / cc_to_agg['AMT_PAYMENT_TOTAL_CURRENT']
    cc_to_agg['RATIO_AMT_PAYMENT_CURRENT_TO_AMT_PAYMENT_TOTAL_CURRENT'] = cc_to_agg['AMT_PAYMENT_CURRENT'] / cc_to_agg['AMT_PAYMENT_TOTAL_CURRENT']
    cc_to_agg['RATIO_AMT_RECEIVABLE_PRINCIPAL_TO_AMT_RECIVABLE'] = cc_to_agg['AMT_RECEIVABLE_PRINCIPAL'] / cc_to_agg['AMT_RECIVABLE']
    
    aggregations = {
        'MONTHS_BALANCE': ['min', 'max', 'size'],
        'AMT_BALANCE': ['min', 'max', 'mean', 'sum'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean', 'sum', 'var'],
        'AMT_DRAWINGS_ATM_CURRENT': ['max'],
        'AMT_DRAWINGS_CURRENT': ['max'],
        'AMT_DRAWINGS_OTHER_CURRENT': ['max'],
        'AMT_DRAWINGS_POS_CURRENT': ['max'],
        'AMT_INST_MIN_REGULARITY': ['max'],
        'AMT_PAYMENT_CURRENT': ['max'],
        'AMT_PAYMENT_TOTAL_CURRENT': ['max'],
        'AMT_RECEIVABLE_PRINCIPAL': ['mean', 'sum'],
        'AMT_RECIVABLE': ['mean', 'sum'],
        'AMT_TOTAL_RECEIVABLE': ['mean'],
        'CNT_DRAWINGS_ATM_CURRENT': ['max'], 
        'CNT_DRAWINGS_CURRENT': ['max'],
        'CNT_DRAWINGS_OTHER_CURRENT': ['max'],
        'CNT_DRAWINGS_POS_CURRENT': ['max'],
        'CNT_INSTALMENT_MATURE_CUM': ['mean', 'sum'],
        'SK_DPD': ['max', 'sum'],
        'SK_DPD_DEF': ['max', 'sum'],
        'NAME_CONTRACT_STATUS': ['mean'],
        
        #Newly engineered feats
        'RATIO_AMT_BALANCE_TO_AMT_CREDIT_LIMIT_ACTUAL': ['min', 'max', 'mean'],
        'SUM_ALL_AMT_DRAWINGS': ['min', 'max', 'mean'],
        'RATIO_AMT_PAYMENT_TOTAL_CURRENT_TO_AMT_TOTAL_RECEIVABLE': ['min', 'max', 'mean'],
        'RATIO_AMT_PAYMENT_CURRENT_TO_AMT_RECIVABLE': ['min', 'max', 'mean'],
        'SUM_ALL_CNT_DRAWINGS': ['min', 'max', 'mean'],
        'RATIO_ALL_AMT_DRAWINGS_TO_ALL_CNT_DRAWINGS': ['min', 'max', 'mean'],
        'DIFF_AMT_TOTAL_RECEIVABLE_AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean'],
        'RATIO_AMT_PAYMENT_CURRENT_TO_AMT_PAYMENT_TOTAL_CURRENT': ['min', 'max', 'mean'],
        'RATIO_AMT_RECEIVABLE_PRINCIPAL_TO_AMT_RECIVABLE': ['min', 'max', 'mean'],
    }
    
    
    # Get numeric codes for the entries in 'NAME_CONTRACT_STATUS'.
    cc_to_agg[['NAME_CONTRACT_STATUS']] = cc_to_agg[['NAME_CONTRACT_STATUS']].apply(lambda x: x.astype('category'))
    cc_to_agg['NAME_CONTRACT_STATUS'] = cc_to_agg.loc[:,'NAME_CONTRACT_STATUS'].cat.codes
    
    # Build the aggregations
    cc_agg_df = cc_to_agg.groupby('SK_ID_CURR').agg(aggregations)
    cc_agg_df.columns = pd.Index([e[0] + "_" + e[1].upper() + '_(CREDIT_CARD)' for e in cc_agg_df.columns.tolist()])
    # Join to main dataframe
    X = X.join(cc_agg_df, on='SK_ID_CURR', how='left')
    del cc_to_agg, cc_agg_df
    gc.collect()
    
    
    # 'COUNT_CREDIT_CARD_LOANS_(CREDIT_CARD)': Number of credit card loans each 
    # borrower has.
    count_credit_card_loans_df = credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV']]
    count_credit_card_loans_df = pd.DataFrame(data=count_credit_card_loans_df.groupby(['SK_ID_CURR'], as_index=True)['SK_ID_PREV'].nunique()).reset_index(level=0, inplace=False)
    count_credit_card_loans_df = count_credit_card_loans_df.rename(index=str, columns = {'SK_ID_PREV': 'COUNT_CREDIT_CARD_LOANS_(CREDIT_CARD)'})
    # Join to main dataframe
    X = X.join(count_credit_card_loans_df.set_index('SK_ID_CURR'), on='SK_ID_CURR', how='left')
    del count_credit_card_loans_df
    gc.collect()
    
    
    return X
def engineer_combined_features(X):
    
    # 'COUNT_PREVIOUS_LOANS_(COMBINED)': Total number of previous loans 
    # received by each borrower. Includes loans previously received from 
    # Home Credit, and loans received from other lenders.
    X['COUNT_PREVIOUS_LOANS_(COMBINED)'] = X['COUNT_BUREAU_LOANS_(BUREAU)'] + X['COUNT_POS_CASH_LOANS_(POS_CASH)'] + X['COUNT_CREDIT_CARD_LOANS_(CREDIT_CARD)']
        
    return X
to_drop = [
    # Categorical features from main data table.
    'EMERGENCYSTATE_MODE', 
    'HOUSETYPE_MODE',
    
    # Binary categorical features from main data table.
    'FLAG_MOBIL',
    'FLAG_DOCUMENT_2',
    'FLAG_DOCUMENT_4',
    'FLAG_DOCUMENT_7',
    'FLAG_DOCUMENT_10',
    'FLAG_DOCUMENT_12',
    'FLAG_DOCUMENT_17',
    'FLAG_DOCUMENT_19',
    'FLAG_DOCUMENT_20',
    'FLAG_DOCUMENT_21',
    
    # Features used for feature engineering, but not necessary 
    # to keep as stand-alone
    'COUNT_POS_CASH_LOANS_(POS_CASH)',
]

def drop_features(X, to_drop):
    
    X.drop(to_drop, axis=1, inplace=True)
    
    return X
# Load, preprocess and engineer features from all data tables:

# Main Table
print('Processing Main table.')
application_train = pd.read_csv("../input/application_train.csv")
application_test = pd.read_csv("../input/application_test.csv")
train_IDs = application_train['SK_ID_CURR']
test_IDs = application_test['SK_ID_CURR']
X, y_train = preprocess_main(application_train, application_test)
X = engineer_main_features(X)
del application_train, application_test
gc.collect()
print('Main table complete.')

# Bureau Table
print('Processing Bureau table.')
bureau = pd.read_csv("../input/bureau.csv")
bureau = replace_XNA_XAP(bureau)
X = engineer_bureau_features(X, bureau)
print('Bureau table complete.')

# Bureau Balance Table
print('Processing Bureau Balance table.')
bureau_balance = pd.read_csv("../input/bureau_balance.csv")
bureau_balance = preprocess_bureau_balance(bureau_balance, bureau)
X = engineer_bureau_balance_features(X, bureau_balance)
del bureau, bureau_balance
gc.collect()
print('Bureau Balance table complete.')

# Previous Application Table
print('Processing Previous Application table.')
previous_application = pd.read_csv("../input/previous_application.csv")
previous_application = preprocess_previous_application(previous_application)
X = engineer_previous_application_features(X, previous_application)
del previous_application
gc.collect()
print('Previous Application table complete.')

# POS Cash Balance Table
print('Processing POS Cash Balance table.')
pos_cash_balance = pd.read_csv("../input/POS_CASH_balance.csv")
pos_cash_balance = replace_XNA_XAP(pos_cash_balance)
X = engineer_pos_cash_balance_features(X, pos_cash_balance)
del pos_cash_balance
gc.collect()
print('POS Cash Balance table complete.')

# Installments Payments Table
print('Processing Installments Payments table.')
# No 'XNA', 'XAP' entries in installments payments data table.
installments_payments = pd.read_csv("../input/installments_payments.csv")
X = engineer_installments_payments_features(X, installments_payments)
del installments_payments
gc.collect()
print('Installments Payments table complete.')

# Credit Card Balance Table
print('Processing Credit Card Balance table.')
credit_card_balance = pd.read_csv("../input/credit_card_balance.csv")
credit_card_balance = replace_XNA_XAP(credit_card_balance)
X = engineer_credit_card_balance_features(X, credit_card_balance)
del credit_card_balance
gc.collect()
print('Credit Card Balance table complete.')

# Make new features from features from all tables
X = engineer_combined_features(X)

# Replace any spaces in column names with underscores.
# (In order to eventually drop any features with low 
#  feature importances. LightGBM automatically fills 
#  spaces in feature names with underscores, so in 
#  order to be able to drop these features, we need 
#  to reformat their names now, so that LightGBM 
#  won't have to.)
X.columns = X.columns.str.replace(' ','_')

# Drop unhelpful features
X = drop_features(X, to_drop)

# Split dataset back into its training and testing segments
X_train = X[X['SK_ID_CURR'].isin(train_IDs)]
X_test = X[X['SK_ID_CURR'].isin(test_IDs)]
X_test.reset_index(drop=True, inplace=True)

# Get borrower IDs ('SK_ID_CURR') for the test set, and then 
# drop the 'SK_ID_CURR' column from both train and test sets.
X_test_borrower_IDs = X_test['SK_ID_CURR']
X_train.drop('SK_ID_CURR', axis=1, inplace=True)
X_test.drop('SK_ID_CURR', axis=1, inplace=True)
del X
gc.collect()

print('Preprocessing complete.')
print('Training table shape: {}'.format(X_train.shape))
print('Testing table shape: {}'.format(X_test.shape))

display(X_train.head())
def display_feature_importances(feat_importance_df, for_cv=False):

    cols = feat_importance_df[['Feature Name', 'Importance Value']].groupby('Feature Name').mean().sort_values(by='Importance Value', ascending=False)[:50].index
    
    best_features = feat_importance_df.loc[feat_importance_df['Feature Name'].isin(cols)]
    
    # The title of the plot depends on whether LightGBM 
    # model training happened during cross-validation, or 
    # while generating test set predictions.
    if for_cv: 
        plot_title = 'LightGBM Top 50 Features (Avg Over 5 Folds)'
    else:
        plot_title = 'LightGBM Top 50 Features (Avg Over 5 Training Rounds)'

    plt.figure(figsize=(8,10), dpi=200)
    sns.barplot(x='Importance Value', y='Feature Name', data=best_features.sort_values(by='Importance Value', ascending=False))
    plt.title(plot_title)
    plt.xlabel('Importance Value')
    plt.ylabel('Feature Name')
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig('LightGBM_Feature_Importances.png')
    plt.show()
# Adapted from notebook created by olivier (https://www.kaggle.com/ogrellier) 
# on kaggle:
# https://www.kaggle.com/ogrellier/python-target-encoding-for-categorical-features

def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))

def target_encode(train_series=None, 
                  test_series=None, 
                  target=None, 
                  min_samples_leaf=1, 
                  smoothing=1,
                  noise_level=0):
    """
    Smoothing is computed like in the following paper by Daniele Micci-Barreca:
    https://kaggle2.blob.core.windows.net/forum-message-attachments/225952/7441/high%20cardinality%20categoricals.pdf
    
    train_series (pd.Series): training categorical feature as a pd.Series
    test_series (pd.Series): test categorical feature as a pd.Series
    target (pd.Series): target data as a pd.Series
    min_samples_leaf (int) : minimum samples to take category average into account
    smoothing (int) : smoothing effect to balance categorical average vs prior  
    
    Returns: Smoothed, target mean encoded train and test series for 
             a given feature.
    """ 
    
    assert len(train_series) == len(target)
    assert train_series.name == test_series.name
    
    temp = pd.concat([train_series, target], axis=1)
    
    # Compute target mean 
    averages = temp.groupby(by=train_series.name)[target.name].agg(['mean', 'count'])
    
    # Compute smoothing
    smoothing = 1 / (1 + np.exp(-(averages['count'] - min_samples_leaf) / smoothing))
    
    # Apply average function to all target data
    prior = target.mean()
    
    # The bigger the count the less full_avg is taken into account
    averages[target.name] = prior * (1 - smoothing) + averages['mean'] * smoothing
    averages.drop(['mean', 'count'], axis=1, inplace=True)
    
    # Apply averages to train series
    ft_train_series = pd.merge(
        train_series.to_frame(train_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=train_series.name,
        how='left')['average'].rename(train_series.name + '_target_encoded').fillna(prior)
    
    # pd.merge does not keep the index so restore it
    ft_train_series.index = train_series.index 
    
    # Apply averages to test series
    ft_test_series = pd.merge(
        test_series.to_frame(test_series.name),
        averages.reset_index().rename(columns={'index': target.name, target.name: 'average'}),
        on=test_series.name,
        how='left')['average'].rename(train_series.name + '_target_encoded').fillna(prior)
    
    # pd.merge does not keep the index so restore it
    ft_test_series.index = test_series.index
    
    # Return slightly-noised, target-mean encoded train series and test series.
    return add_noise(ft_train_series, noise_level), add_noise(ft_test_series, noise_level)
def train_lgbm(params, X_train, y_train, X_val=None, y_val=None, for_cv = False, 
               max_boost_rounds=500, early_stop_rounds=None):
    
    # Convert training set to lightgbm dataset format.
    lightgbm_training = lgb.Dataset(X_train, label=y_train)
    del X_train, y_train
    gc.collect()
    
    # Check to see if training is for cross-validation, or 
    # whether it's for generating test predictions
    if for_cv:
        
        # Train the LightGBM classifier for cross-validation:
        
        # Make sure to convert validation set to lightgbm dataset format.
        lightgbm_val = lgb.Dataset(X_val, label=y_val)
        del X_val, y_val
        gc.collect()
        
        clf_lgb = lgb.train(params=params, 
                            train_set=lightgbm_training, 
                            valid_sets=[lightgbm_val, 
                                        lightgbm_training],
                            valid_names=['val', 'train'], 
                            num_boost_round=max_boost_rounds, 
                            categorical_feature='auto',
                            early_stopping_rounds=early_stop_rounds, verbose_eval=False)
        
    else:

        # Train the LightGBM classifier for generating test 
        # set predictions:
        clf_lgb = lgb.train(params=params, 
                            train_set=lightgbm_training, 
                            valid_sets=None,
                            num_boost_round=max_boost_rounds, 
                            categorical_feature='auto',
                            early_stopping_rounds=None,
                            verbose_eval=False)
    
    return clf_lgb
# Intuition regarding LightGBM parameters comes from 
# the helpful info on this site:
# https://sites.google.com/view/lauraepp/parameters
params = {}
params['learning_rate'] = 0.007 #0.02 #0.2 
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'auc'

# Tune these two together: they are the two 
# most important parameters for preventing 
# overfitting. Theoretical max number of leaves  = 2^(max_depth) - 1.
# Tip: "adjust depth accordingly by allowing a slightly higher depth 
# than the theoretical number of leaves."
params['max_depth'] = 5   #7   #6  #5  #2 #-1 
params['num_leaves'] = 31 #127 #63 #31 #3 #31 

# Row sampling (bagging fraction) and column sampling by tree 
# (feature fraction) are the next two most important parameters 
# to tune for LightGBM.
params['bagging_fraction'] = 1.0
params['feature_fraction'] = 0.3 #0.4 #0.5

# Next most important hyperparameter for reducing overfitting.
# How many observations must exist before putting them in a leaf.
# Increase as dataset gets larger. 
# Default is 20
params['min_data_in_leaf'] = 400 #500 #320 #200

# Helps to build deep trees while avoiding building useless 
# branches of the trees (overfitting).
# Default is 0.0
params['min_gain_to_split'] = 0.007 #0.006

# Maximum number of unique values per features. "LightGBM optimizes 
# the dataset storage depending on the binary power of the parameter 
# max_bin. max_bin = 255 allows to uses 8 bits to store a single value.
# max_bin = 63 would require only 6 bits, while max_bin = 15 would 
# require only 4 bits."
#
# "Binning acts both a regularization method and a training speed up 
# method. By providing less unique values per feature, the model can 
# be trained significantly faster without a large loss in performance.
# In cases where the dataset is closer to a synthetic dataset, the 
# model might perform even better than without binning."
#
# Default value is 255 if using CPU (and 16 if using GPU).
params['max_bin'] = 255

# Penalize large errors without making features more sparse.
params['lambda_l2'] = 22 #10

# Penalize large errors and increase feature sparsity in 
# the process.
params['lambda_l1'] = 8 #30 #22 #16

# Make the model more cost-sensitive when training.
# Potentially delivers higher performance on ranking 
# tasks such as for AUC.
# Default value is 1.
params['scale_pos_weight'] = 1 #2

# Using LightGBM's built-in CV.

# K-Fold CV Parameters
N_FOLDS = 5
CV_RANDOM_SEED = 17

# LightGBM Parameters for CV
MAX_BOOST_ROUNDS = 20000
EARLY_STOP_ROUNDS = 200

# LightGBM Parameters for handling categorical features.
# I cannot use my custom target encoding code with 
# LightGBM's built-in CV. These parameters adjust how 
# LightGBM handles cardinally encoded categorical 
# features:

# Applying jitter to categorical features so that they 
# do not immediately overfit.
# Default value is 10
params['cat_smooth'] = 10

# Minimumum number of data per categorical group. 
# Also helps prevent overfitting to categorical features.
# Default value is 100
params['min_data_per_group'] = 100

# Limits the max threshold points in categorical features.
# Default value is 32
params['max_cat_threshold'] = 4

# L2 regularization for categorical splits.
# Default value is 10
params['cat_l2'] = 11

# Periodically change the LightGBM parameters 
# seed value while tuning.
params['seed'] = 8

def lgb_built_in_cv(X_train, y_train, params, n_folds, num_boost_round, early_stop_rounds, cv_random_seed):
    dtrain = lgb.Dataset(X_train, y_train)
    cv_results = lgb.cv(params,
                  dtrain,
                  categorical_feature = 'auto',
                  nfold=n_folds,
                  stratified=True,
                  num_boost_round=num_boost_round,
                  early_stopping_rounds=early_stop_rounds,
                  verbose_eval=100,
                  seed = cv_random_seed,
                  show_stdv=True)
    print('Highest Average ROC AUC Score (across 5 folds): {}'.format(max(cv_results['auc-mean'])))
    print('Round of Highest Average ROC AUC Score Achieved: {}'.format(np.argmax(cv_results['auc-mean'])))

# Commented out, as this doesn't need to be run when generating final test predictions. 
# lgb_built_in_cv(X_train, y_train, params, N_FOLDS, MAX_BOOST_ROUNDS, EARLY_STOP_ROUNDS, CV_RANDOM_SEED)
def cross_validate(lgbm_params, X_train, y_train, n_folds, max_boost_rounds=500, 
                   early_stop_rounds=50, tme_min_samples_leaf=1, tme_smoothing=1, 
                   tme_noise_level=0, cv_random_seed=42):
    
    # Stratified K-fold cross val
    cv = StratifiedKFold(n_splits=n_folds, random_state=cv_random_seed)

    # Keep track of best validation score and best round 
    # in each fold.
    cv_scores = []
    best_rounds = []
    fold_count = 0
    
    # Store the feature importances after each training fold. 
    # Will eventually plot the top 50 importances, averaged 
    # over all five training folds.
    feat_importance_df = pd.DataFrame()

    for train_index, val_index in cv.split(X_train, y_train):
        fold_count +=1
        X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
        y_train_fold, y_val_fold = y_train.iloc[train_index], y_train.iloc[val_index]

        # Conduct target mean encoding for categorical features:

        # Get all the categorical features in the dataset
        cat_feats = X_train_fold.select_dtypes(['category']).columns.tolist()
        
        # Apply target mean encoding to each of these features, in both 
        # the train and validation sets
        for feature in cat_feats:
            # Get target mean encoded series for train and validation segments
            # of for the feature.
            encoded_train, encoded_val = target_encode(train_series=X_train_fold[feature], 
                                                       test_series=X_val_fold[feature], 
                                                       target=y_train_fold, 
                                                       min_samples_leaf=tme_min_samples_leaf,
                                                       smoothing=tme_smoothing,
                                                       noise_level=tme_noise_level)

            # Merge the target mean encoded train, val series to 
            # the train, val dataframes.
            X_train_fold = pd.merge(X_train_fold, encoded_train.to_frame(encoded_train.name), 
                               how='left', left_on=X_train_fold.index, right_index=True)
            X_val_fold = pd.merge(X_val_fold, encoded_val.to_frame(encoded_val.name), 
                             how='left', left_on=X_val_fold.index, right_index=True)

        # Drop all original categorical feats from the train and 
        # validation dataframes, now that they have all been 
        # successfully target mean encoded.
        X_train_fold.drop(cat_feats, axis=1, inplace=True)
        X_val_fold.drop(cat_feats, axis=1, inplace=True)
        
        # Build and train the LightGBM classifier
        clf_lgb = train_lgbm(lgbm_params, X_train_fold, y_train_fold, X_val_fold, y_val_fold, for_cv = True, max_boost_rounds=max_boost_rounds, early_stop_rounds=early_stop_rounds)
        
        # Save the feature importances after each training fold.
        fold_feat_importance_df = pd.DataFrame()
        fold_feat_importance_df['Feature Name'] = clf_lgb.feature_name()
        fold_feat_importance_df['Importance Value'] = clf_lgb.feature_importance(importance_type='split', iteration=-1)
        fold_feat_importance_df['Fold'] = fold_count
        feat_importance_df = pd.concat([feat_importance_df, fold_feat_importance_df], axis=0)
        
        # Number of rounds until best validation score reached inside fold.
        best_round = clf_lgb.best_iteration
        # Best validation ROC AUC score.
        best_val_roc_auc_score = clf_lgb.best_score['val']['auc']
        # Training ROC AUC score achieved on best round.
        best_train_roc_auc_score = clf_lgb.best_score['train']['auc']
        
        # Keep track of best validation set score and best round 
        # in each fold. They will be averaged and printed out after 
        # all five folds have run.
        cv_scores.append(best_val_roc_auc_score)
        best_rounds.append(best_round)
        
        print("Fold: {}  Validation AUC: {}  Train AUC: {}  Diff Train-Val: {}  Best Round: {}".format(fold_count, best_val_roc_auc_score, best_train_roc_auc_score, round(best_train_roc_auc_score - best_val_roc_auc_score,4), best_round))
        
    print("Average CV Score across 5 folds: {}    Average Best Round: {}".format(np.mean(cv_scores), np.mean(best_rounds)))
    
    # Save the dataframe containing all the feature importances.
    # So that it can be printed out in a cell below.
    global all_feature_importances
    all_feature_importances = feat_importance_df
    
    # Display a graph of LightGBM feature importances, averaged 
    # over the five training folds.
    display_feature_importances(feat_importance_df, for_cv=True)
# Using serial stratified K-Fold CV

# K-Fold CV Parameters
N_FOLDS = 5
CV_RANDOM_SEED = 17

# LightGBM Parameters for CV
MAX_BOOST_ROUNDS = 20000
EARLY_STOP_ROUNDS = 200

# Periodically change the LightGBM parameters 
# seed value while tuning.
params['seed'] = 8

# If running this CV immediately after running the 
# LightGBM built-in CV above: 
#
# Must reset the 'categorical_feature' param to its 
# default value of an empty string. This is because the serial
# CV I wrote uses its own target mean encoding, so I don't want 
# LightGBM to handle categorical features.
params['categorical_feature'] = ''

# Also must reset the following parameters concerning LightGBM's 
# handling of categorical features to their default values.
params['cat_smooth'] = 10.0
params['min_data_per_group'] = 100
params['max_cat_threshold'] = 32
params['cat_l2'] = 10.0

# Target Mean Encoding Parameters
MIN_SAMPLES_LEAF = 1 #7
SMOOTHING = 1 #7
NOISE_LEVEL = 0

# Commented out, as this doesn't need to be run when generating final test predictions.
# # Perform cross validation (using my target mean encoding methods):
# cross_validate(params, X_train, y_train, n_folds=N_FOLDS, max_boost_rounds=MAX_BOOST_ROUNDS, 
#                early_stop_rounds=EARLY_STOP_ROUNDS, tme_min_samples_leaf=MIN_SAMPLES_LEAF, 
#                tme_smoothing=SMOOTHING, tme_noise_level=NOISE_LEVEL,
#                cv_random_seed=CV_RANDOM_SEED)
# feat_impt_in_order = all_feature_importances[['Feature Name', 'Importance Value']].groupby(
#     'Feature Name').mean().sort_values(by='Importance Value', ascending=False).reset_index(level=0, inplace=False)

# display(feat_impt_in_order)
def mean_rank_predictions(y_scores):
    # Blend the five sets of predictions (one set of predictions comes from 
    # one of the training rounds) by ranking each set (column) of predictions, 
    # and then taking the average of each row's five ranks. Finally, scale these 
    # averages to the range [0,1].
    pred_ranks = y_scores.rank(axis=0, method='average')

    # Generate a list that contains the means of all rows' five ranks.
    mean_ranks = pred_ranks.mean(axis=1)

    # Scale the list of means of ranks to the range [0,1]
    mean_ranks_scaler = MinMaxScaler()
    mean_ranked_predictions = mean_ranks_scaler.fit_transform(mean_ranks.values.reshape(-1,1))[:,0]
    
    return mean_ranked_predictions
def write_pred_to_csv(borrower_IDs, predictions, predictions_version):

    # Create the CSV file that will be saved
    file_output = 'kaggle_home_credit_submission_{}.csv'.format(predictions_version)
    # Write to the CSV file
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['SK_ID_CURR','TARGET'])
        # Write a row for each borrower that contains the 
        # prediction probability of their label.
        for index, value in borrower_IDs.iteritems():
            writer.writerow([value, predictions[index]])
# The number of times the lightgbm classifier 
# will be trained and a set of test predictions 
# generated.
NUM_TRAIN_ROUNDS = 5

# Use a different LightGBM seed value for each 
# training round.
SEED_VALUES = [42, 10, 2018, 38, 28]

# Choose number of boosting rounds parameter to be 10% greater 
# than the average best number of boosting rounds during 
# K-fold cross validation. Learned this heuristic from 
# Silogram: https://www.kaggle.com/c/home-credit-default-risk/discussion/58332#348689
MAX_BOOST_ROUNDS = 8615    # 7832 * 1.1 = 8615.2

# If running this prediction generator after running the LightGBM 
# built-in CV above, must reset the 'categorical_feature' param to 
# its default value of an empty string. The training and prediction 
# generator I wrote uses its own target mean encoding, so I don't 
# want LightGBM to handle categorical features.
params['categorical_feature'] = ''

# Also must reset the following parameters concerning LightGBM's 
# handling of categorical features to their default values.
params['cat_smooth'] = 10.0
params['min_data_per_group'] = 100
params['max_cat_threshold'] = 32
params['cat_l2'] = 10.0

# Target Mean Encoding Parameters
MIN_SAMPLES_LEAF = 1 #7
SMOOTHING = 1 #7
NOISE_LEVEL = 0

# Predictions version number 
VERSION = 'final'
# Ensure there is a LightGBM seed value for each training round.
assert NUM_TRAIN_ROUNDS == len(SEED_VALUES)

# Conduct target mean encoding for categorical features:
print('Begin target encoding.')
# Get all the categorical features in the dataset
cat_feats = X_train.select_dtypes(['category']).columns.tolist()

# Apply target mean encoding to each of these features, in both 
# the train and validation sets
for feature in cat_feats:
    # Get target mean encoded series for train and test segments
    # of for the feature.
    encoded_train, encoded_test = target_encode(train_series=X_train[feature], 
                                               test_series=X_test[feature], 
                                               target=y_train, 
                                               min_samples_leaf=MIN_SAMPLES_LEAF,
                                               smoothing=SMOOTHING,
                                               noise_level=NOISE_LEVEL)

    # Merge the target mean encoded train, val series to 
    # the train, val dataframes.
    X_train = pd.merge(X_train, encoded_train.to_frame(encoded_train.name), how='left', left_on=X_train.index, right_index=True)
    X_test = pd.merge(X_test, encoded_test.to_frame(encoded_test.name), how='left', left_on=X_test.index, right_index=True)

# Drop all original categorical feats from the train and 
# validation dataframes, now that they have all been 
# successfully target mean encoded.
X_train.drop(cat_feats, axis=1, inplace=True)
X_test.drop(cat_feats, axis=1, inplace=True)
print('Target encoding completed.')

# List that will contain the testing set predictions 
# computed by the classifier. Each training round, a new 
# set of predictions is appended to this list.
y_score_list = []

# Store the feature importances after each training fold. 
# Will eventually plot the top 50 importances, averaged 
# over all five training folds.
feat_importance_df = pd.DataFrame()

# Build a list of titles (column names) for the training rounds.
# To be used in the dataframe that will contain the predictions 
# made in each round of training.
training_round_names = []
for i in range (NUM_TRAIN_ROUNDS):
    round_name = 'TRAINING_ROUND_' + str(i + 1)
    training_round_names.append(round_name)

# Train a classifier on the entire training set multiple times
for i in range(NUM_TRAIN_ROUNDS):
    print('Begin Training Round {} of {}.'.format(i + 1, NUM_TRAIN_ROUNDS))

    # Update the LightGBM classifier's seed value with a new 
    # number each training round.
    params['seed'] = SEED_VALUES[i]

    # Build a LightGBM classifier and fit it to all the training data.
    clf_lgb = train_lgbm(params, X_train, y_train, X_val=None, y_val=None, for_cv = False, max_boost_rounds=MAX_BOOST_ROUNDS, early_stop_rounds=None)

    # The prediction probabilities made on testing set
    clf_y_score = clf_lgb.predict(X_test)

    # Append the predictions to a list containing all the test predictions
    # from each training round.
    y_score_list.append(pd.Series(clf_y_score, name=training_round_names[i]))

    # Save the feature importances after each training round.
    rd_feat_importance_df = pd.DataFrame()
    rd_feat_importance_df['Feature Name'] = clf_lgb.feature_name()
    rd_feat_importance_df['Importance Value'] = clf_lgb.feature_importance(importance_type='split', iteration=-1)
    rd_feat_importance_df['Round'] = i + 1
    feat_importance_df = pd.concat([feat_importance_df, rd_feat_importance_df], axis=0)

    print('Training Round {} of {} complete.'.format(i + 1, NUM_TRAIN_ROUNDS))

# Display a graph of LightGBM feature importances, averaged 
# over the five training rounds.
display_feature_importances(feat_importance_df, for_cv=False)

# Build a dataframe that holds each training round's predictions
y_scores = pd.concat(y_score_list, axis=1)

# Use mean ranking to blend the five sets of predictions. 
final_predictions = mean_rank_predictions(y_scores)

# Save predictions to a CSV file:
write_pred_to_csv(X_test_borrower_IDs, final_predictions, VERSION)

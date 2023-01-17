# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
!ls ../input
application_train = pd.read_csv("../input/application_train.csv")
application_train.head()
%matplotlib inline
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


from scipy import stats
from scipy.stats import norm, skew #for some statistics


pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x)) #Limiting floats output to 3 decimal points


from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8")) #check the files available in the dire
application_train['TARGET'].value_counts()
df_test = pd.read_csv('../input/application_test.csv')
print(df_test.shape)
df_test['TARGET'] = -11
df = pd.concat([application_train, df_test])
del application_train, df_test
df.shape
df = df[df['CODE_GENDER'] != 'XNA']
live = [_f for _f in df.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
df['NEW_LIVE_IND_SUM'] = df[live].sum(axis=1)
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_FAM_MEMBERS'])
df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
df['CHILDREN_RATIO'] = df['CNT_CHILDREN'] / (1 + df['CNT_FAM_MEMBERS'])
df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / (1 + df['DAYS_BIRTH'])
df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / (1 + df['AMT_ANNUITY'])
df['NEW_CREDIT_TO_GOODS_RATIO'] = df['AMT_CREDIT'] / (1 + df['AMT_GOODS_PRICE'])
df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
df['NEW_EMPLOY_TO_BIRTH_RATIO'] = df['DAYS_EMPLOYED'] / (1 + df['DAYS_BIRTH'])
df['NEW_ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / (1 + df['AMT_INCOME_TOTAL'])
df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
df['NEW_CAR_TO_BIRTH_RATIO'] = df['OWN_CAR_AGE'] / (1 + df['DAYS_BIRTH'])
df['NEW_CAR_TO_EMPLOY_RATIO'] = df['OWN_CAR_AGE'] / ( 1 + df['DAYS_EMPLOYED'])
df['NEW_PHONE_TO_BIRTH_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / (1 + df['DAYS_BIRTH'])
df['NEW_PHONE_TO_EMPLOY_RATIO'] = df['DAYS_LAST_PHONE_CHANGE'] / ( 1 + df['DAYS_EMPLOYED'])
df['NEW_CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / (1 + df['AMT_INCOME_TOTAL'])
df.shape
idx = df['NAME_CONTRACT_TYPE'] == 'Cash loans'
df['cash_loans'] = idx.apply(int)
idx = df['NAME_CONTRACT_TYPE'] == 'Revolving loans'
df['revolving_loans'] = idx.apply(int)
df = df.drop(['NAME_CONTRACT_TYPE'], axis=1)
idx = df['CODE_GENDER'] == 'F'
df['gender_f'] = idx.apply(int)
df = df.drop(['CODE_GENDER'], axis=1)
idx = df['FLAG_OWN_CAR'] == 'Y'
df['have_car'] = idx.apply(int)
df = df.drop(['FLAG_OWN_CAR'], axis=1)
idx = df['FLAG_OWN_REALTY'] == 'Y'
df['have_realty'] = idx.apply(int)
df = df.drop(['FLAG_OWN_REALTY'], axis=1)
# make a Classification for “CNT_CHILDREN > 2”
idx = df['CNT_CHILDREN'] == 0
df['have_no_children'] = idx.apply(int)
idx = df['CNT_CHILDREN'] == 1
df['have_one_children'] = idx.apply(int)
idx = df['CNT_CHILDREN'] == 2
df['have_two_children'] = idx.apply(int)
idx = df['CNT_CHILDREN'] > 2
df['have_gt_two_children'] = idx.apply(int)
df = df.drop(['CNT_CHILDREN'], axis=1)
df['income_credit_ratio'] = df['AMT_INCOME_TOTAL'] / (1 + df['AMT_CREDIT'])
df['income_annuity_ratio'] = df['AMT_INCOME_TOTAL'] / (1 + df['AMT_GOODS_PRICE'])
# encode”
idx = df['NAME_TYPE_SUITE'] == 'Unaccompanied'
df['someone_with_unaccompanied'] = idx.apply(int)
idx = df['NAME_TYPE_SUITE'] == 'Family'
df['someone_with_family'] = idx.apply(int)
idx = df['NAME_TYPE_SUITE'] == 'Spouse, partner'
df['someone_with_Spouse_partner'] = idx.apply(int)
idx = df['NAME_TYPE_SUITE'] == 'Children'
df['someone_with_children'] = idx.apply(int)
idx = df['NAME_TYPE_SUITE'] == 'Other_A'
df['someone_with_other_a'] = idx.apply(int)
idx = df['NAME_TYPE_SUITE'] == 'Other_B'
df['someone_with_other_b'] = idx.apply(int)
idx = df['NAME_TYPE_SUITE'] == 'Group of people'
df['someone_with_group_people'] = idx.apply(int)
df = df.drop(['NAME_TYPE_SUITE'], axis=1)
# remove samples “aternity leave”
idx1 = df["NAME_INCOME_TYPE"]=='Maternity leave'
df = df[~idx1]
# encode
idx = df['NAME_INCOME_TYPE'] == 'Working'
df['income_type_working'] = idx.apply(int)
idx = df['NAME_INCOME_TYPE'] == 'Commercial associate'
df['income_type_com_ass'] = idx.apply(int)
idx = df['NAME_INCOME_TYPE'] == 'Pensioner'
df['income_type_pensioner'] = idx.apply(int)
idx = df['NAME_INCOME_TYPE'] == 'Unemployed'
df['income_type_Unemployed'] = idx.apply(int)
idx = df['NAME_INCOME_TYPE'] == 'State servant'
df['income_type_state_servant'] = idx.apply(int)
idx = df['NAME_INCOME_TYPE'] == 'Student'
df['income_type_student'] = idx.apply(int)
idx = df['NAME_INCOME_TYPE'] == 'Businessman'
df['income_type_businessman'] = idx.apply(int)
df = df.drop(['NAME_INCOME_TYPE'], axis=1)
idx = df['NAME_EDUCATION_TYPE'] == 'Secondary / secondary special'
df['education_secondary_special'] = idx.apply(int)
idx = df['NAME_EDUCATION_TYPE'] == 'Higher education'
df['education_higher_education'] = idx.apply(int)
idx = df['NAME_EDUCATION_TYPE'] == 'Incomplete higher'
df['education_incomplete_higher'] = idx.apply(int)
idx = df['NAME_EDUCATION_TYPE'] == 'Lower secondary'
df['education_lower_secondary'] = idx.apply(int)
idx = df['NAME_EDUCATION_TYPE'] == 'Academic degree'
df['education_academic_degree'] = idx.apply(int)
df = df.drop(['NAME_EDUCATION_TYPE'], axis=1)
idx = df['NAME_FAMILY_STATUS'] == 'Married'
df['family_status_merried'] = idx.apply(int)
idx = df['NAME_FAMILY_STATUS'] == 'Single / not married'
df['family_status_single'] = idx.apply(int)
idx = df['NAME_FAMILY_STATUS'] == 'Civil marriage'
df['family_status_civil_merriage'] = idx.apply(int)
idx = df['NAME_FAMILY_STATUS'] == 'Separated'
df['family_status_separated'] = idx.apply(int)
idx = df['NAME_FAMILY_STATUS'] == 'Widow'
df['family_status_widow'] = idx.apply(int)
df = df.drop(['NAME_FAMILY_STATUS'], axis=1)
idx = df['NAME_HOUSING_TYPE'] == 'House / apartment'
df['housing_type_house'] = idx.apply(int)
idx = df['NAME_HOUSING_TYPE'] == 'With parents'
df['housing_type_parents'] = idx.apply(int)
idx = df['NAME_HOUSING_TYPE'] == 'Municipal apartment'
df['housing_type_municipal'] = idx.apply(int)
idx = df['NAME_HOUSING_TYPE'] == 'Rented apartment'
df['housing_type_rented'] = idx.apply(int)
idx = df['NAME_HOUSING_TYPE'] == 'Office apartment'
df['housing_type_office'] = idx.apply(int)
idx = df['NAME_HOUSING_TYPE'] == 'Co-op apartment'
df['housing_type_coop'] = idx.apply(int)
df = df.drop(['NAME_HOUSING_TYPE'], axis=1)
idx1 = df['have_car'] == 1 # have car
idx2 = df['have_car'] == 0 # no car
idx3 = ~df['OWN_CAR_AGE'].isnull() # have car age
idx4 = idx2 & idx3 # nocar but have car age
idx5 = idx2 & (~idx3) # no car and no car age
df['car_age_with_weight'] = 0
df.loc[idx4,['car_age_with_weight']] = df.loc[idx4,['OWN_CAR_AGE']] * 0.5
df.loc[idx5,['car_age_with_weight']] = 0
df.loc[idx1,['car_age_with_weight']] = df[idx1]['OWN_CAR_AGE']
value_list = df['OCCUPATION_TYPE'].unique().tolist()
value_list.remove(value_list[4])
for col_name in value_list:
    idx = df['OCCUPATION_TYPE'] == col_name
    col_name = col_name.replace(' ', '_')
    name = 'occupation_type_' + col_name
    df[name] = idx.apply(int)
df = df.drop(['OCCUPATION_TYPE'], axis=1)
idx = df['CNT_FAM_MEMBERS'] == 1
df['fam_members_type_1'] = idx.apply(int)
idx = df['CNT_FAM_MEMBERS'].isin([2.0,3.0,4.0])
df['fam_members_type_2_4'] = idx.apply(int)
idx = df['CNT_FAM_MEMBERS'] > 4
df['fam_members_type_5_plus'] = idx.apply(int)

df['fam_members_credit_ratio'] = df['AMT_CREDIT'] / (1 + df['CNT_FAM_MEMBERS'])
df['fam_members_annuity_ratio'] = df['AMT_ANNUITY'] / ( 1+ df['CNT_FAM_MEMBERS'])
idx = df['REGION_RATING_CLIENT'] == 1
df['REGION_RATING_CLIENT_1'] = idx.apply(int)
idx = df['REGION_RATING_CLIENT'] == 2
df['REGION_RATING_CLIENT_2'] = idx.apply(int)
idx = df['REGION_RATING_CLIENT'] == 3
df['REGION_RATING_CLIENT_3'] = idx.apply(int)
df = df.drop(['REGION_RATING_CLIENT'], axis=1)
# Modification(-1 is 2) 
idx =df['REGION_RATING_CLIENT_W_CITY'] == -1
df.loc[idx, 'REGION_RATING_CLIENT_W_CITY'] = 2

idx = df['REGION_RATING_CLIENT_W_CITY'] == 1
df['REGION_RATING_CLIENT_W_CITY_1'] = idx.apply(int)
idx = df['REGION_RATING_CLIENT_W_CITY'] == 2
df['REGION_RATING_CLIENT_W_CITY_2'] = idx.apply(int)
idx = df['REGION_RATING_CLIENT_W_CITY'] == 3
df['REGION_RATING_CLIENT_W_CITY_3'] = idx.apply(int)
df = df.drop(['REGION_RATING_CLIENT_W_CITY'], axis=1)
idx = df['WEEKDAY_APPR_PROCESS_START'].isin(['SUNDAY', 'SATURDAY'])
df['APPR_PROCESS_IN_WEEKEND'] = idx.apply(int)
df = df.drop(['WEEKDAY_APPR_PROCESS_START'], axis=1)
# make a Classification
# 8-12  13-18 19-24-7
idx = df['HOUR_APPR_PROCESS_START'].isin([8,9,11,12])
df['HOUR_APPR_PROCESS_START_morning'] = idx.apply(int)
idx = df['HOUR_APPR_PROCESS_START'].isin([13,14,15,16,17,18])
df['HOUR_APPR_PROCESS_START_afternoon'] = idx.apply(int)
idx = df['HOUR_APPR_PROCESS_START'].isin([0,1,2,3,4,5,6,7,19,20,21,22,23])
df['HOUR_APPR_PROCESS_START_no_worktime'] = idx.apply(int)
df = df.drop(['HOUR_APPR_PROCESS_START'], axis=1)
value_list = df['ORGANIZATION_TYPE'].unique().tolist()
for col_name in value_list:
    idx = df['ORGANIZATION_TYPE'] == col_name
    col_name = col_name.replace(' ', '_')
    col_name = col_name.replace(':', '_')
    name = 'ORGANIZATION_TYPE_' + col_name
    df[name] = idx.apply(int)
df = df.drop(['ORGANIZATION_TYPE'], axis=1)
df['EXT_SOURCE_1'].fillna(df['EXT_SOURCE_1'].mean(), inplace=True)
df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].mean(), inplace=True)
df['EXT_SOURCE_3'].fillna(df['EXT_SOURCE_3'].mean(), inplace=True)

df['APARTMENTS_AVG'].fillna(df['APARTMENTS_AVG'].mean(), inplace=True)
df['BASEMENTAREA_AVG'].fillna(df['BASEMENTAREA_AVG'].mean(), inplace=True)
df['YEARS_BEGINEXPLUATATION_AVG'].fillna(df['YEARS_BEGINEXPLUATATION_AVG'].mean(), inplace=True)
df['YEARS_BUILD_AVG'].fillna(df['YEARS_BUILD_AVG'].mean(), inplace=True)
df['COMMONAREA_AVG'].fillna(df['COMMONAREA_AVG'].mean(), inplace=True)
df['ELEVATORS_AVG'].fillna(df['ELEVATORS_AVG'].mean(), inplace=True)
df['ENTRANCES_AVG'].fillna(df['ENTRANCES_AVG'].mean(), inplace=True)
col_list = ['FLOORSMAX_AVG',
           'FLOORSMIN_AVG',
           'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG', 'NONLIVINGAPARTMENTS_AVG',
           'NONLIVINGAREA_AVG', 'APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE',
           'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE', 
           'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE',
           'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',
           'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI',
           'FLOORSMIN_MEDI', 'LANDAREA_MEDI', 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',
           'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE'
           ]
for col_name in col_list:
    # print(col_name)
    df[col_name].fillna(df[col_name].mean(), inplace=True)
df = df.drop(['FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE'], axis=1)

df['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(0, inplace=True)
# make a Classification
# 0  1-5 6-10 >10  nan
idx = df['OBS_30_CNT_SOCIAL_CIRCLE'] == 0.0
df['OBS_30_CNT_SOCIAL_CIRCLE_1'] = idx.apply(int)
idx = df['OBS_30_CNT_SOCIAL_CIRCLE'].isin([1.0, 2.0, 3.0, 4.0, 5.0])
df['OBS_30_CNT_SOCIAL_CIRCLE_2'] = idx.apply(int)
idx = df['OBS_30_CNT_SOCIAL_CIRCLE'].isin([6.0, 7.0, 8.0, 9.0, 10.0])
df['OBS_30_CNT_SOCIAL_CIRCLE_3'] = idx.apply(int)
idx = df['OBS_30_CNT_SOCIAL_CIRCLE'] > 10
df['OBS_30_CNT_SOCIAL_CIRCLE_4'] = idx.apply(int)
df = df.drop(['OBS_30_CNT_SOCIAL_CIRCLE'], axis=1)
df['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(0, inplace=True)
idx = df['DEF_30_CNT_SOCIAL_CIRCLE'] == 0.0
df['DEF_30_CNT_SOCIAL_CIRCLE_1'] = idx.apply(int)
idx = df['DEF_30_CNT_SOCIAL_CIRCLE'] == 1.0
df['DEF_30_CNT_SOCIAL_CIRCLE_2'] = idx.apply(int)
idx = df['DEF_30_CNT_SOCIAL_CIRCLE'] == 2.0
df['DEF_30_CNT_SOCIAL_CIRCLE_3'] = idx.apply(int)
df = df.drop(['DEF_30_CNT_SOCIAL_CIRCLE'], axis=1)
df['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(0, inplace=True)
idx = df['OBS_60_CNT_SOCIAL_CIRCLE'] == 0.0
df['OBS_60_CNT_SOCIAL_CIRCLE_1'] = idx.apply(int)
idx = df['OBS_60_CNT_SOCIAL_CIRCLE'].isin([1.0, 2.0, 3.0, 4.0, 5.0])
df['OBS_60_CNT_SOCIAL_CIRCLE_2'] = idx.apply(int)
idx = df['OBS_60_CNT_SOCIAL_CIRCLE'] > 5
df['OBS_60_CNT_SOCIAL_CIRCLE_3'] = idx.apply(int)

df = df.drop(['OBS_60_CNT_SOCIAL_CIRCLE'], axis=1)
df['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(0, inplace=True)
idx = df['DEF_60_CNT_SOCIAL_CIRCLE'] == 0.0
df['DEF_60_CNT_SOCIAL_CIRCLE_1'] = idx.apply(int)
idx = df['DEF_60_CNT_SOCIAL_CIRCLE'] == 1.0
df['DEF_60_CNT_SOCIAL_CIRCLE_2'] = idx.apply(int)
idx = df['DEF_60_CNT_SOCIAL_CIRCLE'] > 1.0
df['DEF_60_CNT_SOCIAL_CIRCLE_3'] = idx.apply(int)
df = df.drop(['DEF_60_CNT_SOCIAL_CIRCLE'], axis=1)
# remove some samples
idx = ~df['DAYS_LAST_PHONE_CHANGE'].isnull()
df = df[idx]

std_val = df['DAYS_LAST_PHONE_CHANGE'].std()
mean_val = df['DAYS_LAST_PHONE_CHANGE'].mean()
df['DAYS_LAST_PHONE_CHANGE'] = df['DAYS_LAST_PHONE_CHANGE'].apply(lambda x: (x - mean_val) / std_val)
# remove some samples
idx1 = df['AMT_REQ_CREDIT_BUREAU_HOUR'] != 4
idx2 = df['AMT_REQ_CREDIT_BUREAU_HOUR'] != 3
idx = idx1 & idx2
df = df[idx]

idx = df['AMT_REQ_CREDIT_BUREAU_DAY'] == 9
df = df[~idx]
idx = df['AMT_REQ_CREDIT_BUREAU_DAY'] == 8
df = df[~idx]
idx = df['AMT_REQ_CREDIT_BUREAU_DAY'] == 6
df = df[~idx]
idx = df['AMT_REQ_CREDIT_BUREAU_DAY'] == 5
df = df[~idx]
idx = df['AMT_REQ_CREDIT_BUREAU_DAY'] == 4
df = df[~idx]
idx = df['AMT_REQ_CREDIT_BUREAU_DAY'] == 3
df = df[~idx]
df['AMT_REQ_CREDIT_BUREAU_DAY'].fillna(0, inplace=True)
# remove some samples
idx = df['AMT_REQ_CREDIT_BUREAU_WEEK'] == 5
df = df[~idx]
idx = df['AMT_REQ_CREDIT_BUREAU_WEEK'] == 7
df = df[~idx]
idx = df['AMT_REQ_CREDIT_BUREAU_WEEK'] == 8
df = df[~idx]
idx = df['AMT_REQ_CREDIT_BUREAU_WEEK'] == 4
df = df[~idx]
idx = df['AMT_REQ_CREDIT_BUREAU_WEEK'] == 6
df = df[~idx]
idx = df['AMT_REQ_CREDIT_BUREAU_WEEK'] == 3
df = df[~idx]
# remove some samples
value_list = [23.0, 22.0, 27.0, 24.0, 19.0,
             18.0, 17.0, 16.0, 15.0, 14.0, 13.0,
             12.0, 11.0, 10.0, 8.0, 9.0,]
for value_ in value_list:
    target_list = df[df['AMT_REQ_CREDIT_BUREAU_MON'] == value_]['TARGET'].value_counts().index.tolist()
    if -11 not in target_list:
        idx = df['AMT_REQ_CREDIT_BUREAU_MON'] == value_
        df = df[~idx]
    else:
        print(value_)
df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(0, inplace=True)
# remove some samples
value_list = [261.0, 19.0, 8.0,]
for value_ in value_list:
    target_list = df[df['AMT_REQ_CREDIT_BUREAU_QRT'] == value_]['TARGET'].value_counts().index.tolist()
    if -11 not in target_list:
        idx = df['AMT_REQ_CREDIT_BUREAU_QRT'] == value_
        df = df[~idx]
    else:
        print(value_)
# 按0填充
df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(0, inplace=True)
# remove some samples
value_list = df['AMT_REQ_CREDIT_BUREAU_YEAR'].unique().tolist()[9:]
for value_ in value_list:
    target_list = df[df['AMT_REQ_CREDIT_BUREAU_YEAR'] == value_]['TARGET'].value_counts().index.tolist()
    if -11 not in target_list:
        idx = df['AMT_REQ_CREDIT_BUREAU_YEAR'] == value_
        df = df[~idx]
    else:
        print(value_)
print(df.shape)
df.head()

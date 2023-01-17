%reload_ext autoreload

%autoreload 2

%matplotlib inline



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting data

import seaborn as sns # EDA

from IPython.display import display, Markdown as md

import gc
pd.set_option('display.max_rows', 500)

pd.set_option('display.max_colwidth', 500)

pd.set_option('display.unicode.ambiguous_as_wide', True)
seed_value= 30



# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value

import os

os.environ['PYTHONHASHSEED']=str(seed_value)



# 2. Set the `python` built-in pseudo-random generator at a fixed value

import random

random.seed(seed_value)



# 3. Set the `numpy` pseudo-random generator at a fixed value

import numpy as np

np.random.seed(seed_value)
PATH_IN = '../input/home-credit-default-risk'
train = pd.read_csv(f'{PATH_IN}/application_train.csv', low_memory=False)

test = pd.read_csv(f'{PATH_IN}/application_test.csv', low_memory=False)
train.head(10).transpose()
sns.countplot(train['TARGET'], palette="bwr")

plt.show()
categorical_feat = [

    'NAME_CONTRACT_TYPE', 'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',

    'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE','NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',

    'FLAG_MOBIL', 'FLAG_EMP_PHONE', 'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE', 'FLAG_EMAIL',

    'OCCUPATION_TYPE', 'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',

    'WEEKDAY_APPR_PROCESS_START', 'HOUR_APPR_PROCESS_START', 

    'REG_REGION_NOT_LIVE_REGION', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 

    'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY', 'ORGANIZATION_TYPE',

    'FONDKAPREMONT_MODE', 'HOUSETYPE_MODE', 'WALLSMATERIAL_MODE', 'EMERGENCYSTATE_MODE',

    'FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_3',

    'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6',

    'FLAG_DOCUMENT_7', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9',

    'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12',

    'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15',

    'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_18',

    'FLAG_DOCUMENT_19', 'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21',

]
# for feat in categorical_feat:

#     fig_dims = (30, 5)

#     fig, ax = plt.subplots(1, 2, figsize=fig_dims)

#     train_feat_values = train[feat].unique()

#     sns.countplot(train[feat], order=train_feat_values, ax=ax[0]).set_title(f'{feat}_TRAIN')

#     sns.countplot(test[feat], order=train_feat_values, ax=ax[1]).set_title(f'{feat}_TEST')

#     plt.show()
numerical_feat = [

    'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',

    'AMT_GOODS_PRICE', 'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH',

    'DAYS_EMPLOYED', 'DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'OWN_CAR_AGE',

    'CNT_FAM_MEMBERS',

    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3', 

    'APARTMENTS_AVG', 'BASEMENTAREA_AVG',

    'YEARS_BEGINEXPLUATATION_AVG', 'YEARS_BUILD_AVG', 'COMMONAREA_AVG',

    'ELEVATORS_AVG', 'ENTRANCES_AVG', 'FLOORSMAX_AVG', 'FLOORSMIN_AVG',

    'LANDAREA_AVG', 'LIVINGAPARTMENTS_AVG', 'LIVINGAREA_AVG',

    'NONLIVINGAPARTMENTS_AVG', 'NONLIVINGAREA_AVG', 'APARTMENTS_MODE',

    'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE',

    'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE',

    'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE',

    'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE',

    'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI', 'YEARS_BEGINEXPLUATATION_MEDI',

    'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI',

    'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',

    'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI',

    'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'OBS_30_CNT_SOCIAL_CIRCLE',

    'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',

    'DEF_60_CNT_SOCIAL_CIRCLE', 'DAYS_LAST_PHONE_CHANGE',

    'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',

    'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',

    'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR'

]
# for feat in numerical_feat:

#     fig_dims = (30, 5)

#     fig, ax = plt.subplots(1, 2, figsize=fig_dims)

#     sns.kdeplot(train[feat], ax=ax[0], bw=1, shade=True).set_title(f'{feat}_TRAIN')

#     sns.kdeplot(test[feat], ax=ax[1], bw=1, shade=True).set_title(f'{feat}_TEST')

#     plt.show()
X_train = train.drop(columns=['TARGET'])

y = train['TARGET']

X_test = test.copy()

print(X_train.shape)

print(X_test.shape)
train = None

test = None

gc.collect()
# Function to calculate missing values by column# Funct 

def missing_values_table(df):

        # Total missing values

        mis_val = df.isnull().sum()

        

        # Percentage of missing values

        mis_val_percent = 100 * df.isnull().sum() / len(df)

        

        # Make a table with the results

        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

        

        # Rename the columns

        mis_val_table_ren_columns = mis_val_table.rename(

        columns = {0 : 'Missing Values', 1 : '% of Total Values'})

        

        # Sort the table by percentage of missing descending

        mis_val_table_ren_columns = mis_val_table_ren_columns[

            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(

        '% of Total Values', ascending=False).round(1)

        

        # Print some summary information

        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      

            "There are " + str(mis_val_table_ren_columns.shape[0]) +

              " columns that have missing values.")

        

        # Return the dataframe with missing information

        return mis_val_table_ren_columns
missing_values_table(X_train)
def to_nan(df, value, columns):

    for column in columns:

        df.loc[df[column]==value, column] = np.NaN

    return df
to_nan(X_train, 'XNA', columns=['CODE_GENDER', 'ORGANIZATION_TYPE'])

to_nan(X_test, 'XNA', columns=['CODE_GENDER', 'ORGANIZATION_TYPE'])

gc.collect()
# DAYS_EMPLOYED

to_nan(X_train, 365243, columns=['DAYS_EMPLOYED'])

to_nan(X_test, 365243, columns=['DAYS_EMPLOYED'])

X_train['DAYS_EMPLOYED'].hist()

plt.show()

gc.collect()
def flag_nan(df, df_test=None):

    if df_test is None:

        for column in df.columns:

            if df[column].isna().values.any():

                df[f'{column}_NaN'] = df[column].isna().astype(int)

        return df

    

    for column in df.columns:

        if df[column].isna().values.any() or df_test[column].isna().values.any():

            df[f'{column}_NaN'] = df[column].isna().astype(int)

            df_test[f'{column}_NaN'] = df_test[column].isna().astype(int)

            

            
flag_nan(X_train, X_test)

X_train.columns
from sklearn.impute import SimpleImputer



# Filling NaN for Categorical features:



cate_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=X_train[categorical_feat].mode())

X_train[categorical_feat] = cate_imp.fit_transform(X_train[categorical_feat])

X_test[categorical_feat] = cate_imp.transform(X_test[categorical_feat])



# Filling NaN for Numerical features:



num_imp = SimpleImputer(missing_values=np.nan, strategy='median')

X_train[numerical_feat] = num_imp.fit_transform(X_train[numerical_feat])

X_test[numerical_feat] = num_imp.transform(X_test[numerical_feat])



gc.collect()
def cyclic_encode(df, feature_name, max_val, mapping=None, df_test=None):

    if mapping is not None:

        df[feature_name] = df[feature_name].apply(mapping)

        if df_test is not None:

            df_test[feature_name] = df_test[feature_name].apply(mapping)

            

    df[f'{feature_name}_Sin'] = np.sin(2 * np.pi * df[feature_name] / max_val)

    df[f'{feature_name}_Cos'] = np.cos(2 * np.pi * df[feature_name] / max_val)

    df.drop(columns=feature_name, inplace=True)

    

    if df_test is not None:

        df_test[f'{feature_name}_Sin'] = np.sin(2 * np.pi * df_test[feature_name] / max_val)

        df_test[f'{feature_name}_Cos'] = np.cos(2 * np.pi * df_test[feature_name] / max_val)

        df_test.drop(columns=feature_name, inplace=True)

        return df, df_test

    

    return df
# WEEKDAY_APPR_PROCESS_START

weekday_encoder = {    

    'MONDAY': 0,

    'TUESDAY': 1,

    'WEDNESDAY': 2,

    'THURSDAY': 3,

    'FRIDAY': 4,

    'SATURDAY': 5,

    'SUNDAY': 6

}



X_train['FLAG_WEEKEND_APPR_PROCESS_START'] = X_train['WEEKDAY_APPR_PROCESS_START'].apply(lambda x: x in ('SATURDAY', 'SUNDAY')).astype(int)

X_test['FLAG_WEEKEND_APPR_PROCESS_START'] = X_test['WEEKDAY_APPR_PROCESS_START'].apply(lambda x: x in ('SATURDAY', 'SUNDAY')).astype(int)

cyclic_encode(df=X_train, df_test=X_test, feature_name='WEEKDAY_APPR_PROCESS_START', max_val=6.0, mapping=lambda x: weekday_encoder[x])



categorical_feat.remove('WEEKDAY_APPR_PROCESS_START')



# HOUR_APPR_PROCESS_START



cyclic_encode(df=X_train, df_test=X_test, feature_name='HOUR_APPR_PROCESS_START', max_val=23.0)



categorical_feat.remove('HOUR_APPR_PROCESS_START')



gc.collect()
# Apply dummy encoding on Categorical features:

X_all = pd.concat([X_train, X_test])

X_all = pd.get_dummies(data=X_all, columns=categorical_feat, drop_first=True)

X_all.head(10).transpose()
X_train = X_all[:X_train.shape[0]]

X_test = X_all[X_train.shape[0]:]

print(X_train.shape)

print(X_test.shape)

X_all = None

gc.collect()
# from sklearn.preprocessing import MinMaxScaler



# scaler = MinMaxScaler(feature_range = (0, 1))

# X_train = scaler.fit_transform(X_train)

# X_test = scaler.transform(X_test)
from sklearn.model_selection import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(X_train.iloc[:,1:], y, test_size=0.15, stratify=y, random_state=seed_value)

print('Train shape: ', x_train.shape)

print('Valid shape: ', x_valid.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score



rfr = RandomForestClassifier(

    n_estimators=20,

    max_features=0.5,

    max_samples=0.5,

    min_samples_leaf=125,

    n_jobs=-1,

    oob_score=True,

    random_state=seed_value

)



%time rfr.fit(x_train, y_train)





print(f'ROC_AUC_Train: {roc_auc_score(y_train, rfr.predict_proba(x_train)[:, 1])}')

print(f'ROC_AUC_OOB: {roc_auc_score(y_train, rfr.oob_decision_function_[:, 1])}')

print(f'ROC_AUC_Valid: {roc_auc_score(y_valid, rfr.predict_proba(x_valid)[:, 1])}')
feature_importances = pd.DataFrame(

    {'feature': X_train.columns[1:], 'importance': rfr.feature_importances_}

).sort_values(by='importance', ascending=False)



feature_importances.head(15)
rfr = None

x_train, x_valid, y_train, y_valid = None, None, None, None

gc.collect()
X_train['CREDIT_INCOME_PERCENT'] = X_train['AMT_CREDIT'] / (X_train['AMT_INCOME_TOTAL']+1)

X_train['ANNUITY_INCOME_PERCENT'] = X_train['AMT_ANNUITY'] / (X_train['AMT_INCOME_TOTAL']+1)

X_train['CREDIT_TERM'] = X_train['AMT_ANNUITY'] / (X_train['AMT_CREDIT']+1)

X_train['DAYS_EMPLOYED_PERCENT'] = X_train['DAYS_EMPLOYED'] / (X_train['DAYS_BIRTH']+1)

X_train['GOODS_INCOME_PERCENT'] = X_train['AMT_GOODS_PRICE']/ (X_train['AMT_INCOME_TOTAL']+1)

X_train['GOODS_CREDIT_PERCENT'] = X_train['AMT_GOODS_PRICE']/ (X_train['AMT_CREDIT']+1)
X_test['CREDIT_INCOME_PERCENT'] = X_test['AMT_CREDIT'] / (X_test['AMT_INCOME_TOTAL']+1)

X_test['ANNUITY_INCOME_PERCENT'] = X_test['AMT_ANNUITY'] / (X_test['AMT_INCOME_TOTAL']+1)

X_test['CREDIT_TERM'] = X_test['AMT_ANNUITY'] / (X_test['AMT_CREDIT']+1)

X_test['DAYS_EMPLOYED_PERCENT'] = X_test['DAYS_EMPLOYED'] / (X_test['DAYS_BIRTH']+1)

X_test['GOODS_INCOME_PERCENT'] = X_test['AMT_GOODS_PRICE']/ (X_test['AMT_INCOME_TOTAL']+1)

X_test['GOODS_CREDIT_PERCENT'] = X_test['AMT_GOODS_PRICE']/ (X_test['AMT_CREDIT']+1)
previous_application = pd.read_csv(f'{PATH_IN}/previous_application.csv', low_memory=False)

previous_application.head(10).transpose()
cyclic_encode(previous_application, 'WEEKDAY_APPR_PROCESS_START', max_val=6.0, mapping=lambda x: weekday_encoder[x])

cyclic_encode(previous_application, 'HOUR_APPR_PROCESS_START', max_val=23.0).head()
categorical = [

    'NAME_CONTRACT_TYPE',

    'FLAG_LAST_APPL_PER_CONTRACT', 'NFLAG_LAST_APPL_IN_DAY',

    'NAME_CASH_LOAN_PURPOSE','NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE',

    'CODE_REJECT_REASON', 'NAME_TYPE_SUITE', 'NAME_CLIENT_TYPE',

    'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO', 'NAME_PRODUCT_TYPE',

    'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY',

    'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',

    'NFLAG_INSURED_ON_APPROVAL']



numerical = [feat for feat in previous_application.columns[2:] if feat not in categorical]

print(numerical)
to_nan(previous_application, 'XNA', categorical)

to_nan(previous_application, 365243, numerical)

flag_nan(previous_application)

gc.collect()
from sklearn.impute import SimpleImputer



# Filling NaN for Categorical features:



cate_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=previous_application[categorical].mode())

previous_application[categorical] = cate_imp.fit_transform(previous_application[categorical])



# Filling NaN for Numerical features:



num_imp = SimpleImputer(missing_values=np.nan, strategy='mean')

previous_application[numerical] = num_imp.fit_transform(previous_application[numerical])



cate_imp, num_imp = None, None

gc.collect()
# Onehot

previous_application = pd.get_dummies(data=previous_application, columns=categorical, drop_first=True)

categorical = [feat for feat in previous_application.columns[2:] if feat not in numerical]
# Groupby

PA_count = previous_application[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()

PA_count.columns = ['CNT_PREV']



PA_num = previous_application[['SK_ID_CURR']+numerical].groupby('SK_ID_CURR').mean()

PA_num.columns = ['mean_pa_' + col for col in PA_num.columns]



PA_cat = previous_application[['SK_ID_CURR']+categorical].groupby('SK_ID_CURR').sum()

PA_cat.columns = ['sum_pa_' + col for col in PA_cat.columns]



previous_application = PA_count.merge(right=PA_num, left_index=True, right_index=True).merge(right=PA_cat, left_index=True, right_index=True)

previous_application.shape
X_train = X_train.merge(right=previous_application, how='left', left_on='SK_ID_CURR', right_index=True)

X_test = X_test.merge(right=previous_application, how='left', left_on='SK_ID_CURR', right_index=True)

print(X_train.shape)

print(X_test.shape)



PA_count, PA_num, PA_cat, previous_application = None, None, None, None

gc.collect()
bureau = pd.read_csv(f'{PATH_IN}/bureau.csv', low_memory=False)

bureau_balance = pd.read_csv(f'{PATH_IN}/bureau_balance.csv', low_memory=False)
bureau = bureau.merge(right=bureau_balance[['SK_ID_BUREAU', 'MONTHS_BALANCE']].groupby('SK_ID_BUREAU').count(), left_on='SK_ID_BUREAU', right_index=True)

bureau.head(10).transpose()
categorical = ['CREDIT_ACTIVE', 'CREDIT_CURRENCY','CREDIT_TYPE']

numerical = [feat for feat in bureau.columns[2:] if feat not in  categorical]



flag_nan(bureau)
from sklearn.impute import SimpleImputer



# Filling NaN for Categorical features:



cate_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=bureau[categorical].mode())

bureau[categorical] = cate_imp.fit_transform(bureau[categorical])



# Filling NaN for Numerical features:



num_imp = SimpleImputer(missing_values=np.nan, strategy='mean')

bureau[numerical] = num_imp.fit_transform(bureau[numerical])



cate_imp, num_imp = None, None

gc.collect()
# Onehot

bureau = pd.get_dummies(data=bureau, columns=categorical, drop_first=True)

categorical = [feat for feat in bureau.columns[2:] if feat not in numerical]
bureau.head()
# Groupby

bureau_count = bureau[['SK_ID_CURR', 'SK_ID_BUREAU']].groupby('SK_ID_CURR').count()

bureau_count.columns = ['CNT_BUREAU']



bureau_num = bureau[['SK_ID_CURR']+numerical].groupby('SK_ID_CURR').mean()

bureau_num.columns = ['mean_br_' + col for col in bureau_num.columns]



bureau_cat = bureau[['SK_ID_CURR']+categorical].groupby('SK_ID_CURR').sum()

bureau_cat.columns = ['sum_br_' + col for col in bureau_cat.columns]



bureau = bureau_count.merge(right=bureau_num, left_index=True, right_index=True).merge(right=bureau_cat, left_index=True, right_index=True)

bureau.shape
X_train = X_train.merge(right=bureau, how='left', left_on='SK_ID_CURR', right_index=True)

X_test = X_test.merge(right=bureau, how='left', left_on='SK_ID_CURR', right_index=True)

print(X_train.shape)

print(X_test.shape)



bureau_count, bureau_num, bureau_cat, bureau = None, None, None, None

gc.collect()
installments_payments = pd.read_csv(f'{PATH_IN}/installments_payments.csv', low_memory=False)

installments_payments.head(10).transpose()
# Count of previous installments

cnt_inst = installments_payments[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()

installments_payments['SK_ID_PREV'] = installments_payments['SK_ID_CURR'].map(cnt_inst['SK_ID_PREV'])



# Average values for all other variables in installments payments

avg_inst = installments_payments.groupby('SK_ID_CURR').mean()

avg_inst.columns = ['mean_i_' + f_ for f_ in avg_inst.columns]



X_train = X_train.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')

X_test = X_test.merge(right=avg_inst.reset_index(), how='left', on='SK_ID_CURR')



cnt_inst, avg_inst, installments_payments = None, None, None

gc.collect()
pcb = pd.read_csv(f'{PATH_IN}/POS_CASH_balance.csv', low_memory=False)

pcb.head(10).transpose()
# Count of pos cash for a given ID

pcb_count = pcb[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()

pcb['SK_ID_PREV'] = pcb['SK_ID_CURR'].map(pcb_count['SK_ID_PREV'])



# Average values for all other variables in pos cash

pcb_avg = pcb.groupby('SK_ID_CURR').mean()



X_train = X_train.merge(right=pcb_avg.reset_index(), how='left', on='SK_ID_CURR')

X_test = X_test.merge(right=pcb_avg.reset_index(), how='left', on='SK_ID_CURR')



pcb_count, pcv_avg, pcb = None, None, None

gc.collect()
credit_card_balance = pd.read_csv(f'{PATH_IN}/credit_card_balance.csv', low_memory=False)

credit_card_balance.head(10).transpose()
# Count of previous credit card balance for a given ID

nb_prevs = credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()

credit_card_balance['SK_ID_PREV'] = credit_card_balance['SK_ID_CURR'].map(nb_prevs['SK_ID_PREV'])



# Average values of all other columns

avg_cc_bal = credit_card_balance.groupby('SK_ID_CURR').mean()

avg_cc_bal.columns = ['cc_bal_' + f_ for f_ in avg_cc_bal.columns]



X_train = X_train.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')

X_test = X_test.merge(right=avg_cc_bal.reset_index(), how='left', on='SK_ID_CURR')



nb_prevs, avg_cc_bal, creadit_card_balance = None, None, None

gc.collect()
from sklearn.impute import SimpleImputer



all_imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)

X_train[:] = all_imp.fit_transform(X_train[:])

X_test[:] = all_imp.transform(X_test[:])



all_imp = None

gc.collect()
from sklearn.model_selection import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(X_train.iloc[:, 1:], y, test_size=0.15, stratify=y, random_state=seed_value)

print('Train shape: ', x_train.shape)

print('Valid shape: ', x_valid.shape)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_auc_score



rfr = RandomForestClassifier(

    n_estimators=20,

    max_features=0.5,

    max_samples=0.5,

    min_samples_leaf=125,

    n_jobs=-1,

    oob_score=True,

    random_state=seed_value

)



%time rfr.fit(x_train, y_train)





print(f'ROC_AUC_Train: {roc_auc_score(y_train, rfr.predict_proba(x_train)[:, 1])}')

print(f'ROC_AUC_Valid: {roc_auc_score(y_valid, rfr.predict_proba(x_valid)[:, 1])}')

print(f'ROC_AUC_OOB: {roc_auc_score(y_train, rfr.oob_decision_function_[:, 1])}')
feature_importances = pd.DataFrame(

    {'feature': X_train.columns[1:], 'importance': rfr.feature_importances_}

).sort_values(by='importance', ascending=False)

feature_importances
drop_features = feature_importances.loc[feature_importances['importance'] < 0.000001, 'feature']

drop_features.count()
X_train.drop(columns=drop_features, inplace=True)

X_test.drop(columns=drop_features, inplace=True)

print(X_train.shape)

print(X_test.shape)
rfr = RandomForestClassifier(

    n_estimators=150,

    max_features=0.5,

    max_samples=0.5,

    min_samples_leaf=125,

    n_jobs=-1,

    oob_score=True,

    random_state=seed_value

)



%time rfr.fit(X_train.iloc[:, 1:], y)



print(f'ROC_AUC_Train: {roc_auc_score(y, rfr.predict_proba(X_train.iloc[:, 1:])[:, 1])}')

print(f'ROC_AUC_OOB: {roc_auc_score(y, rfr.oob_decision_function_[:, 1])}')
feature_importances = pd.DataFrame(

    {'feature': X_train.columns[1:], 'importance': rfr.feature_importances_}

).sort_values(by='importance', ascending=False)

feature_importances.head(15)
y_predict = rfr.predict_proba(X_test.iloc[:, 1:])[:, 1]

random_forest_with_feature_engineering_submission = pd.DataFrame({'SK_ID_CURR': X_test['SK_ID_CURR'].astype(int), 'TARGET': y_predict})

random_forest_with_feature_engineering_submission.head()
random_forest_with_feature_engineering_submission.to_csv('random_forest_with_feature_engineering.csv', index=False)
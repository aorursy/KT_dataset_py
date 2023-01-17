import numpy as np

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')

from sklearn.preprocessing import Imputer

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
def show_kde_and_bins(df, col):
    global fig_num
    plt.figure(fig_num)
    sns.kdeplot(df.loc[df['TARGET'] == 0, col], label = 'target == 0')
    sns.kdeplot(df.loc[df['TARGET'] == 1, col], label = 'target == 1')
    plt.xlabel(col)
    plt.ylabel('Density')
    
    plt.figure(fig_num + 1)
    axes = plt.gca()
    axes.set_xlim([df[col].min(), df[col].max()])
    staging = df[[col,'TARGET']]
    staging['COL_BINNED'] = pd.cut(staging[col], bins = 10)
    final = staging.groupby('COL_BINNED').mean()
    plt.bar(final[col], final['TARGET'])
    plt.xlabel(col)
    plt.ylabel('AVG TARGET')
    
    fig_num += 2
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')

print(train.shape)

# columns with outliers:
# DAYS_EMPLOYED

outlier = train[train['DAYS_EMPLOYED'] > 30000]
non_outlier = train[train['DAYS_EMPLOYED'] < 30000]

# new column for outlier
train['DAYS_EMPLOYED_OUTLIER'] = 0
train.loc[train['DAYS_EMPLOYED'] > 30000, 'DAYS_EMPLOYED_OUTLIER'] = 1

# fill in outliers with mean
train.loc[train['DAYS_EMPLOYED'] > 30000, 'DAYS_EMPLOYED'] = non_outlier['DAYS_EMPLOYED'].mean()

# clean up DAYS_BIRTH and DAYS_EMPLOYED to be easy to understand
train['YEARS_BIRTH'] = train['DAYS_BIRTH'] / -365
train['YEARS_EMPLOYED'] = train['DAYS_EMPLOYED'] / -365

train = pd.get_dummies(train)

# columns with outliers:
# DAYS_EMPLOYED

outlier_te = test[test['DAYS_EMPLOYED'] > 30000]
non_outlier_te = test[test['DAYS_EMPLOYED'] < 30000]

# new column for outlier
test['DAYS_EMPLOYED_OUTLIER'] = 0
test.loc[test['DAYS_EMPLOYED'] > 30000, 'DAYS_EMPLOYED_OUTLIER'] = 1

# fill in outliers with mean
test.loc[test['DAYS_EMPLOYED'] > 30000, 'DAYS_EMPLOYED'] = non_outlier_te['DAYS_EMPLOYED'].mean()

# clean up DAYS_BIRTH and DAYS_EMPLOYED to be easy to understand
test['YEARS_BIRTH'] = test['DAYS_BIRTH'] / -365
test['YEARS_EMPLOYED'] = test['DAYS_EMPLOYED'] / -365

test = pd.get_dummies(test)

target = train['TARGET']

train, test = train.align(test, join = 'inner', axis = 1)

train['TARGET'] = target

#corr = train.corr()
#print(corr['TARGET'].sort_values(ascending = False))

# strong positive corr -> DAYS_BIRTH, DAYS_EMPLOYED,REGION_RATING_CLIENT_W_CITY 
# strong negative corr -> EXT_SOURCE 1,2,3
# Let's explore!

# KDE / pd.cut()

fig_num = 1

    
show_kde_and_bins(train, 'EXT_SOURCE_1')
show_kde_and_bins(train, 'EXT_SOURCE_2')
show_kde_and_bins(train, 'EXT_SOURCE_3')
show_kde_and_bins(train, 'REGION_RATING_CLIENT_W_CITY')
show_kde_and_bins(train, 'YEARS_BIRTH')
show_kde_and_bins(train, 'YEARS_EMPLOYED')
bureau = pd.read_csv('../input/bureau.csv')
bureau = pd.get_dummies(bureau)

# odd columns
# amt_credit_max_overdue
# AMT_CREDIT_SUM_LIMIT
# AMT_CREDIT_SUM_OVERDUE

def outliers(df, col):
    mean = df[col].mean()
    std_dev = df[col].std()
    df['Z_SCORE'] = (df[col] - mean) / std_dev
    print(df[col].describe())
    print('mean: ', mean,'. std dev: ', std_dev)
    print(df.loc[df['Z_SCORE'] > 5, [col, 'Z_SCORE']])

#outliers(bureau, 'AMT_CREDIT_MAX_OVERDUE')
#outliers(bureau, 'AMT_CREDIT_SUM_LIMIT')
#outliers(bureau, 'AMT_CREDIT_SUM_OVERDUE')
# .. definitely outliers but leaving there since it seems legit (not one value or anything mysterious)

# metrics on bureau
bureau_staging = bureau \
    .drop(columns = ['SK_ID_BUREAU']) \
    .groupby('SK_ID_CURR') \
    .agg(['count','mean','min','max', 'sum']) \
    .reset_index()

columns = ['SK_ID_CURR']

# Convert multi-level index from .agg() into clean columns
# borrowing from: https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
for var in bureau_staging.columns.levels[0]:
    if var != 'SK_ID_CURR':
        for stat in bureau_staging.columns.levels[1][:-1]:
            columns.append('bureau_%s_%s' % (var, stat))

bureau_staging.columns = columns

train = pd.merge(
    train,
    bureau_staging,
    how = 'left',
    on = 'SK_ID_CURR'
)

test = pd.merge(
    test,
    bureau_staging,
    how = 'left',
    on = 'SK_ID_CURR'
)

# columns.append('TARGET')

# bureau_corr = tr[columns]
#tr = imputer.fit_transform(tr)
#print(tr)

# corr = bureau_corr.corr()
# print(corr['TARGET'].sort_values(ascending = False))

# strong-ish corrs:
# bureau_DAYS_CREDIT_mean  
# bureau_CREDIT_ACTIVE_Active_mean
# bureau_DAYS_CREDIT_min

show_kde_and_bins(train, 'bureau_DAYS_CREDIT_mean')
show_kde_and_bins(train, 'bureau_CREDIT_ACTIVE_Active_mean')
show_kde_and_bins(train, 'bureau_DAYS_CREDIT_min')
previous_application = pd.read_csv('../input/previous_application.csv')

previous_application = previous_application[
    [
        'SK_ID_CURR',
        'CODE_REJECT_REASON',                       
        'NAME_CONTRACT_STATUS',                      
        'NAME_PRODUCT_TYPE'
    ]
]
previous_application = pd.get_dummies(previous_application)

previous_application_staging = previous_application \
    .groupby('SK_ID_CURR') \
    .agg(['count','mean','min','max', 'sum']) \
    .reset_index()

columns = ['SK_ID_CURR']

# Convert multi-level index from .agg() into clean columns
# borrowing from: https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
for var in previous_application_staging.columns.levels[0]:
    if var != 'SK_ID_CURR':
        for stat in previous_application_staging.columns.levels[1][:-1]:
            columns.append('previous_application_%s_%s' % (var, stat))

previous_application_staging.columns = columns

previous_application = previous_application_staging[
    [
        'SK_ID_CURR',
        'previous_application_CODE_REJECT_REASON_XAP_mean',                       
        'previous_application_NAME_CONTRACT_STATUS_Approved_mean',                 
        'previous_application_NAME_CONTRACT_STATUS_Refused_mean',                      
        'previous_application_NAME_CONTRACT_STATUS_Refused_sum',                     
        'previous_application_CODE_REJECT_REASON_SCOFR_max',                        
        'previous_application_NAME_PRODUCT_TYPE_walk-in_sum'
    ]
]

train = pd.merge(
    train,
    previous_application_staging,
    how = 'left',
    on = 'SK_ID_CURR'
)

test = pd.merge(
    test,
    previous_application_staging,
    how = 'left',
    on = 'SK_ID_CURR'
)
POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
POS_CASH_balance = POS_CASH_balance[['SK_ID_CURR', 'MONTHS_BALANCE']]

POS_CASH_balance_staging = POS_CASH_balance \
    .groupby('SK_ID_CURR') \
    .agg(['count','mean','min','max', 'sum']) \
    .reset_index()

columns = ['SK_ID_CURR']

# Convert multi-level index from .agg() into clean columns
# borrowing from: https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
for var in POS_CASH_balance_staging.columns.levels[0]:
    if var != 'SK_ID_CURR':
        for stat in POS_CASH_balance_staging.columns.levels[1][:-1]:
            columns.append('POS_CASH_balance_%s_%s' % (var, stat))

POS_CASH_balance_staging.columns = columns

train = pd.merge(
    train,
    POS_CASH_balance_staging,
    how = 'left',
    on = 'SK_ID_CURR'
)

test = pd.merge(
    test,
    POS_CASH_balance_staging,
    how = 'left',
    on = 'SK_ID_CURR'
)
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')

credit_card_balance_staging = credit_card_balance \
    .groupby('SK_ID_CURR') \
    .agg(['count','mean','min','max', 'sum']) \
    .reset_index()

columns = ['SK_ID_CURR']

# Convert multi-level index from .agg() into clean columns
# borrowing from: https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
for var in credit_card_balance_staging.columns.levels[0]:
    if var != 'SK_ID_CURR':
        for stat in credit_card_balance_staging.columns.levels[1][:-1]:
            columns.append('credit_card_balance_%s_%s' % (var, stat))

credit_card_balance_staging.columns = columns

train = pd.merge(
    train,
    credit_card_balance_staging,
    how = 'left',
    on = 'SK_ID_CURR'
)

test = pd.merge(
    test,
    credit_card_balance_staging,
    how = 'left',
    on = 'SK_ID_CURR'
)
y = train['TARGET']
imputer = Imputer(strategy = 'mean')
x_cols_app = [
    'EXT_SOURCE_1', 
    'EXT_SOURCE_2', 
    'EXT_SOURCE_3', 
    'REGION_RATING_CLIENT_W_CITY', 
    'YEARS_BIRTH',
    'YEARS_EMPLOYED'
]

x_app = train[x_cols_app]
x_app = imputer.fit_transform(x_app)

x_app_train, x_app_test, y_app_train, y_app_test = train_test_split(x_app, y, test_size = 0.33, random_state = 0)

model_app = RandomForestClassifier()
model_app.fit(x_app_train, y_app_train)
predictions_app = model_app.predict_proba(x_app_test)

auc_app = roc_auc_score(y_app_test, predictions_app[:,1])
print('AUC: app only:', auc_app)
x_cols_app_bureau = [
    'EXT_SOURCE_1', 
    'EXT_SOURCE_2', 
    'EXT_SOURCE_3', 
    'REGION_RATING_CLIENT_W_CITY', 
    'YEARS_BIRTH',
    'YEARS_EMPLOYED',
    'bureau_DAYS_CREDIT_mean',
    'bureau_CREDIT_ACTIVE_Active_mean',
    'bureau_DAYS_CREDIT_min'
]

x_app_bureau = train[x_cols_app_bureau]
x_app_bureau = imputer.fit_transform(x_app_bureau)

x_app_bureau_train, x_app_bureau_test, y_app_bureau_train, y_app_bureau_test = train_test_split(x_app_bureau, y, test_size = 0.33, random_state = 0)

model_app_bureau = RandomForestClassifier()
model_app_bureau.fit(x_app_bureau_train, y_app_bureau_train)
predictions_app_bureau = model_app_bureau.predict_proba(x_app_bureau_test)

auc_app_bureau = roc_auc_score(y_app_bureau_test, predictions_app_bureau[:,1])
print('AUC: app + bureau:', auc_app_bureau)
x_cols_app_bureau_prev = [
    'EXT_SOURCE_1', 
    'EXT_SOURCE_2', 
    'EXT_SOURCE_3', 
    'REGION_RATING_CLIENT_W_CITY', 
    'YEARS_BIRTH',
    'YEARS_EMPLOYED',
    'bureau_DAYS_CREDIT_mean',
    'bureau_CREDIT_ACTIVE_Active_mean',
    'bureau_DAYS_CREDIT_min',
    'previous_application_CODE_REJECT_REASON_XAP_mean',                       
    'previous_application_NAME_CONTRACT_STATUS_Approved_mean',                 
    'previous_application_NAME_CONTRACT_STATUS_Refused_mean',                      
    'previous_application_NAME_CONTRACT_STATUS_Refused_sum',                     
    'previous_application_CODE_REJECT_REASON_SCOFR_max',                        
    'previous_application_NAME_PRODUCT_TYPE_walk-in_sum' 
]

x_app_bureau_prev = train[x_cols_app_bureau_prev]
x_app_bureau_prev = imputer.fit_transform(x_app_bureau_prev)

x_app_bureau_prev_train, x_app_bureau_prev_test, y_app_bureau_prev_train, y_app_bureau_prev_test = train_test_split(x_app_bureau_prev, y, test_size = 0.33, random_state = 0)

model_app_bureau_prev = RandomForestClassifier()
model_app_bureau_prev.fit(x_app_bureau_prev_train, y_app_bureau_prev_train)
predictions_app_bureau_prev = model_app_bureau_prev.predict_proba(x_app_bureau_prev_test)

auc_app_bureau_prev = roc_auc_score(y_app_bureau_prev_test, predictions_app_bureau_prev[:,1])
print('AUC: application_train.csv + bureau.csv + previous_application.csv:', auc_app_bureau_prev)
#previous_application_CODE_REJECT_REASON_XAP_mean                           -0.073930
#previous_application_NAME_CONTRACT_STATUS_Approved_mean                    -0.063521
#previous_application_NAME_CONTRACT_STATUS_Refused_mean                      0.077671
#previous_application_NAME_CONTRACT_STATUS_Refused_sum                       0.064469
#previous_application_CODE_REJECT_REASON_SCOFR_max                           0.063657
#previous_application_NAME_PRODUCT_TYPE_walk-in_sum                          0.062628
x_cols_app_bureau_prev_pos = [
    'EXT_SOURCE_1', 
    'EXT_SOURCE_2', 
    'EXT_SOURCE_3', 
    'REGION_RATING_CLIENT_W_CITY', 
    'YEARS_BIRTH',
    'YEARS_EMPLOYED',
    'bureau_DAYS_CREDIT_mean',
    'bureau_CREDIT_ACTIVE_Active_mean',
    'bureau_DAYS_CREDIT_min',
    'previous_application_CODE_REJECT_REASON_XAP_mean',                       
    'previous_application_NAME_CONTRACT_STATUS_Approved_mean',                 
    'previous_application_NAME_CONTRACT_STATUS_Refused_mean',                      
    'previous_application_NAME_CONTRACT_STATUS_Refused_sum',                     
    'previous_application_CODE_REJECT_REASON_SCOFR_max',                        
    'previous_application_NAME_PRODUCT_TYPE_walk-in_sum',
    'POS_CASH_balance_MONTHS_BALANCE_min'
]

x_app_bureau_prev_pos = train[x_cols_app_bureau_prev_pos]
x_app_bureau_prev_pos = imputer.fit_transform(x_app_bureau_prev_pos)

x_app_bureau_prev_pos_train, x_app_bureau_prev_pos_test, y_app_bureau_prev_pos_train, y_app_bureau_prev_pos_test = train_test_split(x_app_bureau_prev_pos, y, test_size = 0.33, random_state = 0)

model_app_bureau_prev_pos = RandomForestClassifier()
model_app_bureau_prev_pos.fit(x_app_bureau_prev_pos_train, y_app_bureau_prev_pos_train)
predictions_app_bureau_prev_pos = model_app_bureau_prev_pos.predict_proba(x_app_bureau_prev_pos_test)

auc_app_bureau_prev_pos = roc_auc_score(y_app_bureau_prev_pos_test, predictions_app_bureau_prev_pos[:,1])
print('AUC: application_train.csv + bureau.csv + previous_application.csv + POS_CASH_balance.csv:', auc_app_bureau_prev_pos)
#previous_application_CODE_REJECT_REASON_XAP_mean                           -0.073930
#previous_application_NAME_CONTRACT_STATUS_Approved_mean                    -0.063521
#previous_application_NAME_CONTRACT_STATUS_Refused_mean                      0.077671
#previous_application_NAME_CONTRACT_STATUS_Refused_sum                       0.064469
#previous_application_CODE_REJECT_REASON_SCOFR_max                           0.063657
#previous_application_NAME_PRODUCT_TYPE_walk-in_sum                          0.062628
x_cols_app_bureau_prev_pos_credit = [
    'EXT_SOURCE_1', 
    'EXT_SOURCE_2', 
    'EXT_SOURCE_3', 
    'REGION_RATING_CLIENT_W_CITY', 
    'YEARS_BIRTH',
    'YEARS_EMPLOYED',
    'bureau_DAYS_CREDIT_mean',
    'bureau_CREDIT_ACTIVE_Active_mean',
    'bureau_DAYS_CREDIT_min',
    'previous_application_CODE_REJECT_REASON_XAP_mean',                       
    'previous_application_NAME_CONTRACT_STATUS_Approved_mean',                 
    'previous_application_NAME_CONTRACT_STATUS_Refused_mean',                      
    'previous_application_NAME_CONTRACT_STATUS_Refused_sum',                     
    'previous_application_CODE_REJECT_REASON_SCOFR_max',                        
    'previous_application_NAME_PRODUCT_TYPE_walk-in_sum',
    'POS_CASH_balance_MONTHS_BALANCE_min',
    'credit_card_balance_CNT_DRAWINGS_ATM_CURRENT_mean',
    'credit_card_balance_CNT_DRAWINGS_CURRENT_max',
    'credit_card_balance_AMT_BALANCE_mean',
    'credit_card_balance_AMT_TOTAL_RECEIVABLE_mean'
]

x_app_bureau_prev_pos_credit = train[x_cols_app_bureau_prev_pos_credit]
x_app_bureau_prev_pos_credit = imputer.fit_transform(x_app_bureau_prev_pos_credit)

x_app_bureau_prev_pos_credit_train, x_app_bureau_prev_pos_credit_test, y_app_bureau_prev_pos_credit_train, y_app_bureau_prev_pos_credit_test = train_test_split(x_app_bureau_prev_pos_credit, y, test_size = 0.33, random_state = 0)

model_app_bureau_prev_pos_credit = RandomForestClassifier()
model_app_bureau_prev_pos_credit.fit(x_app_bureau_prev_pos_credit_train, y_app_bureau_prev_pos_credit_train)
predictions_app_bureau_prev_pos_credit = model_app_bureau_prev_pos_credit.predict_proba(x_app_bureau_prev_pos_credit_test)

auc_app_bureau_prev_pos_credit = roc_auc_score(y_app_bureau_prev_pos_credit_test, predictions_app_bureau_prev_pos_credit[:,1])
print('AUC: application_train.csv + bureau.csv + previous_application.csv + POS_CASH_balance.csv + credit_card_balance.csv:', auc_app_bureau_prev_pos_credit)
import lightgbm as lgb

x_app_bureau_prev_pos_credit_train, x_app_bureau_prev_pos_credit_test, y_app_bureau_prev_pos_credit_train, y_app_bureau_prev_pos_credit_test = train_test_split(x_app_bureau_prev_pos_credit, y, test_size = 0.33, random_state = 0)

model = lgb.LGBMClassifier(
    n_estimators=10000, 
    objective = 'binary', 
    class_weight = 'balanced', 
    learning_rate = 0.05, 
    reg_alpha = 0.1, 
    reg_lambda = 0.1, 
    subsample = 0.8, 
    n_jobs = -1, 
    random_state = 50
)

model.fit(
    x_app_bureau_prev_pos_credit_train,
    y_app_bureau_prev_pos_credit_train,
    eval_metric = 'auc'
)

p = model.predict_proba(x_app_bureau_prev_pos_credit_test)

lightGBM_guinea = roc_auc_score(y_app_bureau_prev_pos_credit_test, p[:,1])
print('LGBM:', lightGBM_guinea)
test_submit = test[x_cols_app_bureau_prev_pos_credit]
test_submit = imputer.fit_transform(test_submit)

predictions_final = model.predict_proba(test_submit)
submit = pd.DataFrame({
    "SK_ID_CURR": test['SK_ID_CURR'],
    "TARGET": predictions_final[:,1]
})

submit.to_csv('submit.csv', index = False)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import gc 
plt.style.use('fivethirtyeight')
#sns.palplot(sns.color_palette())

%matplotlib inline 

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly as py
import plotly.graph_objs as go

init_notebook_mode(connected=True) #do not miss this line
from plotly import tools


# For model estimation
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, Imputer
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold, StratifiedKFold


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
PATH = "../input"
# Any results you write to the current directory are saved as output.
data = pd.read_csv(PATH+"/application_train.csv")
test = pd.read_csv(PATH+"/application_test.csv")
bureau = pd.read_csv(PATH+"/bureau.csv")
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv")
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv")
installments_payments = pd.read_csv(PATH+"/installments_payments.csv")
previous_application = pd.read_csv(PATH+"/previous_application.csv")
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv")
data = pd.read_csv(PATH+"/application_train.csv", nrows=10000)
test = pd.read_csv(PATH+"/application_test.csv", nrows=10000)
bureau = pd.read_csv(PATH+"/bureau.csv", nrows=10000)
bureau_balance = pd.read_csv(PATH+"/bureau_balance.csv", nrows=10000)
credit_card_balance = pd.read_csv(PATH+"/credit_card_balance.csv", nrows=10000)
installments_payments = pd.read_csv(PATH+"/installments_payments.csv", nrows=10000)
previous_application = pd.read_csv(PATH+"/previous_application.csv", nrows=10000)
POS_CASH_balance = pd.read_csv(PATH+"/POS_CASH_balance.csv", nrows=10000)
data.columns.values

## Feature Engineering training set
data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
data['CODE_GENDER'].replace({'XNA': 'F'}, inplace=True)
data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
data['YEARS_BUILD_CREDIT'] = data['AMT_CREDIT']/data['YEARS_BUILD_AVG']
data['Annuity_Income'] = data['AMT_ANNUITY']/data['AMT_INCOME_TOTAL']
data['Income_Cred'] = data['AMT_CREDIT']/data['AMT_INCOME_TOTAL']
data['EMP_AGE'] = data['DAYS_EMPLOYED']/data['DAYS_BIRTH']
data['Income_PP'] = data['AMT_INCOME_TOTAL']/data['CNT_FAM_MEMBERS']
data['CHILDREN_RATIO'] = (1 + data['CNT_CHILDREN']) / data['CNT_FAM_MEMBERS']
data['PAYMENTS'] = data['AMT_ANNUITY']/ data['AMT_CREDIT']
#data['Annuity_Credit'] = data['AMT_CREDIT']/data['AMT_ANNUITY']
data['NEW_CREDIT_TO_GOODS_RATIO'] = data['AMT_CREDIT'] / data['AMT_GOODS_PRICE']
data['GOODS_INCOME'] =  data['AMT_GOODS_PRICE']/data['AMT_INCOME_TOTAL']
# data['SOURCE_1_PERCENT'] = data['EXT_SOURCE_1']/(data['EXT_SOURCE_1']+data['EXT_SOURCE_2']+data['EXT_SOURCE_3'])
# data['SOURCE_2_PERCENT'] = data['EXT_SOURCE_2']/(data['EXT_SOURCE_1']+data['EXT_SOURCE_2']+data['EXT_SOURCE_3'])
# data['SOURCE_3_PERCENT'] = data['EXT_SOURCE_3']/(data['EXT_SOURCE_1']+data['EXT_SOURCE_2']+data['EXT_SOURCE_3'])
data['Ext_source_mult'] = data['EXT_SOURCE_1'] * data['EXT_SOURCE_2'] * data['EXT_SOURCE_3']
data['Ext_SOURCE_MEAN'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
data['Ext_SOURCE_SD'] = data[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis = 1)






columns = ['Annuity_Income', 'Income_Cred', 'EMP_AGE', 'Income_PP']
#df[columns].describe()

## Feature engineering test set
test['CODE_GENDER'].replace({'XNA': 'F'}, inplace=True)
test['YEARS_BUILD_CREDIT'] = test['AMT_CREDIT']/test['YEARS_BUILD_AVG']
test['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
test['Annuity_Income'] = test['AMT_ANNUITY']/test['AMT_INCOME_TOTAL']
test['Income_Cred'] = test['AMT_CREDIT']/test['AMT_INCOME_TOTAL']
test['EMP_AGE'] = test['DAYS_EMPLOYED']/test['DAYS_BIRTH']
test['Income_PP'] = test['AMT_INCOME_TOTAL']/test['CNT_FAM_MEMBERS']
test['CHILDREN_RATIO'] = (1 + test['CNT_CHILDREN']) / test['CNT_FAM_MEMBERS']
#test['Annuity_Credit'] =test['AMT_CREDIT']/ test['AMT_ANNUITY']
test['PAYMENTS'] = test['AMT_ANNUITY']/ test['AMT_CREDIT']
test['NEW_CREDIT_TO_GOODS_RATIO'] = test['AMT_CREDIT'] / test['AMT_GOODS_PRICE']
test['GOODS_INCOME'] =  test['AMT_GOODS_PRICE']/test['AMT_INCOME_TOTAL']
# test['SOURCE_1_PERCENT'] = test['EXT_SOURCE_1']/(test['EXT_SOURCE_1']+test['EXT_SOURCE_2']+test['EXT_SOURCE_3'])
# test['SOURCE_2_PERCENT'] = test['EXT_SOURCE_2']/(test['EXT_SOURCE_1']+test['EXT_SOURCE_2']+test['EXT_SOURCE_3'])
# test['SOURCE_3_PERCENT'] = test['EXT_SOURCE_3']/(test['EXT_SOURCE_1']+test['EXT_SOURCE_2']+test['EXT_SOURCE_3'])
test['Ext_source_mult'] = test['EXT_SOURCE_1'] * test['EXT_SOURCE_2'] * test['EXT_SOURCE_3']
test['Ext_SOURCE_MEAN'] = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis = 1)
test['Ext_SOURCE_SD'] = test[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis = 1)

# def plot_target(data, feature, xlab= '', ylab= '', title= ""):
#     plt.figure(figsize=(12,9))
#     sns.kdeplot(data.loc[data['TARGET'] == 0, feature], label = 'target == 0')

#     # KDE plot of loans which were not repaid on time
#     sns.kdeplot(data.loc[data['TARGET'] == 1, feature], label = 'target == 1')
    
#     # Labeling of plot
#     plt.xlabel(feature); plt.ylabel('Density'); plt.title("Distribution of %s"%(feature));
# plot_target(data, 'Annuity_Income')
# plot_target(data,'Income_Cred')
# plot_target(data, 'EMP_AGE')
# plot_target(data,'Income_PP')
# df1 = bureau
# def new_features(df1, group_by ,stats, data_name):
#     columns = [group_by]
#     data_features = df1.groupby(group_by).agg(stats).reset_index()
#     for var in data_features.columns.levels[0]:
#         # ignore grouping variable
#         if var != group_by:
#             # get rid of original variable
#             for stat in data_features.columns.levels[1][:-1]:
#                 columns.append('%s_%s_%s' %(data_name,var,stat))
#     data_features.columns = columns            
#     data = bureau.merge(data_features , on='SK_ID_CURR', how = 'left')
#     return data
# bureau_new = new_features(bureau.drop(columns = ['SK_ID_BUREAU']),'SK_ID_CURR', stats ,data_name = 'bureau')
# bureau_new.head()
# COUNT
bureau_new = bureau
group = bureau_new[['SK_ID_CURR', 'DAYS_CREDIT']].groupby('SK_ID_CURR')['DAYS_CREDIT'].count().reset_index().rename(index=str, columns={'DAYS_CREDIT': 'BUREAU_LOAN_COUNT'})
bureau_new = bureau_new.merge(group, how = 'left', on = 'SK_ID_CURR')
bureau_new.head()
del group
group = bureau_new[['SK_ID_CURR', 'CREDIT_TYPE']].groupby('SK_ID_CURR')['CREDIT_TYPE'].nunique().reset_index().rename(index=str, columns = {'CREDIT_TYPE': 'LOAN_TYPES_PER_CUST'})
bureau_new = bureau_new.merge(group,on = ['SK_ID_CURR'], how = 'left')
bureau_new.head()
del group
bureau_new["AVERAGE_LOAN_TYPE"] = bureau_new['BUREAU_LOAN_COUNT']/bureau_new['LOAN_TYPES_PER_CUST']
replace = {'Active': 1, 'Closed':0, 'Sold': 1, 'Bad debt': 1}
bureau_new['CREDIT_ACTIVE'] = bureau_new['CREDIT_ACTIVE'].replace(replace)
gp = bureau_new.groupby('SK_ID_CURR')['CREDIT_ACTIVE'].mean().reset_index().rename(index=str, columns={'CREDIT_ACTIVE': 'ACTIVE_LOANS_PERCENTAGE'})

bureau_new = bureau_new.merge(gp, on = 'SK_ID_CURR', how = 'left')
bureau_new.head()
del gp
# gp = bureau_new[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT']].groupby('SK_ID_CURR')
# gp1 = gp.apply(lambda x: x.sort_values(['DAYS_CREDIT'], ascending = False)).reset_index(drop=True)

# # Difference between the days
# gp1["DAYS_CREDIT1"] = gp1["DAYS_CREDIT"]*-1
# gp1['DAYS_DIFF']  = gp1.groupby(by = ['SK_ID_CURR'])['DAYS_CREDIT1'].diff()
# gp1['DAYS_DIFF'] = gp1['DAYS_DIFF'].fillna(0).astype('uint32')
# del gp1['DAYS_CREDIT'], gp1['DAYS_CREDIT1'], gp1['SK_ID_CURR']
# bureau_new = bureau_new.merge(gp1, on = 'SK_ID_BUREAU', how = 'left')
def repl(x):
    if x < 0:
        y = 0
    else:
        y= 1
    return y
bureau_new['CREDIT_ENDDATE_BINARY'] = bureau_new['DAYS_CREDIT_ENDDATE'].apply(lambda x: repl(x))
grp = bureau_new.groupby('SK_ID_CURR')['CREDIT_ENDDATE_BINARY'].mean().reset_index().rename(index=str, columns={'CREDIT_ENDDATE_BINARY': 'CREDIT_ENDDATE_PERCENTAGE'})
bureau_new = bureau_new.merge(grp, on = 'SK_ID_CURR', how = 'left')
del grp
# get some summary stats of numeric variables
num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],

    }

bureau_agg = bureau_new.groupby('SK_ID_CURR').agg({**num_aggregations})
bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
bureau_agg.reset_index(inplace=True)

#now merge with bureau_new on SK_ID_CURR
bureau_merge = bureau_new.merge(bureau_agg, on = 'SK_ID_CURR', how = 'left')
del bureau_agg
buro_cat_features = [bcol for bcol in bureau_merge.columns if bureau_merge[bcol].dtype == 'object']
buro = pd.get_dummies(bureau_merge, columns=buro_cat_features)

# Bureau Balance 
cat_columns = [col for col in bureau_balance.columns if bureau_balance[col].dtype == 'object']
bureau_balance = pd.get_dummies(bureau_balance,cat_columns, dummy_na = True)
bb_group = bureau_balance.groupby('SK_ID_BUREAU').agg(['min', 'max', 'mean'])
bb_group.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_group.columns.tolist()])
bb_group.reset_index(inplace=True)

buro = buro.merge(bb_group, on = 'SK_ID_BUREAU', how = 'left')
avg_buro = buro.groupby('SK_ID_CURR').mean() ## this gives us average values for each columns as we have multiple loans per person

# # Number of loans per person
avg_buro['buro_count'] = buro[['SK_ID_BUREAU', 'SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
del avg_buro['SK_ID_BUREAU'], bb_group
cat_columns = [col for col in installments_payments.columns if installments_payments[col].dtype == 'object']
installments_payments = pd.get_dummies(installments_payments,cat_columns, dummy_na = True)
installments_payments['AMOUNT_DIFF'] = installments_payments['AMT_INSTALMENT'] - installments_payments['AMT_PAYMENT']
installments_payments['AMOUNT_PERC'] =  installments_payments['AMT_PAYMENT']/installments_payments['AMT_INSTALMENT']

# Was it paid on early or late?
installments_payments['DAYS_P'] =  installments_payments['DAYS_ENTRY_PAYMENT']-installments_payments['DAYS_INSTALMENT']
installments_payments['DAYS_I'] =  installments_payments['DAYS_INSTALMENT']-installments_payments['DAYS_ENTRY_PAYMENT']
# installments_payments['DAYS_P'] = installments_payments['DAYS_P'].apply(lambda x: x if x > 0 else 0)
# installments_payments['DAYS_I'] = installments_payments['DAYS_I'].apply(lambda x: x if x > 0 else 0)

aggregations = {
        'NUM_INSTALMENT_VERSION': ['nunique'],
        'DAYS_P': ['max', 'mean', 'sum'],
        'DAYS_I': ['max', 'mean', 'sum'],
        'AMOUNT_DIFF': ['max', 'mean', 'sum', 'var'],
        'AMOUNT_PERC': ['max', 'mean', 'sum', 'var'],
        'AMT_INSTALMENT': ['max', 'mean', 'sum'],
        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
    }
for cat in cat_columns:
    aggregations[cat] = ['mean']
installments_payments_agg = installments_payments.groupby('SK_ID_CURR').agg(aggregations)
installments_payments_agg['INSTAL_COUNT'] = installments_payments.groupby('SK_ID_CURR').size()
installments_payments_agg.columns = pd.Index(['INSTALL_' + e[0] + "_" + e[1].upper() for e in installments_payments_agg.columns.tolist()])

installments_payments = installments_payments.merge(installments_payments_agg, how = 'left', on = 'SK_ID_CURR')
del installments_payments_agg
## Features
# 365243 is NAN
previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

#previous_application[previous_application['AMT_DOWN_PAYMENT'] < 0] = 0
previous_application['INTEREST_PERC'] = (previous_application['RATE_INTEREST_PRIMARY']/100)*previous_application['AMT_DOWN_PAYMENT']
previous_application['INTEREST_ANN_PERC'] = (previous_application['RATE_INTEREST_PRIMARY']/100)*previous_application['AMT_ANNUITY']
previous_application['INTEREST_CREDIT_PERC'] = (previous_application['RATE_INTEREST_PRIMARY']/100)*previous_application['AMT_CREDIT']
previous_application['FIRST_LAST'] = previous_application['DAYS_FIRST_DUE'] - previous_application['DAYS_LAST_DUE']

previous_application['APPLICATION_ACTUAL_CREDIT'] = previous_application['AMT_APPLICATION']/previous_application['AMT_CREDIT']

num_aggregations = {
        'AMT_ANNUITY': ['min', 'max', 'mean'],
        'AMT_APPLICATION': ['min', 'max', 'mean'],
        'AMT_CREDIT': ['min', 'max', 'mean'],
        'INTEREST_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
        'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
        'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
        'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
        'DAYS_DECISION': ['min', 'max', 'mean'],
        'CNT_PAYMENT': ['mean', 'sum'],
        'FIRST_LAST': ['mean', 'max', 'min']
    }

prev_agg = previous_application.groupby('SK_ID_CURR').agg({**num_aggregations})
prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
previous_application = previous_application.merge(prev_agg, on = 'SK_ID_CURR', how = 'left')
del prev_agg
# Previous Applications: Approved Applications - only numerical features
approved = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Approved']
approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
approved_agg.columns = pd.Index(['APPROVED_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
previous_application = previous_application.join(approved_agg, how='left', on='SK_ID_CURR')

# Previous Applications: Refused Applications - only numerical features
refused = previous_application[previous_application['NAME_CONTRACT_STATUS'] == 'Refused']
refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
refused_agg.columns = pd.Index(['REFUSED_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
previous_application = previous_application.join(refused_agg, how='left', on='SK_ID_CURR')

previous_application = previous_application.groupby('SK_ID_CURR').mean().reset_index(inplace=True)
# del previous_application['SK_ID_PREV']
aggregations = {
        'MONTHS_BALANCE': ['max', 'mean', 'size'],
        'SK_DPD': ['max', 'mean'],
        'SK_DPD_DEF': ['max', 'mean']
    }
POS_CASH_AGG = POS_CASH_balance.groupby('SK_ID_CURR').agg(aggregations)
POS_CASH_AGG.columns = pd.Index(['POS_CASH_' + e[0] + "_" + e[1].upper() for e in POS_CASH_AGG.columns.tolist()])
POS_CASH_AGG['COUNT'] = POS_CASH_AGG.groupby('SK_ID_CURR').size()

cat_columns = [col for col in POS_CASH_balance.columns if POS_CASH_balance[col].dtype == 'object']
POS_CASH_balance = pd.get_dummies(POS_CASH_balance,cat_columns, dummy_na = True)
POS_CASH_balance = POS_CASH_balance.merge(POS_CASH_AGG, how = 'left', on = 'SK_ID_CURR')
POS_CASH_balance.head()
POS_CASH_balance = POS_CASH_balance.groupby('SK_ID_CURR').mean().reset_index()
del POS_CASH_AGG, POS_CASH_balance['SK_ID_PREV']

y = data['TARGET']
del data['TARGET']
#One-hot encoding of categorical features in data and test sets
categorical_features = [col for col in data.columns if data[col].dtype == 'object']

one_hot_df = pd.concat([data,test])
one_hot_df = pd.get_dummies(one_hot_df, columns=categorical_features)

data = one_hot_df.iloc[:data.shape[0],:]
test = one_hot_df.iloc[data.shape[0]:,]

print(data.shape, test.shape)
#Bureau and Bureau Balance
data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
print(data.shape, test.shape)

# Previous Application
data = data.merge(right=previous_application.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=previous_application.reset_index(), how='left', on='SK_ID_CURR')
print(data.shape, test.shape)

# POS_CASH_BALANCE
data = data.merge(right=POS_CASH_balance.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=POS_CASH_balance.reset_index(), how='left', on='SK_ID_CURR')
print(data.shape, test.shape)


# Installments_payments
data = data.merge(right=installments_payments.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=installments_payments.reset_index(), how='left', on='SK_ID_CURR')
print(data.shape, test.shape)
gc.collect()
#Remove features with many missing values
print('Removing features with more than 80% missing...')
test = test[test.columns[data.isnull().mean() < 0.80]]
data = data[data.columns[data.isnull().mean() < 0.80]]

print(data.shape, test.shape)
from lightgbm import LGBMClassifier
import gc

gc.enable()

folds = KFold(n_splits=4, shuffle=True, random_state=546789)
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])

feature_importance_df = pd.DataFrame()

feats = [f for f in data.columns if f not in ['SK_ID_CURR']]

for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
    trn_x, trn_y = data[feats].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[feats].iloc[val_idx], y.iloc[val_idx]
    
    clf = LGBMClassifier(
        n_estimators=10000,
        learning_rate=0.03,
        num_leaves=34,
        colsample_bytree=0.9,
        subsample=0.8,
        max_depth=8,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=300,
        silent=-1,
        verbose=-1,
        )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=100, early_stopping_rounds=100  #30
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()

print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 

test['TARGET'] = sub_preds

test[['SK_ID_CURR', 'TARGET']].to_csv('submission1LGBM.csv', index=False)

# Plot feature importances
cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)];
# Plot importances
import seaborn as sns
best_features.head()

import numpy as np 

import pandas as pd 

from sklearn.preprocessing import LabelEncoder

import random

import gc



from lightgbm import LGBMClassifier

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.model_selection import KFold, StratifiedKFold

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import os

print(os.listdir("../input"))
# transform to categorical variable



def categorical_encoder(df, key, one_hot = False) : 

    df_ = df.copy()

    categorical_cols = df.columns[df.dtypes == 'object'].tolist()

    numeric_cols = df.columns[df.dtypes != 'object'].tolist()

    numeric_cols = [col for col in numeric_cols if "SK_ID" not in col]

    

    dummies = None

    if one_hot:

        dummies = pd.get_dummies(df, columns = categorical_cols, dummy_na = True)

        dummies = dummies[ [col for col in dummies.columns if col not in numeric_cols]]

        dummies[key] = df_[key]

        

    encoder = LabelEncoder()

    for col in categorical_cols:

        df_[col] = encoder.fit_transform(df[col].fillna("NA"))

        

    return df_,dummies,categorical_cols,numeric_cols

                              
# customized aggregation function

# key: group condition

# aggregation: aggregation conditions(in dictionary)

# multiple: multiple aggregation conditions or not



def aggregation_customed(df, key, aggregation, multiple = True, suffix = False):

    df_agg = df.groupby(key).agg(aggregation)

    if multiple:

        df_agg.columns = [(a+'_'+b).lower().title() for a,b in df_agg.columns]

    else:

        df_agg.columns = [(a+'_'+b).lower().title() for a,b in aggregation.items()]

    if suffix:

        df_agg.columns = [a+'_'+suffix for a in df_agg.columns]

    return df_agg
# join function

# merged: base df for merge

# df: df to be merged



def join(merged, df, key, aggregation, suffix):

    df_agg = df.groupby(key).agg(aggregation)

    if suffix:

        df_agg.columns = [(suffix + '_' + a +'_'+b).upper() for a,b in df_agg.columns]

    merged = merged.merge(df_agg, left_on = key, right_index = True, how = 'left')

    return merged
debug = True

sample_size = 5000 if debug else None
app_df = pd.read_csv('../input/application_train.csv', nrows = sample_size)

app_test = pd.read_csv('../input/application_test.csv', nrows = sample_size)

app_df = app_df.append(app_test).reset_index()



del app_test

gc.collect



print('Application Original Shape :',app_df.shape)

app_df.head()
app_df, dummies, categorical, numeric = categorical_encoder(app_df, 'SK_ID_CURR', True)

app_df['Days_Employed_Perc'] = app_df['DAYS_EMPLOYED'] / app_df['DAYS_BIRTH']

app_df['Income_Credit_Perc'] = app_df['AMT_INCOME_TOTAL'] / app_df['AMT_CREDIT']

app_df['Income_Per_Person'] = app_df['AMT_INCOME_TOTAL'] / app_df['CNT_FAM_MEMBERS']

app_df['Annuity_Income_Perc'] = app_df['AMT_ANNUITY'] / app_df['AMT_INCOME_TOTAL']

app_df['Loan_Income_Ratio'] = app_df['AMT_CREDIT'] / app_df['AMT_INCOME_TOTAL']

app_df['Annuity_Length'] = app_df['AMT_CREDIT'] / app_df['AMT_ANNUITY']

app_df['Children_ratio'] = app_df['CNT_CHILDREN'] / app_df['CNT_FAM_MEMBERS']

app_df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)

print('Original application shape :', app_df.shape)
bureau_balance_df = pd.read_csv('../input/bureau_balance.csv', nrows = sample_size)

key = 'SK_ID_BUREAU'

bureau_balance_df, dummies, categorical, numeric = categorical_encoder(bureau_balance_df, key, True)
print('BUREAU_BALANCE ORIGINAL SHAPE :',bureau_balance_df.shape)

bureau_balance_df.head()
bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}

bureau_balance_agg = aggregation_customed(bureau_balance_df, key, bb_aggregations)

bureau_balance_agg = bureau_balance_agg.merge(dummies.groupby(key).mean(), left_index=True, right_index=True, how='left')

print('BUREAU_BALANCE AGGREGATION SHAPE :', bureau_balance_agg.shape)

bureau_balance_agg.head()
bureau_df = pd.read_csv('../input/bureau.csv', nrows = sample_size)

key = 'SK_ID_CURR'

print('BUREAU Original SHAPE :', bureau_df.shape)

bureau_df = pd.merge(bureau_df, bureau_balance_agg, left_on='SK_ID_BUREAU', right_index=True, how='left')

bureau_df, dummies, categorical, numeric = categorical_encoder(bureau_df, key, True)

print('BUREAU Merged SHAPE :', bureau_df.shape)

bureau_df.head()
num_aggregations = {}

for cal in numeric[ : -9] : 

    num_aggregations.update({cal : ['min', 'max', 'mean', 'sum']})

for cal in ['Days_credit'] : 

    num_aggregations.update({cal.upper() : ['min', 'max', 'mean', 'sum', 'var']})

num_aggregations
bureau_agg = aggregation_customed(bureau_df, key, num_aggregations)

bureau_agg = bureau_agg.merge(dummies.groupby(key).mean(), left_index=True, right_index=True, how='left')

print('BUREAU AGGREGATION SHAPE :', bureau_agg.shape)
active = bureau_df[bureau_df.CREDIT_ACTIVE == 1] 

closed = bureau_df[bureau_df.CREDIT_ACTIVE == 0] 

bureau_agg = bureau_agg.merge(aggregation_customed(active, key, num_aggregations, True, 'Ative'), left_index=True, right_index=True, how='left')

bureau_agg = bureau_agg.merge(aggregation_customed(closed, key, num_aggregations, True, 'Closed'), left_index=True, right_index=True, how='left')



print('BUREAU AGGREGATION SHAPE :', bureau_agg.shape)
del bureau_balance_df, bureau_df

gc.collect()
prev_app_df = pd.read_csv('../input/previous_application.csv', nrows = sample_size)

prev_app_df['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)

prev_app_df['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)

prev_app_df['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)

prev_app_df['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)

prev_app_df['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)

prev_app_df['APP_CREDIT_PERC'] = prev_app_df['AMT_APPLICATION'] / prev_app_df['AMT_CREDIT']



prev_app_df, dummies, categorical, numeric = categorical_encoder(prev_app_df, key, True)
aggregations = {'SK_ID_CURR': 'count', 'AMT_CREDIT': 'sum'}

prev_app_agg = aggregation_customed(prev_app_df, key, aggregations, False)

print('PREVIOUS_APPLICATION AGGREGATION SHAPE :', prev_app_agg.shape)
num_aggregations = {}

for cal in numeric : 

    num_aggregations.update({cal : ['min', 'max', 'mean', 'sum']})

    

for cal in ['APP_CREDIT_PERC', 'AMT_CREDIT', 'AMT_ANNUITY'] : 

    num_aggregations.update({cal : ['min', 'max', 'mean', 'sum', 'var']})



prev_app_agg = prev_app_agg.merge(aggregation_customed(prev_app_df, key, num_aggregations), left_index=True, right_index=True, how='left')

prev_app_agg = prev_app_agg.merge(dummies.groupby(key).mean(), left_index=True, right_index=True, how='left')

print('PREVIOUS_APPLICATION AGG SHAPE :', prev_app_agg.shape)
# Mode of Categorical Variables

temp = prev_app_df[categorical + [key]].groupby(key).agg({k: lambda x: str(x.mode().iloc[0]) for k in categorical})

temp.columns = [(i+'_Mode').lower().title() for i in temp.columns]

prev_app_agg = prev_app_agg.merge(temp, left_index=True, right_index=True, how='left')

print('PREVIOUS_APPLICATION AGG SHAPE :', prev_app_agg.shape)
# divide by case of contract status

approved = prev_app_df[ prev_app_df.NAME_CONTRACT_STATUS == 0 ]

refused = prev_app_df[ prev_app_df.NAME_CONTRACT_STATUS == 1 ]

cancelled = prev_app_df[ prev_app_df.NAME_CONTRACT_STATUS == 2 ]



prev_app_agg = prev_app_agg.merge( aggregation_customed(approved, key, num_aggregations, True, 'Approved'), left_index=True, right_index=True, how='left')

prev_app_agg = prev_app_agg.merge( aggregation_customed(refused, key, num_aggregations, True, 'Refused'), left_index=True, right_index=True, how='left')

prev_app_agg = prev_app_agg.merge( aggregation_customed(cancelled, key, num_aggregations, True, 'Cancelled'), left_index=True, right_index=True, how='left')

print('PREVIOUS_APPLICATION AGG SHAPE :', prev_app_agg.shape)
prev_app_agg[['Name_Goods_Category_Mode','Name_Portfolio_Mode','Name_Seller_Industry_Mode']].head()
prev_app_agg.iloc[:,-16:-1].head()
prev_app_df[['SK_ID_CURR']+categorical].sort_values('SK_ID_CURR').head()
del prev_app_df

gc.collect()
key = 'SK_ID_CURR'

pos_cash_df = pd.read_csv('../input/POS_CASH_balance.csv', nrows = sample_size)

pos_cash_df, dummies, categorical, numeric = categorical_encoder(pos_cash_df, key, True)
num_aggregations = {}

for cal in numeric : 

    num_aggregations.update({cal : ['min', 'max', 'mean']})

    

for cal in ['MONTHS_BALANCE'] : 

    num_aggregations.update({cal : ['min', 'max', 'mean', 'size']})



pos_cash_agg = aggregation_customed(pos_cash_df, key, num_aggregations)

pos_cash_agg = pos_cash_agg.merge(dummies.groupby(key).mean(), left_index=True, right_index=True, how='left')

print('POS_CASH Shape :', pos_cash_agg.shape)
wa = lambda x: np.average(x, weights = -1 / pos_cash_df.loc[x.index, 'MONTHS_BALANCE'])

f = {'CNT_INSTALMENT': wa, 'CNT_INSTALMENT_FUTURE': wa, 'SK_DPD': wa, 'SK_DPD_DEF': wa}

temp = pos_cash_df.groupby(key)['CNT_INSTALMENT','CNT_INSTALMENT_FUTURE','SK_DPD', 'SK_DPD_DEF'].agg(f)

temp.columns = [i.lower().title()+'_WeightedA' for i in temp.columns]

pos_cash_agg = pos_cash_agg.merge(temp, left_index=True, right_index=True, how='left')

print('POS_CASH Shape :', pos_cash_agg.shape)
recent_idx = pos_cash_df.groupby(key).MONTHS_BALANCE.idxmax()

idx = pos_cash_df.groupby(key).mean().index

temp = pos_cash_df.iloc[recent_idx][categorical]

temp.columns, temp.index = [i.lower().title() + '_MostRecent' for i in temp.columns], idx



pos_cash_agg = pos_cash_agg.merge(temp, left_index=True, right_index=True, how='left')

print('POS_CASH Shape :', pos_cash_agg.shape)

pos_cash_agg.head()
del pos_cash_df

gc.collect()
install_df = pd.read_csv('../input/installments_payments.csv', nrows = sample_size)

install_df['PAYMENT_Perc'] = install_df['AMT_PAYMENT'] / install_df['AMT_INSTALMENT']

install_df['PAYMENT_Diff'] = install_df['AMT_INSTALMENT'] - install_df['AMT_PAYMENT']

install_df['Dpd'] = install_df['DAYS_ENTRY_PAYMENT'] - install_df['DAYS_INSTALMENT']

install_df['Dbd'] = install_df['DAYS_INSTALMENT'] - install_df['DAYS_ENTRY_PAYMENT']

install_df, dummies, categorical, numeric = categorical_encoder(install_df, key, True)
num_aggregations = {

        'NUM_INSTALMENT_VERSION': ['nunique'],

        'DPD': ['max', 'mean', 'sum'],

        'DBD': ['max', 'mean', 'sum'],

        'PAYMENT_Perc': ['max', 'mean', 'sum', 'var'],

        'PAYMENT_Diff': ['max', 'mean', 'sum', 'var'],

        'AMT_INSTALMENT': ['max', 'mean', 'sum'],

        'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],

        'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']

    }

num_aggregations = {}

for cal in numeric : 

    num_aggregations.update({cal : ['min', 'max', 'mean', 'sum']})

    

for cal in ['PAYMENT_Perc', 'PAYMENT_Diff'] : 

    num_aggregations.update({cal : ['min', 'max', 'mean', 'sum', 'var']})

num_aggregations.update({'NUM_INSTALMENT_VERSION': ['nunique']})    



install_agg = aggregation_customed(install_df, key, num_aggregations)

print('Installment Shape :', install_agg.shape)

install_agg.head()
credit_card_df = pd.read_csv('../input/credit_card_balance.csv', nrows = sample_size)

credit_card_df, dummies, categorical, numeric = categorical_encoder(credit_card_df, key, True)
num_aggregations = {i : ['min', 'max', 'mean', 'sum', 'var'] for i in numeric}

credit_card_agg = aggregation_customed(credit_card_df, key, num_aggregations)

credit_card_agg = credit_card_agg.merge(dummies.groupby(key).mean(), right_index=True, left_index=True, how = 'left')

credit_card_agg['CreditCard_Count'] = credit_card_df.groupby(key).size().values

credit_card_agg.shape
wa = lambda x : np.average(x ,weights = credit_card_df.iloc[x.index]['MONTHS_BALANCE'] )

temp = credit_card_df.groupby(key).agg(wa)

temp.columns = [i.lower().title()+'_WeightedA' for i in temp.columns]

credit_card_agg = credit_card_agg.merge(temp, right_index=True, left_index=True, how = 'left')

print('CREDIT_CARD AGG SHAPE :', credit_card_agg.shape)
recent_idx = credit_card_df.groupby(key).MONTHS_BALANCE.idxmax()

idx = credit_card_df.groupby(key).mean().index

temp = credit_card_df.iloc[recent_idx][categorical]

temp.columns, temp.index = [i.lower().title() + '_MostRecent' for i in temp.columns], idx

credit_card_agg = credit_card_agg.merge(temp, right_index=True, left_index=True, how = 'left')



print('CREDIT_CARD AGG SHAPE :', credit_card_agg.shape)

credit_card_agg.head()
del credit_card_df, install_df

gc.collect()
Aggregated_df = [bureau_agg, prev_app_agg, pos_cash_agg, install_agg, credit_card_agg]

for agg in Aggregated_df : 

    print(agg.shape)

sum( [i.shape[1] for i in Aggregated_df] )
for agg in Aggregated_df : 

    print(app_df.shape)

    app_df = app_df.merge(agg, left_on=key, right_index=True, how = 'left')

print('Merged SHAPE :',app_df.shape)



app_df.head()
#del Aggregated_df

#gc.collect()
app_df.to_csv("merged.csv",index = False)
# Divide into training/validation and test data

app_df, _, _, _ = categorical_encoder(app_df, 'SK_ID_CURR', True)



train = app_df[app_df['TARGET'].notnull()]

test = app_df[app_df['TARGET'].isnull()]

print("Train shape: {}, Test shape: {}".format(train.shape, test.shape))

#del app_df

gc.collect()
# 10 fold Cross validation

num_folds = 10



folds = KFold(n_splits= num_folds, shuffle=True, random_state=2018)



train_preds = np.zeros(train.shape[0])

test_preds = np.zeros(test.shape[0])



importance_df = pd.DataFrame()

cols = [col for col in train.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]



for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[cols], train['TARGET'])):

    train_x, train_y = train[cols].iloc[train_idx], train['TARGET'].iloc[train_idx]

    valid_x, valid_y = train[cols].iloc[valid_idx], train['TARGET'].iloc[valid_idx]





    lgb = LGBMClassifier(

        nthread=4,

        n_estimators=10000,

        learning_rate=0.02,

        num_leaves=34,

        colsample_bytree=0.9497,

        subsample=0.8715,

        max_depth=8,

        reg_alpha=0.041545,

        reg_lambda=0.0735,

        min_split_gain=0.0222,

        min_child_weight=39.3250,

        silent=-1,

        verbose=-1, )



    lgb.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 

        eval_metric= 'auc', verbose= 200, early_stopping_rounds= 200)



    train_preds[valid_idx] = lgb.predict_proba(valid_x, num_iteration=lgb.best_iteration_)[:, 1]

    test_preds += lgb.predict_proba(test[cols], num_iteration=lgb.best_iteration_)[:, 1] / folds.n_splits



    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = cols

    fold_importance_df["importance"] = lgb.feature_importances_

    fold_importance_df["fold"] = n_fold + 1

    importance_df = pd.concat([importance_df, fold_importance_df], axis=0)

    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, train_preds[valid_idx])))

    

    del lgb, train_x, train_y, valid_x, valid_y

    gc.collect()

          

test['TARGET'] = test_preds

test[['SK_ID_CURR', 'TARGET']].to_csv("submission.csv", index= False)

print('Full AUC score %.6f' % roc_auc_score(train['TARGET'], train_preds))





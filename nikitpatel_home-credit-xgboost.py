import numpy as np
import numpy
import pandas as pd
import gc
import time
from contextlib import contextmanager
from lightgbm import LGBMClassifier
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score, roc_curve,r2_score,mean_absolute_error
from sklearn.model_selection import KFold, StratifiedKFold,train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

### plot packages
import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
import plotly.figure_factory as ff
import matplotlib as plt
import plotly.graph_objs as go
import plotly.tools as tls
%matplotlib inline
import cufflinks as cf
cf.go_offline()

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
num_rows = None
nan_as_category = True
print ("Start Train Test ................. ")
df = pd.read_csv('../input/application_train.csv', nrows= num_rows)
test_df = pd.read_csv('../input/application_test.csv', nrows= num_rows)
print("Train samples: {}, test samples: {}".format(len(df), len(test_df)))
df = df.append(test_df).reset_index()
del test_df
gc.collect()
# Categorical features: Binary features and One-Hot encoding
for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
    df[bin_feature], uniques = pd.factorize(df[bin_feature])
df, cat_cols = one_hot_encoder(df, nan_as_category)
# NaN values for DAYS_EMPLOYED: 365.243 -> nan
df['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
df['DAYS_EMPLOYED_PERC'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
df['INCOME_CREDIT_PERC'] = df['AMT_INCOME_TOTAL'] / df['AMT_CREDIT']
df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
df['ANNUITY_INCOME_PERC'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
a = df['DAYS_EMPLOYED_PERC'].tolist()
a = [x for x in a if str(x) != 'nan']
b = df['INCOME_CREDIT_PERC'].tolist()
b = [x for x in b if str(x) != 'nan']
c = df['INCOME_PER_PERSON'].tolist()
c = [x for x in c if str(x) != 'nan']
d = df['ANNUITY_INCOME_PERC'].tolist()
d = [x for x in d if str(x) != 'nan']
# data = [go.Histogram(x=a,marker=dict(color='#1F77B4'),opacity=0.75)]
# layout = go.Layout(title='ANNUITY_INCOME_PERC',xaxis=dict(title='Value'),
#     yaxis=dict(title='Count'))
# fig = go.Figure(data=data, layout=layout)
# py.iplot(fig, filename='ANNUITY_INCOME_PERC')

# data = [go.Histogram(x=b,marker=dict(color='#FF7F0E'),opacity=0.75)]
# layout = go.Layout(title='INCOME_CREDIT_PERC',xaxis=dict(title='Value'),
#     yaxis=dict(title='Count'))
# fig = go.Figure(data=data, layout=layout)
# py.iplot(fig, filename='INCOME_CREDIT_PERC')

# data = [go.Histogram(x=c,marker=dict(color='#2CA02C'),opacity=0.75)]
# layout = go.Layout(title='INCOME_PER_PERSON',xaxis=dict(title='Value'),
#     yaxis=dict(title='Count'))
# fig = go.Figure(data=data, layout=layout)
# py.iplot(fig, filename='INCOME_PER_PERSON')

# data = [go.Histogram(x=d,marker=dict(color='#D62728'),opacity=0.75)]
# layout = go.Layout(title='ANNUITY_INCOME_PERC',xaxis=dict(title='Value'),
#     yaxis=dict(title='Count'))
# fig = go.Figure(data=data, layout=layout)
# py.iplot(fig, filename='ANNUITY_INCOME_PERC')

# del d,data
# gc.collect()
print ("End Train Test.................. ")
print ("Start Bureau................ ")
bureau = pd.read_csv('../input/bureau.csv', nrows = num_rows)
# a = bureau["CREDIT_TYPE"].value_counts()
# a = pd.DataFrame({'labels': a.index,
#                    'values': a.values
#                   })
# a.iplot(kind='pie',labels='labels',values='values', title='CREDIT_TYPE\'s', hole = 0.5)

# b = bureau["CREDIT_ACTIVE"].value_counts()
# b = pd.DataFrame({'labels': b.index,
#                    'values': b.values
#                   })
# b.iplot(kind='pie',labels='labels',values='values', title='CREDIT_ACTIVE\'s', hole = 0.5)

# c = bureau["CREDIT_CURRENCY"].value_counts()
# c = pd.DataFrame({'labels': c.index,
#                    'values': c.values
#                   })
# c.iplot(kind='pie',labels='labels',values='values', title='CREDIT_CURRENCY\'s', hole = 0.5)

# del a,b,c
# gc.collect()
bb = pd.read_csv('../input/bureau_balance.csv', nrows = num_rows)
bb, bb_cat = one_hot_encoder(bb, nan_as_category)
bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)  
# Bureau balance: Perform aggregations and merge with bureau.csv
bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
for col in bb_cat:
    bb_aggregations[col] = ['mean']
bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
bureau.drop(columns= 'SK_ID_BUREAU', inplace= True)
del bb, bb_agg
gc.collect()
# Bureau and bureau_balance numeric features
num_aggregations = {
    'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
    'CREDIT_DAY_OVERDUE': ['max', 'mean'],
    'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
    'AMT_CREDIT_MAX_OVERDUE': ['mean'],
    'CNT_CREDIT_PROLONG': ['sum'],
    'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
    'AMT_CREDIT_SUM_OVERDUE': ['mean'],
    'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
    'DAYS_CREDIT_UPDATE': ['min', 'max', 'mean'],
    'AMT_ANNUITY': ['max', 'mean'],
    'MONTHS_BALANCE_MIN': ['min'],
    'MONTHS_BALANCE_MAX': ['max'],
    'MONTHS_BALANCE_SIZE': ['mean', 'sum']
}
# Bureau and bureau_balance categorical features
cat_aggregations = {}
for cat in bureau_cat: cat_aggregations[cat] = ['mean']
for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])

# Bureau: Active credits - using only numerical aggregations
active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
active_agg.columns = pd.Index(['ACT_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
bureau_agg = bureau_agg.join(active_agg, how='left')
del active, active_agg
gc.collect()
# Bureau: Closed credits - using only numerical aggregations
closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
closed_agg.columns = pd.Index(['CLS_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
bureau_agg = bureau_agg.join(closed_agg, how='left')
del closed, closed_agg, bureau
gc.collect()
print ("End Bureau................ ")
print ("Start previous_application ................ ")
prev = pd.read_csv('../input/previous_application.csv', nrows = num_rows)
# a = prev["NAME_CONTRACT_TYPE"].value_counts()
# a = pd.DataFrame({'labels': a.index,
#                    'values': a.values
#                   })
# a.iplot(kind='pie',labels='labels',values='values', title='NAME_CONTRACT_TYPE\'s', hole = 0.5)

# a = prev["WEEKDAY_APPR_PROCESS_START"].value_counts()
# a.iplot(kind='bar', xTitle = 'Week Name', yTitle = "Count", title = 'WEEKDAY_APPR_PROCESS_START', color = 'red')

# a = prev["NAME_SELLER_INDUSTRY"].value_counts()
# a.iplot(kind='bar', xTitle = 'NAME_SELLER_INDUSTRY Name', yTitle = "Count", title = 'NAME_SELLER_INDUSTRY', color = 'green')

# a = prev["PRODUCT_COMBINATION"].value_counts()
# a.iplot(kind='bar', xTitle = 'PRODUCT_COMBINATION Name', yTitle = "Count", title = 'PRODUCT_COMBINATION', color = 'blue')

# del a
# gc.collect()
prev, cat_cols = one_hot_encoder(prev, nan_as_category= True)
# Days 365.243 values -> nan
prev['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace= True)
prev['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace= True)
prev['DAYS_LAST_DUE'].replace(365243, np.nan, inplace= True)
prev['DAYS_TERMINATION'].replace(365243, np.nan, inplace= True)
prev['APP_CREDIT_PERC'] = prev['AMT_APPLICATION'] / prev['AMT_CREDIT']
# a = prev['APP_CREDIT_PERC'].tolist()
# a = [x for x in a if str(x) != 'nan']

# data = [go.Histogram(x=a,marker=dict(color='#1F77B4'),opacity=0.75)]
# layout = go.Layout(title='APP_CREDIT_PERC',xaxis=dict(title='Value'),
#     yaxis=dict(title='Count'))
# fig = go.Figure(data=data, layout=layout)
# py.iplot(fig, filename='APP_CREDIT_PERC')

# del a,data
# gc.collect()
# Previous applications numeric features
num_aggregations = {
    'AMT_ANNUITY': ['min', 'max', 'mean'],
    'AMT_APPLICATION': ['min', 'max', 'mean'],
    'AMT_CREDIT': ['min', 'max', 'mean'],
    'APP_CREDIT_PERC': ['min', 'max', 'mean', 'var'],
    'AMT_DOWN_PAYMENT': ['min', 'max', 'mean'],
    'AMT_GOODS_PRICE': ['min', 'max', 'mean'],
    'HOUR_APPR_PROCESS_START': ['min', 'max', 'mean'],
    'RATE_DOWN_PAYMENT': ['min', 'max', 'mean'],
    'DAYS_DECISION': ['min', 'max', 'mean'],
    'CNT_PAYMENT': ['mean', 'sum'],
}
# Previous applications categorical features
cat_aggregations = {}
for cat in cat_cols:
    cat_aggregations[cat] = ['mean']

prev_agg = prev.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])
# Previous Applications: Approved Applications - only numerical features
approved = prev[prev['NAME_CONTRACT_STATUS_Approved'] == 1]
approved_agg = approved.groupby('SK_ID_CURR').agg(num_aggregations)
approved_agg.columns = pd.Index(['APR_' + e[0] + "_" + e[1].upper() for e in approved_agg.columns.tolist()])
prev_agg = prev_agg.join(approved_agg, how='left')
# Previous Applications: Refused Applications - only numerical features
refused = prev[prev['NAME_CONTRACT_STATUS_Refused'] == 1]
refused_agg = refused.groupby('SK_ID_CURR').agg(num_aggregations)
refused_agg.columns = pd.Index(['REF_' + e[0] + "_" + e[1].upper() for e in refused_agg.columns.tolist()])
prev_agg = prev_agg.join(refused_agg, how='left')
del refused, refused_agg, approved, approved_agg, prev
gc.collect()
    
prev_agg.shape
print ("End previous_application ................ ")
print ("Start POS_CASH_balance ................ ")
pos = pd.read_csv('../input/POS_CASH_balance.csv', nrows = num_rows)
pos, cat_cols = one_hot_encoder(pos, nan_as_category= True)
# Features
aggregations = {
    'MONTHS_BALANCE': ['max', 'mean', 'size'],
    'SK_DPD': ['max', 'mean'],
    'SK_DPD_DEF': ['max', 'mean']
}
for cat in cat_cols:
    aggregations[cat] = ['mean']

pos_agg = pos.groupby('SK_ID_CURR').agg(aggregations)
pos_agg.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper() for e in pos_agg.columns.tolist()])
# Count pos cash accounts
pos_agg['POS_COUNT'] = pos.groupby('SK_ID_CURR').size()
del pos
gc.collect()
pos_agg.shape
print ("Start POS_CASH_balance ................ ")
ins = pd.read_csv('../input/installments_payments.csv', nrows = num_rows)
ins, cat_cols = one_hot_encoder(ins, nan_as_category= True)
# Percentage and difference paid in each installment (amount paid and installment value)
ins['PAYMENT_PERC'] = ins['AMT_PAYMENT'] / ins['AMT_INSTALMENT']
ins['PAYMENT_DIFF'] = ins['AMT_INSTALMENT'] - ins['AMT_PAYMENT']
# Days past due and days before due (no negative values)
ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)
# Features: Perform aggregations
aggregations = {
    'NUM_INSTALMENT_VERSION': ['nunique'],
    'DPD': ['max', 'mean', 'sum'],
    'DBD': ['max', 'mean', 'sum'],
    'PAYMENT_PERC': ['max', 'mean', 'sum', 'var'],
    'PAYMENT_DIFF': ['max', 'mean', 'sum', 'var'],
    'AMT_INSTALMENT': ['max', 'mean', 'sum'],
    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
    'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum']
}
for cat in cat_cols:
    aggregations[cat] = ['mean']
ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
ins_agg.columns = pd.Index(['INS_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])
# Count installments accounts
ins_agg['INS_COUNT'] = ins.groupby('SK_ID_CURR').size()
del ins
gc.collect()
ins_agg.shape
   
print ("End POS_CASH_balance ................ ")
print ("Start credit_card_balance ................ ")
cc = pd.read_csv('../input/credit_card_balance.csv', nrows = num_rows)
# a = cc["NAME_CONTRACT_STATUS"].value_counts()
# a = pd.DataFrame({'labels': a.index,
#                    'values': a.values
#                   })
# a.iplot(kind='pie',labels='labels',values='values', title='NAME_CONTRACT_STATUS\'s', hole = 0.5)

# del a
# gc.collect()
cc, cat_cols = one_hot_encoder(cc, nan_as_category= True)
# General aggregations
cc.drop(columns = ['SK_ID_PREV'], inplace = True)
cc_agg = cc.groupby('SK_ID_CURR').agg(['min', 'max', 'mean', 'sum', 'var'])
cc_agg.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper() for e in cc_agg.columns.tolist()])
# Count credit card lines
cc_agg['CC_COUNT'] = cc.groupby('SK_ID_CURR').size()

del cc
gc.collect()
cc_agg.shape
print ("End credit_card_balance ................ ")
with timer("Process bureau and bureau_balance"):
    print("Bureau df shape:", bureau_agg.shape)
    df = df.join(bureau_agg, how='left',on='SK_ID_CURR')
    gc.collect()
with timer("Process previous_applications"):
    print("Previous applications df shape:", prev_agg.shape)
    df = df.join(prev_agg, how='left', on='SK_ID_CURR')
    gc.collect()
with timer("Process POS-CASH balance"): 
    print("Pos-cash balance df shape:", pos_agg.shape)
    df = df.join(pos_agg, how='left', on='SK_ID_CURR')
    gc.collect()
with timer("Process installments payments"): 
    print("Installments payments df shape:", ins_agg.shape)
    df = df.join(ins_agg, how='left', on='SK_ID_CURR')
    gc.collect()
with timer("Process credit card balance"):
    print("Credit card balance df shape:", cc_agg.shape)
    df = df.join(cc_agg, how='left', on='SK_ID_CURR')
    gc.collect()
del bureau_agg,prev_agg,pos_agg,ins_agg,cc_agg
gc.collect()
print("Done .;..............")


train_df = df[df['TARGET'].notnull()]
test_df = df[df['TARGET'].isnull()]
print("Starting Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
del df
gc.collect()
train_df = train_df.drop(['index'],axis=1)
test_df = test_df.drop(['index','TARGET'],axis=1)
train_df = train_df.fillna(0)
test_df = test_df.fillna(0)
label = u'TARGET'
a = list(train_df.columns)
a.remove(label)
labels = train_df[label]
data_only = train_df[list(a)]
col_name = data_only.columns
#data_only = data_only.fillna(0)

X_train, X_test, y_train, y_test = train_test_split(data_only, labels, test_size=0.1,random_state = 42)
# LightGBM parameters found by Bayesian optimization
# clf_lgbm = LGBMClassifier(
#             nthread=4,
#             n_estimators=10000,
#             learning_rate=0.02,
#             num_leaves=34,
#             colsample_bytree=0.9497036,
#             subsample=0.8715623,
#             max_depth=8,
#             reg_alpha=0.041545473,
#             reg_lambda=0.0735294,
#             min_split_gain=0.0222415,
#             min_child_weight=39.3259775,
#             silent=-1,
#             verbose=-1, )

#clf_lgbm.fit(X_train,y_train)
# imp = clf_lgbm.feature_importances_
# col_name = data_only.columns
# d = {'name': col_name,'value':imp}
# d = pd.DataFrame(data =d)
# d = d.sort_values(['value'], ascending=False)
# temp = d.set_index('name')
# temp[:50].iplot(kind='bar',title="Feature IMportant by LGBMClassifier ")
# del clf_lgbm, temp
# gc.collect()
clf_xgBoost = xgb.XGBClassifier(
    learning_rate =0.01, n_estimators=1000, max_depth=4, min_child_weight=4, subsample=0.8, colsample_bytree=0.8, objective= 'binary:logistic', 
            nthread=4, scale_pos_weight=2, seed=27)
# Fit the models
clf_xgBoost.fit(data_only,labels)
# imp = clf_xgBoost.feature_importances_
# col_name = data_only.columns
# d1 = {'name': col_name,'value':imp}
# d1 = pd.DataFrame(data =d1)
# d1 = d1.sort_values(['value'], ascending=False)
# temp = d1.set_index('name')
# temp[:50].iplot(kind='bar',title="Feature IMportant by XGBClassifier")
# del clf_xgBoost, temp
# gc.collect()

# clf_catboost = CatBoostClassifier(iterations=1200,
#                               learning_rate=0.1,
#                               depth=7,
#                               l2_leaf_reg=40,
#                               bootstrap_type='Bernoulli',
#                               subsample=0.7,
#                               scale_pos_weight=5,
#                               eval_metric='AUC',
#                               metric_period=50,
#                               od_type='Iter',
#                               od_wait=45,
#                               random_seed=15,
#                               allow_writing_files=False)

# clf_catboost.fit(data_only,labels,verbose=True)
pred = clf_xgBoost.predict_proba(test_df)
test_df['TARGET'] = pred[:, 0]
test_df[['SK_ID_CURR', 'TARGET']].to_csv('submission_clf_xgBoost.csv', index= False)
















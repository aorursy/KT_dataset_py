import numpy as np
import pandas as pd

from datetime import date, datetime
import gc

import matplotlib.pyplot as plt
import seaborn as sns


from imblearn.pipeline import make_pipeline as make_pipeline_imb # To do our transformation in a unique time
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
from imblearn.metrics import classification_report_imbalanced

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, KFold, GridSearchCV
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import precision_score, recall_score, fbeta_score, confusion_matrix, precision_recall_curve, accuracy_score, mean_squared_error

import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
df_train = pd.read_csv('../input/train.csv')
df_hist_trans = pd.read_csv('../input/historical_transactions.csv')
df_new_merchant_trans = pd.read_csv('../input/new_merchant_transactions.csv')
df_train.head()
for df in [df_hist_trans,df_new_merchant_trans]:
    df['category_2'].fillna(1.0,inplace=True)
    df['category_3'].fillna('A',inplace=True)
    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)
def get_new_columns(name,aggs):
    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
for df in [df_hist_trans,df_new_merchant_trans]:
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    df['year'] = df['purchase_date'].dt.year
    df['weekofyear'] = df['purchase_date'].dt.weekofyear
    df['month'] = df['purchase_date'].dt.month
    df['dayofweek'] = df['purchase_date'].dt.dayofweek
    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)
    df['hour'] = df['purchase_date'].dt.hour
    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})
    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 
    df['month_diff'] = ((datetime.today() - df['purchase_date']).dt.days)//30
    df['month_diff'] += df['month_lag']
aggs = {}
for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']

aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['authorized_flag'] = ['sum', 'mean']
aggs['weekend'] = ['sum', 'mean']
aggs['category_1'] = ['sum', 'mean']
aggs['card_id'] = ['size']

for col in ['category_2','category_3']:
    df_hist_trans[col+'_mean'] = df_hist_trans.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']    

new_columns = get_new_columns('hist',aggs)
df_hist_trans_group = df_hist_trans.groupby('card_id').agg(aggs)
df_hist_trans_group.columns = new_columns
df_hist_trans_group.reset_index(drop=False,inplace=True)
df_hist_trans_group['hist_purchase_date_diff'] = (df_hist_trans_group['hist_purchase_date_max'] - df_hist_trans_group['hist_purchase_date_min']).dt.days
df_hist_trans_group['hist_purchase_date_average'] = df_hist_trans_group['hist_purchase_date_diff']/df_hist_trans_group['hist_card_id_size']
df_hist_trans_group['hist_purchase_date_uptonow'] = (datetime.today() - df_hist_trans_group['hist_purchase_date_max']).dt.days

df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')

del df_hist_trans_group;
gc.collect()
aggs = {}
for col in ['month','hour','weekofyear','dayofweek','year','subsector_id','merchant_id','merchant_category_id']:
    aggs[col] = ['nunique']
aggs['purchase_amount'] = ['sum','max','min','mean','var']
aggs['installments'] = ['sum','max','min','mean','var']
aggs['purchase_date'] = ['max','min']
aggs['month_lag'] = ['max','min','mean','var']
aggs['month_diff'] = ['mean']
aggs['weekend'] = ['sum', 'mean']
aggs['category_1'] = ['sum', 'mean']
aggs['card_id'] = ['size']

for col in ['category_2','category_3']:
    df_new_merchant_trans[col+'_mean'] = df_new_merchant_trans.groupby([col])['purchase_amount'].transform('mean')
    aggs[col+'_mean'] = ['mean']
    
new_columns = get_new_columns('new_hist',aggs)
df_hist_trans_group = df_new_merchant_trans.groupby('card_id').agg(aggs)
df_hist_trans_group.columns = new_columns
df_hist_trans_group.reset_index(drop=False,inplace=True)
df_hist_trans_group['new_hist_purchase_date_diff'] = (df_hist_trans_group['new_hist_purchase_date_max'] - df_hist_trans_group['new_hist_purchase_date_min']).dt.days
df_hist_trans_group['new_hist_purchase_date_average'] = df_hist_trans_group['new_hist_purchase_date_diff']/df_hist_trans_group['new_hist_card_id_size']
df_hist_trans_group['new_hist_purchase_date_uptonow'] = (datetime.today() - df_hist_trans_group['new_hist_purchase_date_max']).dt.days

df_train = df_train.merge(df_hist_trans_group,on='card_id',how='left')

del df_hist_trans_group;
gc.collect()
del df_hist_trans;
gc.collect()

del df_new_merchant_trans;
gc.collect()

df_train.head()
df_train['first_active_month'] = pd.to_datetime(df_train['first_active_month'])
df_train['dayofweek'] = df_train['first_active_month'].dt.dayofweek
df_train['weekofyear'] = df_train['first_active_month'].dt.weekofyear
df_train['month'] = df_train['first_active_month'].dt.month
df_train['elapsed_time'] = (datetime.today() - df_train['first_active_month']).dt.days
df_train['hist_first_buy'] = (df_train['hist_purchase_date_min'] - df_train['first_active_month']).dt.days
df_train['new_hist_first_buy'] = (df_train['new_hist_purchase_date_min'] - df_train['first_active_month']).dt.days
for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\
                    'new_hist_purchase_date_min']:
    df_train[f] = df_train[f].astype(np.int64) * 1e-9
df_train['card_id_total'] = df_train['new_hist_card_id_size']+df_train['hist_card_id_size']
df_train['purchase_amount_total'] = df_train['new_hist_purchase_amount_sum'] + df_train['hist_purchase_amount_sum']
df_train = pd.get_dummies(df_train, columns=['feature_1', 'feature_2'])

df_train.head()
df_train['outliers_target'] = 0
df_train.loc[df_train['target'] < -30, 'outliers_target'] = 1
df_train['outliers_target'].value_counts()
df_train.columns[df_train.isna().any()]
df_train.drop(['new_hist_month_nunique', 'new_hist_hour_nunique',
       'new_hist_weekofyear_nunique', 'new_hist_dayofweek_nunique',
       'new_hist_year_nunique', 'new_hist_subsector_id_nunique',
       'new_hist_merchant_id_nunique', 'new_hist_merchant_category_id_nunique',
       'new_hist_purchase_amount_sum', 'new_hist_purchase_amount_max',
       'new_hist_purchase_amount_min', 'new_hist_purchase_amount_mean',
       'new_hist_purchase_amount_var', 'new_hist_installments_sum',
       'new_hist_installments_max', 'new_hist_installments_min',
       'new_hist_installments_mean', 'new_hist_installments_var',
       'new_hist_month_lag_max', 'new_hist_month_lag_min',
       'new_hist_month_lag_mean', 'new_hist_month_lag_var',
       'new_hist_month_diff_mean', 'new_hist_weekend_sum',
       'new_hist_weekend_mean', 'new_hist_category_1_sum',
       'new_hist_category_1_mean', 'new_hist_card_id_size',
       'new_hist_category_2_mean_mean', 'new_hist_category_3_mean_mean',
       'new_hist_purchase_date_diff', 'new_hist_purchase_date_average',
       'new_hist_purchase_date_uptonow', 'new_hist_first_buy', 'card_id_total',
       'purchase_amount_total'], axis=1,inplace=True)
X = df_train.drop(['target','card_id','first_active_month','outliers_target'],axis=1)
y = df_train['outliers_target']
def print_results(headline, true_value, pred):
    print(headline)
    print("accuracy: {}".format(accuracy_score(true_value, pred)))
    print("precision: {}".format(precision_score(true_value, pred)))
    print("recall: {}".format(recall_score(true_value, pred)))
    print("f2: {}".format(fbeta_score(true_value, pred, beta=2)))
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.20)

classifier = RandomForestClassifier

# build model with SMOTE imblearn
smote_pipeline = make_pipeline_imb(SMOTE(random_state=42), \
                                   classifier(random_state=42))

smote_model = smote_pipeline.fit(X_train, y_train)
smote_prediction = smote_model.predict(X_test)


print("normal data distribution: {}".format(Counter(y)))

X_smote, y_smote = SMOTE().fit_sample(X, y)

print("SMOTE data distribution: {}".format(Counter(y_smote)))
print("Confusion Matrix: ")
print(confusion_matrix(y_test, smote_prediction))

print('\nSMOTE Pipeline Score {}'.format(smote_pipeline.score(X_test, y_test)))

print_results("\nSMOTE + RandomForest classification", y_test, smote_prediction)

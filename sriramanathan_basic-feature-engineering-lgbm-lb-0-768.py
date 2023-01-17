#Feature engineering inspired from Shanth
#https://www.kaggle.com/c/home-credit-default-risk/discussion/57750
    
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
import gc
pd.set_option('display.max_rows', 500) 
pd.set_option('display.max_colwidth', -1)

# CSV Data Loading
data = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
prev = pd.read_csv('../input/previous_application.csv')
buro = pd.read_csv('../input/bureau.csv')
buro=buro.rename( {'AMT_ANNUITY': 'AMT_ANNUITY_buro'}, axis='columns')

prev.columns=[u'SK_ID_PREV', u'SK_ID_CURR', u'NAME_CONTRACT_TYPE_prev',
       u'AMT_ANNUITY_prev', u'AMT_APPLICATION_prev', u'AMT_CREDIT_prev',
       u'AMT_DOWN_PAYMENT_prev', u'AMT_GOODS_PRICE_prev',
       u'WEEKDAY_APPR_PROCESS_START_prev', u'HOUR_APPR_PROCESS_START_prev',
       u'FLAG_LAST_APPL_PER_CONTRACT_prev', u'NFLAG_LAST_APPL_IN_DAY_prev',
       u'RATE_DOWN_PAYMENT_prev', u'RATE_INTEREST_PRIMARY_prev',
       u'RATE_INTEREST_PRIVILEGED_prev', u'NAME_CASH_LOAN_PURPOSE_prev',
       u'NAME_CONTRACT_STATUS_prev', u'DAYS_DECISION_prev',
       u'NAME_PAYMENT_TYPE_prev', u'CODE_REJECT_REASON_prev',
       u'NAME_TYPE_SUITE_prev', u'NAME_CLIENT_TYPE_prev',
       u'NAME_GOODS_CATEGORY_prev', u'NAME_PORTFOLIO_prev',
       u'NAME_PRODUCT_TYPE_prev', u'CHANNEL_TYPE_prev',
       u'SELLERPLACE_AREA_prev', u'NAME_SELLER_INDUSTRY_prev',
       u'CNT_PAYMENT_prev', u'NAME_YIELD_GROUP_prev',
       u'PRODUCT_COMBINATION_prev', u'DAYS_FIRST_DRAWING_prev',
       u'DAYS_FIRST_DUE_prev', u'DAYS_LAST_DUE_1ST_VERSION_prev',
       u'DAYS_LAST_DUE_prev', u'DAYS_TERMINATION_prev',
       u'NFLAG_INSURED_ON_APPROVAL_prev']

#repayment behaviour based on repayment amount
repay=pd.read_csv('../input/installments_payments.csv')
repay['AMT_PAYMENT']=repay['AMT_PAYMENT'].fillna(0)
repay['installment_remaining']=repay['AMT_INSTALMENT']-repay['AMT_PAYMENT']
repay_gps=repay.groupby('SK_ID_PREV').agg(sum)
repay_gps=repay_gps.reset_index()
repay_gps['completion_rate']=repay_gps['installment_remaining']/repay_gps['AMT_INSTALMENT']

#repayment behaviour across all previous applications
repay_curr=repay_gps.groupby('SK_ID_CURR').agg('mean')
repay_curr=repay_curr.reset_index()
repays=repay_curr[['SK_ID_CURR','installment_remaining','completion_rate']]
#bureau loans freq
buros=buro.groupby('SK_ID_CURR').agg('count')
buros=buros.reset_index()
buros=buros[['SK_ID_CURR', u'SK_ID_BUREAU']]
buros.columns=[u'SK_ID_CURR', u'prev_loan_count']
#active loans count
buro_active=buro[buro['CREDIT_ACTIVE']=='Active']
buros_active=buro_active.groupby('SK_ID_CURR').agg('count')
buros_active=buros_active.reset_index()
buros_active=buros_active[['SK_ID_CURR', u'SK_ID_BUREAU']]
buros_active.columns=[u'SK_ID_CURR', u'active_loan_count']

#closed loans count
buro_closed=buro[buro['CREDIT_ACTIVE']=='Closed']
buros_closed=buro_closed.groupby('SK_ID_CURR').agg('count')
buros_closed=buros_closed.reset_index()
buros_closed=buros_closed[['SK_ID_CURR', u'SK_ID_BUREAU']]
buros_closed.columns=[u'SK_ID_CURR', u'closed_loan_count']
#repayment behaviour based on freq of repayment 
#needs refinement
repay=pd.read_csv('../input/installments_payments.csv')
repay['AMT_PAYMENT']=repay['AMT_PAYMENT'].fillna('nill')

missed=repay[repay['AMT_PAYMENT']=='nill']['SK_ID_PREV'].value_counts()
missed=pd.DataFrame(missed)
missed=missed.reset_index()
missed.columns=['SK_ID_PREV','miss']

all_repays=repay['SK_ID_PREV'].value_counts()
all_repays=pd.DataFrame(all_repays)
all_repays=all_repays.reset_index()
all_repays.columns=['SK_ID_PREV','all']

all_repays=all_repays.merge(missed,on='SK_ID_PREV',how='left')
all_repays['miss']=all_repays['miss'].fillna(0)

all_repays['repay_rate']=(all_repays['all']-all_repays['miss'])/all_repays['all']
#bureau data and prev home credit loans merges
y = data['TARGET']
del data['TARGET']
 
avg_prev = prev.groupby('SK_ID_CURR').mean()
cnt_prev = prev[['SK_ID_CURR', 'SK_ID_PREV']].groupby('SK_ID_CURR').count()
avg_prev['nb_app'] = cnt_prev['SK_ID_PREV']
del avg_prev['SK_ID_PREV']

avg_buro = buro.groupby('SK_ID_CURR').mean()
avg_buro['buro_count'] = buro[['SK_ID_BUREAU','SK_ID_CURR']].groupby('SK_ID_CURR').count()['SK_ID_BUREAU']
del avg_buro['SK_ID_BUREAU']

data = data.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
data = data.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')

test = test.merge(right=avg_prev.reset_index(), how='left', on='SK_ID_CURR')
test = test.merge(right=avg_buro.reset_index(), how='left', on='SK_ID_CURR')
#credits and bureau balance merge
ccd=pd.read_csv('../input/credit_card_balance.csv')
ccd_gp=ccd.groupby('SK_ID_CURR').agg('mean')

#previous balance monthly
bb=pd.read_csv('../input/bureau_balance.csv')
bb_balance=bb.groupby('SK_ID_BUREAU').agg('mean')
bbb=bb_balance.reset_index()

#credit card balance
ccd_gpi=ccd_gp.reset_index()

ccd_gpi.head()

data=data.merge(ccd_gpi,on='SK_ID_CURR',how='left')

test=test.merge(ccd_gpi,on='SK_ID_CURR',how='left')
#dummification of non-numeric columns
objects=data.dtypes[data.dtypes=='object']
object_cols=list(objects.index)
numeric_cols=set(data.columns).difference(set(object_cols))
train_cols=list(data.columns)
test=test[train_cols]
train_test=data.append(test)

#dummification
train_test=pd.get_dummies(train_test,columns=object_cols)
#feature engineering
#loan obligation
train_test['loan_obli']=train_test['AMT_CREDIT']/(train_test['AMT_INCOME_TOTAL']-train_test['AMT_CREDIT'])
#first job age
train_test['FIRST_JOB']=train_test['DAYS_BIRTH']-train_test['DAYS_EMPLOYED']
#repayment behaviour
train_test=train_test.merge(repays,on='SK_ID_CURR',how='left')
#bureau loans count
train_test=train_test.merge(buros,on='SK_ID_CURR',how='left')
train_test['prev_loan_count']=train_test['prev_loan_count'].fillna(0)
#active loan count
train_test=train_test.merge(buros_active,on='SK_ID_CURR',how='left')
train_test=train_test.merge(buros_closed,on='SK_ID_CURR',how='left')
#split
train=train_test[0:len(data)]
test=train_test[len(data):len(train_test)]
del train['SK_ID_CURR']
fil_features=train.columns
#cross validation
feats=train.columns
folds = KFold(n_splits=5, shuffle=True, random_state=123)
oof_preds = np.zeros(train.shape[0])
sub_preds = np.zeros(test.shape[0])
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train)):
    trn_x, trn_y = train[fil_features].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = train[fil_features].iloc[val_idx], y.iloc[val_idx]
    
    clf = LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.1,
        num_leaves=123,
        colsample_bytree=.8,
        subsample=.7,
        max_depth=15,
        reg_alpha=.1,
        reg_lambda=.1,
        min_split_gain=.01,
        min_child_weight=2,
        scale_pos_weight=5,
    )
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=250, early_stopping_rounds=150
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += clf.predict_proba(test[fil_features], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
    
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))   

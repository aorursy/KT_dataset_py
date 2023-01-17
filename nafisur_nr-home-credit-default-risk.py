import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score
import gc

application_test=pd.read_csv('../input/application_test.csv')
application_train=pd.read_csv('../input/application_train.csv')
bureau=pd.read_csv('../input/bureau.csv')
bureau_balance=pd.read_csv('../input/bureau_balance.csv')
credit_card_balance=pd.read_csv('../input/credit_card_balance.csv')
installments_payments=pd.read_csv('../input/installments_payments.csv')
POS_CASH_balance=pd.read_csv('../input/POS_CASH_balance.csv')
previous_application=pd.read_csv('../input/previous_application.csv')
def basic_info(df):
    print('Num of rows and columns: ',df.shape)
    print('Missing value status: ',df.isnull().values.any())
    print('Columns names:\n ',df.columns.values)
    return df.head()
def check_missing_data(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = ((df.isnull().sum()/df.isnull().count())*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
def categorical_features(df):
    cat_features=df.columns[df.dtypes=='object']
    return list(cat_features)
def onehot_encoding(df,cat_features_name):
    df=pd.get_dummies(df,columns=cat_features_name)
    return df
basic_info(bureau)
categorical_features(bureau)
bureau.CREDIT_ACTIVE.value_counts()
bureau.CREDIT_CURRENCY.value_counts()
bureau.CREDIT_TYPE.value_counts()
check_missing_data(bureau)[0:9]
bureau.AMT_CREDIT_SUM.plot()
bureau.AMT_CREDIT_SUM.plot(kind='box')
bureau.AMT_CREDIT_SUM.describe()
print(bureau.AMT_CREDIT_SUM.max())
print(bureau.AMT_CREDIT_SUM.mean())
print(bureau.AMT_CREDIT_SUM.median())
bureau.AMT_CREDIT_SUM.fillna(value=bureau.AMT_CREDIT_SUM.median(),inplace=True)
bureau.DAYS_CREDIT_ENDDATE.describe()
bureau['DAYS_CREDIT_ENDDATE']=np.where(bureau.DAYS_CREDIT_ENDDATE.isnull(),bureau.DAYS_ENDDATE_FACT,bureau.DAYS_CREDIT_ENDDATE)
bureau.DAYS_CREDIT_ENDDATE.describe()
bureau.DAYS_CREDIT_ENDDATE.plot(kind='box')
bureau.DAYS_CREDIT_ENDDATE.fillna(value=0.0,inplace=True)
bureau.drop('DAYS_ENDDATE_FACT',axis=1,inplace=True)
bureau[['AMT_ANNUITY','AMT_CREDIT_MAX_OVERDUE','AMT_CREDIT_SUM_LIMIT','AMT_CREDIT_SUM_DEBT']].describe()
bureau.AMT_CREDIT_MAX_OVERDUE.fillna(0.0,inplace=True)
bureau.AMT_CREDIT_SUM_LIMIT.fillna(0.0,inplace=True)
bureau.AMT_CREDIT_SUM_DEBT.fillna(0.0,inplace=True)
bureau.drop('AMT_ANNUITY',axis=1,inplace=True)
check_missing_data(bureau).head()
bureau_onehot=onehot_encoding(bureau,categorical_features(bureau))
bureau_onehot.head()
check_missing_data(bureau_onehot).head()
# clean up. Free RAM space
del bureau
gc.collect()
basic_info(bureau_balance)
month_count=bureau_balance.groupby('SK_ID_BUREAU').size()
bureau_balance.STATUS.value_counts()
bureau_balance_unstack=bureau_balance.groupby('SK_ID_BUREAU')['STATUS'].value_counts(normalize = False).unstack('STATUS')
bureau_balance_unstack.columns=['status_DPD0','status_DPD1','status_DPD2','status_DPD3','status_DPD4','status_DPD5','status_closed','status_X']
bureau_balance_unstack['month_count']=month_count
bureau_balance_unstack.fillna(value=0,inplace=True)
bureau_balance_unstack.head()
# clean up. Free RAM space
del bureau_balance
gc.collect()
bureau_merge=bureau_onehot.merge(bureau_balance_unstack,how='left',on='SK_ID_BUREAU')
cnt_id_bureau=bureau_merge[['SK_ID_CURR','SK_ID_BUREAU']].groupby('SK_ID_CURR').size()
bureau_final_median=bureau_merge.groupby('SK_ID_CURR').median().drop('SK_ID_BUREAU',axis=1)
bureau_final_median['cnt_id_bureau']=cnt_id_bureau
bureau_final_median.fillna(0,inplace=True)
bureau_final_median.head()
# clean up. Free RAM space
del bureau_merge,bureau_onehot,bureau_balance_unstack
gc.collect()
basic_info(previous_application)
categorical_features(previous_application)
check_missing_data(previous_application).head(10)
previous_application.drop(['RATE_INTEREST_PRIVILEGED','RATE_INTEREST_PRIMARY'],axis=1,inplace=True)
previous_application.AMT_CREDIT.fillna(previous_application.AMT_CREDIT.median(),inplace=True)
previous_application.CHANNEL_TYPE.value_counts()
previous_application.drop(['PRODUCT_COMBINATION','NAME_TYPE_SUITE',],axis=1,inplace=True)
previous_application.RATE_DOWN_PAYMENT.plot()
previous_application.RATE_DOWN_PAYMENT.describe()
previous_application.RATE_DOWN_PAYMENT.fillna(previous_application.RATE_DOWN_PAYMENT.median(),inplace=True)
previous_application.AMT_DOWN_PAYMENT.describe()
previous_application.AMT_DOWN_PAYMENT.plot()
previous_application.AMT_DOWN_PAYMENT.fillna(0.0,inplace=True)
previous_application.NFLAG_INSURED_ON_APPROVAL.fillna(0,inplace=True)
previous_application.AMT_GOODS_PRICE.plot()
previous_application.AMT_GOODS_PRICE.describe()
previous_application.AMT_GOODS_PRICE.fillna(previous_application.AMT_GOODS_PRICE.mean(),inplace=True)
previous_application.AMT_ANNUITY.plot()
previous_application.AMT_ANNUITY.describe()
previous_application.AMT_ANNUITY.fillna(previous_application.AMT_ANNUITY.mean(),inplace=True)
previous_application.CNT_PAYMENT.describe()
previous_application.CNT_PAYMENT.fillna(previous_application.CNT_PAYMENT.median(),inplace=True)
previous_application.head()
previous_application_onehot=onehot_encoding(previous_application,categorical_features(previous_application))
cnt_id_prev1=previous_application_onehot[['SK_ID_CURR','SK_ID_PREV']].groupby('SK_ID_CURR').size()
previous_application_mean=previous_application_onehot.groupby('SK_ID_CURR').mean().drop('SK_ID_PREV',axis=1)
previous_application_mean['cnt_id_prev1']=cnt_id_prev1
previous_application_mean.fillna(0,inplace=True)
previous_application_mean.head()
previous_application_min=previous_application_onehot.groupby('SK_ID_CURR').min().drop('SK_ID_PREV',axis=1)

previous_application_max=previous_application_onehot.groupby('SK_ID_CURR').max().drop('SK_ID_PREV',axis=1)

previous_application_median=previous_application_onehot.groupby('SK_ID_CURR').median().drop('SK_ID_PREV',axis=1)

previous_application_merge=previous_application_mean.merge(previous_application_min,on='SK_ID_CURR').merge(previous_application_max,on='SK_ID_CURR').merge(previous_application_median,on='SK_ID_CURR')
previous_application_merge['cnt_id_prev1']=cnt_id_prev1
previous_application_merge.fillna(0,inplace=True)
previous_application_merge.head()
# clean up. Free RAM space
del previous_application,previous_application_max,previous_application_mean,previous_application_min,previous_application_onehot

gc.collect()
basic_info(POS_CASH_balance)
POS_CASH_balance.NAME_CONTRACT_STATUS.value_counts()
check_missing_data(POS_CASH_balance)
POS_CASH_balance.CNT_INSTALMENT_FUTURE.describe()
POS_CASH_balance.CNT_INSTALMENT_FUTURE.plot(kind='box')
POS_CASH_balance.CNT_INSTALMENT_FUTURE.fillna(POS_CASH_balance.CNT_INSTALMENT_FUTURE.median(),inplace=True)
POS_CASH_balance.CNT_INSTALMENT.plot(kind='box')
POS_CASH_balance.CNT_INSTALMENT.describe()
POS_CASH_balance.drop('CNT_INSTALMENT',axis=1,inplace=True)
POS_CASH_balance_onehot=onehot_encoding(POS_CASH_balance,categorical_features(POS_CASH_balance))
POS_CASH_balance_onehot.head()
cnt_id_prev2=POS_CASH_balance_onehot[['SK_ID_CURR','SK_ID_PREV']].groupby('SK_ID_CURR').size()
POS_CASH_balance_median=POS_CASH_balance_onehot.groupby('SK_ID_CURR').median().drop('SK_ID_PREV',axis=1)
POS_CASH_balance_median['cnt_id_prev2']=cnt_id_prev2
POS_CASH_balance_median.fillna(0,inplace=True)
POS_CASH_balance_median.head()
# clean up. Free RAM space
del POS_CASH_balance,POS_CASH_balance_onehot
gc.collect()
basic_info(credit_card_balance)
check_missing_data(credit_card_balance).head()
categorical_features(credit_card_balance)
credit_card_balance.NAME_CONTRACT_STATUS.value_counts()
credit_card_balance_onehot=onehot_encoding(credit_card_balance,categorical_features(credit_card_balance))
credit_card_balance_onehot.fillna(credit_card_balance_onehot.median(),inplace=True)
credit_card_balance.head()
cnt_id_prev3=credit_card_balance_onehot[['SK_ID_CURR','SK_ID_PREV']].groupby('SK_ID_CURR').size()
credit_card_balance_median=credit_card_balance_onehot.groupby('SK_ID_CURR').median().drop('SK_ID_PREV',axis=1)
credit_card_balance_median['cnt_id_prev3']=cnt_id_prev3
credit_card_balance_median.fillna(0,inplace=True)
credit_card_balance_median.head()
# clean up. Free RAM space
del credit_card_balance,credit_card_balance_onehot
gc.collect()
basic_info(installments_payments)
check_missing_data(installments_payments)
categorical_features(installments_payments)
installments_payments.dropna(inplace=True)
cnt_id_prev4=installments_payments[['SK_ID_CURR','SK_ID_PREV']].groupby('SK_ID_CURR').size()
installments_payments_min=installments_payments.groupby('SK_ID_CURR').min().drop('SK_ID_PREV',axis=1)
installments_payments_max=installments_payments.groupby('SK_ID_CURR').max().drop('SK_ID_PREV',axis=1)
installments_payments_median=installments_payments.groupby('SK_ID_CURR').median().drop('SK_ID_PREV',axis=1)
installments_payments_merge=installments_payments_min.merge(installments_payments_max,on='SK_ID_CURR').merge(installments_payments_median,on='SK_ID_CURR')
installments_payments_merge['cnt_id_prev4']=cnt_id_prev4
installments_payments_merge.fillna(0,inplace=True)
installments_payments_merge.head()
# clean up. Free RAM space
del installments_payments,installments_payments_max,installments_payments_min
gc.collect()
application_train.head()
application_test.tail()
target=application_train['TARGET']
application_train.drop('TARGET',axis=1,inplace=True)
application_train['TARGET']=target
application_train.head()
application_test['TARGET']=-999
df=pd.concat([application_train,application_test])
check_missing_data(df).head()
categorical_features(df)
df_onehot=onehot_encoding(df,categorical_features(df))
df_onehot.shape
df_onehot.fillna(0,inplace=True)
check_missing_data(df_onehot).head()
# clean up. Free RAM space
del application_test,application_train,df
gc.collect()
total=df_onehot.merge(right=bureau_final_median,on='SK_ID_CURR',how='left').merge(right=previous_application_median,on='SK_ID_CURR',how='left').merge(right=POS_CASH_balance_median,on='SK_ID_CURR',how='left').merge(right=credit_card_balance_median,on='SK_ID_CURR',how='left').merge(right=installments_payments_merge,on='SK_ID_CURR',how='left')

total.shape
df_total=total.fillna(0)
df_total.head()
# clean up. Free RAM space
del total,df_onehot,bureau_final_median,previous_application_merge,previous_application_median
del POS_CASH_balance_median,credit_card_balance_median,installments_payments_median,installments_payments_merge
gc.collect()
df_train=df_total[df_total.TARGET!=-999]
# print(df_train.shape)
# print(application_train.shape)
# df_train.head()
df_test=df_total[df_total.TARGET==-999]
# print(df_test.shape)
# print(application_test.shape)
# df_test.head()
test=df_test.drop(columns=["SK_ID_CURR",'TARGET'],axis=1)
test.shape
y=df_train['TARGET'].values
y
train=df_train.drop(columns=["SK_ID_CURR",'TARGET'],axis=1).values
train.shape
# clean up. Free RAM space
del df_train,df_test,df_total
gc.collect()
gc.collect()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(train,y,test_size=0.2)
# clean up. Free RAM space
del train
gc.collect()
import lightgbm
train_data=lightgbm.Dataset(X_train,label=y_train)
valid_data=lightgbm.Dataset(X_test,label=y_test)

params = {'boosting_type': 'gbdt',
          'max_depth' : 10,
          'objective': 'binary',
          'nthread': 5,
          'num_leaves': 64,
          'learning_rate': 0.1,
          'max_bin': 512,
          'subsample_for_bin': 200,
          'subsample': 1,
          'subsample_freq': 1,
          'colsample_bytree': 0.8,
          'reg_alpha': 5,
          'reg_lambda': 10,
          'min_split_gain': 0.005,
          'min_child_weight': 1,
          'min_child_samples': 5,
          'scale_pos_weight': 1,
          'num_class' : 1,
          'metric' : 'auc'
          }
lgbm = lightgbm.train(params,
                 train_data,
                 25000,
                 valid_sets=valid_data,
                 early_stopping_rounds= 80,
                 verbose_eval= 10
                 )
#Predict on test set and write to submit
predictions_lgbm_prob = lgbm.predict(test.values)
sub=pd.read_csv('../input/sample_submission.csv')
sub.TARGET=predictions_lgbm_prob
sub.to_csv('sub.csv',index=False)

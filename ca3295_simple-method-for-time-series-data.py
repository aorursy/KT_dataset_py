# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd

import seaborn as sns
import lightgbm as lgb
ins = pd.read_csv('../input/installments_payments.csv')
ins.head()
ins['tw1']=np.exp(ins['DAYS_ENTRY_PAYMENT']*0.01)
ins['tw2']=np.exp(ins['DAYS_ENTRY_PAYMENT']*0.05)
ins['tw3']=np.exp(ins['DAYS_ENTRY_PAYMENT']*0.25)
ts_view=ins[['DAYS_ENTRY_PAYMENT', 'tw1', 'tw2', 'tw3']][ins['DAYS_ENTRY_PAYMENT']>-100]
ts_view.drop_duplicates('DAYS_ENTRY_PAYMENT', inplace=True)
sns.regplot('DAYS_ENTRY_PAYMENT', 'tw1', data=ts_view, fit_reg=False)
sns.regplot('DAYS_ENTRY_PAYMENT', 'tw2', data=ts_view, fit_reg=False)
sns.regplot('DAYS_ENTRY_PAYMENT', 'tw3', data=ts_view, fit_reg=False)
#Are the payments late/early?

ins['DPD'] = ins['DAYS_ENTRY_PAYMENT'] - ins['DAYS_INSTALMENT']
ins['DBD'] = ins['DAYS_INSTALMENT'] - ins['DAYS_ENTRY_PAYMENT']
ins['DPD'] = ins['DPD'].apply(lambda x: x if x > 0 else 0)
ins['DBD'] = ins['DBD'].apply(lambda x: x if x > 0 else 0)

#Now let's weight these new values

ins['DPD_tw1']=ins['DPD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw1'])
ins['DPD_tw2']=ins['DPD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw2'])
ins['DPD_tw3']=ins['DPD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw3'])

ins['DBD_tw1']=ins['DBD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw1'])
ins['DBD_tw2']=ins['DBD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw2'])
ins['DBD_tw3']=ins['DBD']*np.exp(ins['DAYS_ENTRY_PAYMENT']*ins['tw3'])

#Let's weight the value of payments made

ins['AMT_PAYMENT_tw1']=ins['AMT_PAYMENT']*ins['tw1']
ins['AMT_PAYMENT_tw2']=ins['AMT_PAYMENT']*ins['tw2']
ins['AMT_PAYMENT_tw3']=ins['AMT_PAYMENT']*ins['tw3']





# Features: Perform aggregations
# The natural thing to want to do here is to sum the time-weights and weighted values so that we can get weighted-averages
# You might want to experiment with this a bit more - perhaps other aggregations will prove useful?
aggregations = {
    'NUM_INSTALMENT_VERSION': ['nunique'],
    'DPD': ['max', 'mean', 'sum'],
    'DBD': ['max', 'mean', 'sum'],
    'AMT_INSTALMENT': ['max', 'mean', 'sum'],
    'AMT_PAYMENT': ['min', 'max', 'mean', 'sum'],
    'DAYS_ENTRY_PAYMENT': ['max', 'mean', 'sum'],
    'DPD_tw1':['sum'],
    'DPD_tw2':['sum'],
    'DPD_tw3':['sum'],
    'DBD_tw1':['sum'],
    'DBD_tw2':['sum'],
    'DBD_tw3':['sum'],
    'tw1':['sum'],
    'tw2':['sum'],
    'tw3':['sum'],


    'AMT_PAYMENT_tw1':['sum'],
    
    'AMT_PAYMENT_tw2':['sum'],
    'AMT_PAYMENT_tw3':['sum']
    
}

    
ins_agg = ins.groupby('SK_ID_CURR').agg(aggregations)
ins_agg.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper() for e in ins_agg.columns.tolist()])



ins_agg['INSTAL_weighted_tw1_DPD_SUM']=ins_agg['INSTAL_DPD_tw1_SUM']/ins_agg['INSTAL_tw1_SUM']
ins_agg['INSTAL_weighted_tw2_DPD_SUM']=ins_agg['INSTAL_DPD_tw2_SUM']/ins_agg['INSTAL_tw2_SUM']
ins_agg['INSTAL_weighted_tw3_DPD_SUM']=ins_agg['INSTAL_DPD_tw3_SUM']/ins_agg['INSTAL_tw3_SUM']

ins_agg['INSTAL_weighted_tw1_DBD_SUM']=ins_agg['INSTAL_DBD_tw1_SUM']/ins_agg['INSTAL_tw1_SUM']
ins_agg['INSTAL_weighted_tw2_DBD_SUM']=ins_agg['INSTAL_DBD_tw2_SUM']/ins_agg['INSTAL_tw2_SUM']
ins_agg['INSTAL_weighted_tw3_DBD_SUM']=ins_agg['INSTAL_DBD_tw3_SUM']/ins_agg['INSTAL_tw3_SUM']

ins_agg['INSTAL_weighted_tw1_AMTPAY_SUM']=ins_agg['INSTAL_AMT_PAYMENT_tw1_SUM']/ins_agg['INSTAL_tw1_SUM']
ins_agg['INSTAL_weighted_tw2_AMTPAY_SUM']=ins_agg['INSTAL_AMT_PAYMENT_tw2_SUM']/ins_agg['INSTAL_tw2_SUM']
ins_agg['INSTAL_weighted_tw3_AMTPAY_SUM']=ins_agg['INSTAL_AMT_PAYMENT_tw3_SUM']/ins_agg['INSTAL_tw3_SUM']
ins_agg.head()
targets=pd.read_csv('../input/application_train.csv')[['SK_ID_CURR', 'TARGET']]

ins_agg=ins_agg.join(targets.set_index('SK_ID_CURR'))

train=ins_agg[pd.notnull(ins_agg['TARGET'])]
train=train.fillna(0)
target = np.array(train['TARGET'])
train=train.drop(['TARGET'],axis=1)
estimator=lgb.LGBMClassifier()

estimator.fit(train, target)
import shap

shap_values = shap.TreeExplainer(estimator).shap_values(train[0:2000])
shap.summary_plot(shap_values, train[0:2000], max_display=800)
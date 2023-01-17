# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nmpy_df as nmd

import missingno

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
bureau = pd.read_csv('../input/home-credit-default-risk/bureau.csv',sep=',')

bureau_balance = pd.read_csv('../input/home-credit-default-risk/bureau_balance.csv',sep=',')

credit_card_balance = pd.read_csv('../input/home-credit-default-risk/credit_card_balance.csv',sep=',')

previous_application = pd.read_csv('../input/home-credit-default-risk/previous_application.csv',sep=',')

#installments_payments = pd.read_csv('../input/home-credit-default-risk/installments_payments.csv',sep=',')

pos_cash_balance = pd.read_csv('../input/home-credit-default-risk/POS_CASH_balance.csv',sep=',')

application_train = pd.read_csv('../input/home-credit-default-risk/application_train.csv',sep=',')
nmd.df_val_nan(previous_application[['NAME_CONTRACT_TYPE','NAME_CONTRACT_STATUS','DAYS_TERMINATION']])
nmd.df_val_nan(pos_cash_balance[['NAME_CONTRACT_STATUS','SK_DPD_DEF','CNT_INSTALMENT_FUTURE']])
nmd.df_val_nan(credit_card_balance[['NAME_CONTRACT_STATUS','SK_DPD_DEF']])
nmd.df_val_nan(bureau[['CREDIT_TYPE','CREDIT_ACTIVE','CREDIT_DAY_OVERDUE','AMT_CREDIT_SUM_OVERDUE','DAYS_CREDIT_ENDDATE']])
nmd.df_val_nan(bureau_balance[['STATUS']])
previous_application.NAME_CONTRACT_TYPE.value_counts()
previous_application.NAME_CONTRACT_STATUS.value_counts()
previous_application.DAYS_TERMINATION.value_counts().sort_index()
#installments_payments.AMT_INSTALMENT.value_counts()
#installments_payments.AMT_PAYMENT.value_counts()
pos_cash_balance.MONTHS_BALANCE.value_counts()
pos_cash_balance.NAME_CONTRACT_STATUS.value_counts()
pos_cash_balance.SK_DPD_DEF.value_counts()
credit_card_balance.MONTHS_BALANCE.value_counts()
credit_card_balance.NAME_CONTRACT_STATUS.value_counts()
credit_card_balance.SK_DPD_DEF.value_counts()
bureau.CREDIT_ACTIVE.value_counts()
bureau.CREDIT_DAY_OVERDUE.value_counts()
bureau_balance.MONTHS_BALANCE.value_counts()
bureau_balance.STATUS.value_counts()
previous_application.groupby(['SK_ID_CURR','NAME_CONTRACT_STATUS']).count().head(20)
X='SK_ID_CURR'

Y='NAME_CONTRACT_TYPE'

table___name_contract_type = previous_application[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,fill_value=0,margins=True,margins_name='TOTAL')

table___name_contract_type.div(table___name_contract_type.iloc[:,-1], axis='index').applymap('{:.2f}'.format)
X='SK_ID_CURR'

Y='NAME_CONTRACT_STATUS'

table___name_contract_status = previous_application[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,fill_value=0,margins=True,margins_name='TOTAL')

table___name_contract_status.div(table___name_contract_status.iloc[:,-1], axis='index').applymap('{:.2f}'.format)
previous_application.groupby(['SK_ID_CURR','NAME_CONTRACT_TYPE'])['NAME_CONTRACT_STATUS'].value_counts()
X='SK_ID_CURR'

Y='NAME_CONTRACT_STATUS'

table___name_contract_status_pos = pos_cash_balance[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,fill_value=0,margins=True,margins_name='TOTAL')

#table___name_contract_status_pos = table___name_contract_status_pos.div(table___name_contract_status_pos.iloc[:,-1], axis='index').applymap('{:.2f}'.format)

table___name_contract_status_pos
X='SK_ID_CURR'

Y='NAME_CONTRACT_STATUS'

table___name_contract_status_cc = credit_card_balance[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,fill_value=0,margins=True,margins_name='TOTAL')

#table___name_contract_status_cc = table___name_contract_status_cc.div(table___name_contract_status_cc.iloc[:,-1], axis='index').applymap('{:.2f}'.format)

table___name_contract_status_cc
X='SK_ID_CURR'

Y='CREDIT_ACTIVE'

table___credit_active = bureau[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,fill_value=0,margins=True,margins_name='TOTAL')

table___credit_active = table___credit_active.div(table___credit_active.iloc[:,-1], axis='index').applymap('{:.2f}'.format)

table___credit_active
w1_hc_extra_features = pd.DataFrame()

w1_hc_extra_features
w1_hc_extra_features.columns = w1_hc_extra_features.columns.map(' '.join).str.strip()

w1_hc_extra_features
X = 'SK_ID_CURR'

Y = 'NAME_CONTRACT_TYPE'

Z = 'NAME_CONTRACT_STATUS'

table___hc_previous_credit_decision_cnt = previous_application[[X,Y,Z]].pivot_table(columns=[Y,Z], index=X,aggfunc=len,fill_value=0,margins=True,margins_name='TOTAL')

table___hc_previous_credit_decision_cnt.columns = pd.MultiIndex.from_tuples([(f'hcpc_c_{a}',b) for a,b in table___hc_previous_credit_decision_cnt.columns])

table___hc_previous_credit_decision_cnt
w1_hc_extra_features = pd.concat([w1_hc_extra_features,table___hc_previous_credit_decision_cnt.iloc[:-1,:-1]], axis=1, join='outer', sort=False)

w1_hc_extra_features.columns = w1_hc_extra_features.columns.map(' '.join).str.strip()

w1_hc_extra_features
X='SK_ID_CURR'

Y='NAME_CONTRACT_STATUS'

table___hc_previous_credit_decision_ratio = previous_application[[X,Y]].pivot_table(index=X,columns=Y,aggfunc=len,fill_value=0,margins=True,margins_name='TOTAL')

table___hc_previous_credit_decision_ratio = table___hc_previous_credit_decision_ratio.div(table___hc_previous_credit_decision_ratio.iloc[:,-1], axis='index').applymap('{:.2f}'.format)

table___hc_previous_credit_decision_ratio.columns = 'hcpc_r_'+table___hc_previous_credit_decision_ratio.columns

table___hc_previous_credit_decision_ratio
w1_hc_extra_features = pd.concat([w1_hc_extra_features,table___hc_previous_credit_decision_ratio.iloc[:-1,:-1]], axis=1, join='outer', sort=False)

w1_hc_extra_features
X = 'SK_ID_CURR'

Y = 'NAME_CONTRACT_STATUS'

Z = ''

table___hc_previous_pos_credit_status = pos_cash_balance[[X,Y]].pivot_table(columns=[Y], index=X,aggfunc=len,fill_value=0,margins=True,margins_name='TOTAL')

table___hc_previous_pos_credit_status.columns = 'hcpb_c_'+table___hc_previous_pos_credit_status.columns

table___hc_previous_pos_credit_status
w1_hc_extra_features = pd.concat([w1_hc_extra_features,table___hc_previous_pos_credit_status.iloc[:-1,:-1]], axis=1, join='outer', sort=False)

w1_hc_extra_features
X = 'SK_ID_CURR'

Y = 'NAME_CONTRACT_STATUS'

Z = ''

table___hc_previous_cc_credit_status = credit_card_balance[[X,Y]].pivot_table(columns=[Y], index=X,aggfunc=len,fill_value=0,margins=True,margins_name='TOTAL')

table___hc_previous_cc_credit_status.columns = 'hccb_c_'+table___hc_previous_cc_credit_status.columns

table___hc_previous_cc_credit_status
w1_hc_extra_features = pd.concat([w1_hc_extra_features,table___hc_previous_cc_credit_status.iloc[:-1,:-1]], axis=1, join='outer', sort=False)

w1_hc_extra_features
# Attention au Nan convertis en 1 > bonne solution ?

table___hc_previous_credit_status_temp = previous_application[['SK_ID_CURR','DAYS_TERMINATION']].copy()

table___hc_previous_credit_status_temp['DAYS_TERMINATION'] = table___hc_previous_credit_status_temp['DAYS_TERMINATION'].apply(lambda x: 0 if x<=0 else 1)

table___hc_previous_credit_status_active_cnt = table___hc_previous_credit_status_temp.groupby(['SK_ID_CURR'])['DAYS_TERMINATION'].sum()

table___hc_previous_credit_status_active_cnt
w1_hc_extra_features['hcpc_c_Active_credit'] = table___hc_previous_credit_status_active_cnt

w1_hc_extra_features
table___hc_previous_pos_credit_enddate_temp = pos_cash_balance[['SK_ID_CURR','SK_ID_PREV','CNT_INSTALMENT_FUTURE']].copy()

table___hc_previous_pos_credit_enddate_temp = table___hc_previous_pos_credit_enddate_temp.groupby(['SK_ID_CURR','SK_ID_PREV']).min()*30

table___hc_previous_pos_credit_enddate_sum = table___hc_previous_pos_credit_enddate_temp.groupby(['SK_ID_CURR']).sum()

table___hc_previous_pos_credit_enddate_sum
(table___hc_previous_pos_credit_enddate_sum.isna() == True).sum()
w1_hc_extra_features['hcpb_s_Remaining_days'] = table___hc_previous_pos_credit_enddate_sum

w1_hc_extra_features
table___hc_previous_pos_credit_delay_temp = pos_cash_balance[['SK_ID_CURR','SK_ID_PREV','SK_DPD_DEF']].copy()

table___hc_previous_pos_credit_delay_temp['SK_DPD_DEF'] = table___hc_previous_pos_credit_delay_temp['SK_DPD_DEF'].apply(lambda x: 0 if x==0 else (1 if 1 <= x <= 30 else(2 if 31 <= x <= 60 else(3 if 61 <= x <= 90 else(4 if 91 <= x <= 120 else 5)))))

table___hc_previous_pos_credit_delay_mean_pre = table___hc_previous_pos_credit_delay_temp.groupby(['SK_ID_CURR','SK_ID_PREV']).mean().applymap('{:.3f}'.format).astype('float64')

table___hc_previous_pos_credit_delay_mean_pre

table___hc_previous_pos_credit_delay_mean = table___hc_previous_pos_credit_delay_mean_pre.groupby(['SK_ID_CURR']).mean().applymap('{:.3f}'.format)

table___hc_previous_pos_credit_delay_mean
w1_hc_extra_features['hcpb_m_DPD_DEFcat'] = table___hc_previous_pos_credit_delay_mean

w1_hc_extra_features
table___hc_previous_pos_credit_delay_temp2 = table___hc_previous_pos_credit_delay_temp.copy()

table___hc_previous_pos_credit_delay_temp2['SK_DPD_DEF'] = table___hc_previous_pos_credit_delay_temp2['SK_DPD_DEF'].apply(lambda x: 1 if x==5 else 0)

table___hc_previous_pos_credit_fatal_flag_pre = table___hc_previous_pos_credit_delay_temp2.groupby(['SK_ID_CURR','SK_ID_PREV']).max()

table___hc_previous_pos_credit_fatal_flag_pre

table___hc_previous_pos_credit_fatal_flag_cnt = table___hc_previous_pos_credit_fatal_flag_pre.groupby(['SK_ID_CURR']).count()

table___hc_previous_pos_credit_fatal_flag_cnt
w1_hc_extra_features['hcpb_c_Default'] = table___hc_previous_pos_credit_fatal_flag_cnt

w1_hc_extra_features
table___hc_previous_cc_credit_delay_temp = credit_card_balance[['SK_ID_CURR','SK_ID_PREV','SK_DPD_DEF']].copy()

table___hc_previous_cc_credit_delay_temp['SK_DPD_DEF'] = table___hc_previous_cc_credit_delay_temp['SK_DPD_DEF'].apply(lambda x: 0 if x==0 else (1 if 1 <= x <= 30 else(2 if 31 <= x <= 60 else(3 if 61 <= x <= 90 else(4 if 91 <= x <= 120 else 5)))))

table___hc_previous_cc_credit_delay_mean_pre = table___hc_previous_cc_credit_delay_temp.groupby(['SK_ID_CURR','SK_ID_PREV']).mean().applymap('{:.3f}'.format).astype('float64')

table___hc_previous_cc_credit_delay_mean_pre

table___hc_previous_cc_credit_delay_mean = table___hc_previous_cc_credit_delay_mean_pre.groupby(['SK_ID_CURR']).mean().applymap('{:.3f}'.format)

table___hc_previous_cc_credit_delay_mean
w1_hc_extra_features['hccb_m_DPD_DEFcat'] = table___hc_previous_cc_credit_delay_mean

w1_hc_extra_features
w1_bu_extra_features = pd.DataFrame()

w1_bu_extra_features
w1_bub_extra_features = pd.DataFrame()

w1_bub_extra_features
table___bur_previous_credit_delay_temp = bureau_balance.copy()

table___bur_previous_credit_delay_temp = table___bur_previous_credit_delay_temp.replace(['C','X'],'0')

table___bur_previous_credit_delay_temp.STATUS = table___bur_previous_credit_delay_temp.STATUS.astype('int64')

table___bur_previous_credit_delay_mean = table___bur_previous_credit_delay_temp.groupby(['SK_ID_BUREAU']).mean().applymap('{:.3f}'.format)['STATUS']

table___bur_previous_credit_delay_mean
w1_bub_extra_features['bub_m_DPD_DEFcat'] = table___bur_previous_credit_delay_mean

w1_bub_extra_features
table___bur_previous_credit_delay_temp2 = table___bur_previous_credit_delay_temp.copy()

table___bur_previous_credit_delay_temp2['STATUS'] = table___bur_previous_credit_delay_temp2['STATUS'].apply(lambda x: 1 if x==5 else 0)

table___bur_previous_credit_fatal_flag = table___bur_previous_credit_delay_temp2.groupby(['SK_ID_BUREAU'])['STATUS'].max()

table___bur_previous_credit_fatal_flag
w1_bub_extra_features['bub_f_Default'] = table___bur_previous_credit_fatal_flag

w1_bub_extra_features['bub_m_DPD_DEFcat'] = w1_bub_extra_features['bub_m_DPD_DEFcat'].astype('float64')

w1_bub_extra_features['bub_f_Default'] = w1_bub_extra_features['bub_f_Default'].astype('int64')

w1_bub_extra_features
W = 'SK_ID_CURR'

X = 'SK_ID_BUREAU'

Y = 'CREDIT_TYPE'

Z = 'CREDIT_ACTIVE'

table___bur_previous_credit_status = bureau[[W,X,Y,Z]].pivot_table(columns=[Y,Z], index=[W,X],aggfunc=len,fill_value=0,margins=True,margins_name='TOTAL')

table___bur_previous_credit_status.columns = pd.MultiIndex.from_tuples([(f'bu_c_{a}',b) for a,b in table___bur_previous_credit_status.columns])

table___bur_previous_credit_status
w1_bu_extra_features = pd.concat([w1_bu_extra_features,table___bur_previous_credit_status.iloc[:-1,:-1]], axis=1, join='outer', sort=False)

w1_bu_extra_features
table___bur_previous_credit_enddate_temp = bureau[['SK_ID_CURR','SK_ID_BUREAU','DAYS_CREDIT_ENDDATE']].copy()

table___bur_previous_credit_enddate_temp['DAYS_CREDIT_ENDDATE'] = table___bur_previous_credit_enddate_temp['DAYS_CREDIT_ENDDATE'].apply(lambda x: 0 if x<=0 else x)

table___bur_previous_credit_enddate_sum = table___bur_previous_credit_enddate_temp.groupby(['SK_ID_CURR','SK_ID_BUREAU']).sum().applymap('{:.0f}'.format)

table___bur_previous_credit_enddate_sum
(table___bur_previous_credit_enddate_sum['DAYS_CREDIT_ENDDATE'].isna() == True).sum()
w1_bu_extra_features['bu_s_Remaining_days'] = table___bur_previous_credit_enddate_sum.astype('int64')

w1_bu_extra_features
table___bur_previous_credit_amt_overdue_temp = bureau[['SK_ID_CURR','SK_ID_BUREAU','AMT_CREDIT_SUM_OVERDUE']].copy()

table___bur_previous_credit_amt_overdue_sum = table___bur_previous_credit_amt_overdue_temp.groupby(['SK_ID_CURR','SK_ID_BUREAU']).sum().applymap('{:.0f}'.format)

table___bur_previous_credit_amt_overdue_sum
w1_bu_extra_features['bu_s_Amt_overdue'] = table___bur_previous_credit_amt_overdue_sum.astype('int64')

w1_bu_extra_features
w1_bu_extra_features.columns = w1_bu_extra_features.columns.map(' '.join).str.strip()

w1_bu_extra_features
w1_bu_extra_features = w1_bu_extra_features.join(w1_bub_extra_features, how='outer')

w1_bu_extra_features
w1_bu_extra_features.xs(6842885, level=1, drop_level=False)
columntosum = w1_bu_extra_features.columns[:-2].tolist()

columntosum
w1_bu_extra_features = w1_bu_extra_features.groupby(['SK_ID_CURR']).agg({

                                                 'bu_c_Another type of loan Active': np.sum,

                                                 'bu_c_Another type of loan Closed': np.sum,

                                                 'bu_c_Another type of loan Sold': np.sum,

                                                 'bu_c_Car loan Active': np.sum,

                                                 'bu_c_Car loan Closed': np.sum,

                                                 'bu_c_Car loan Sold': np.sum,

                                                 'bu_c_Cash loan (non-earmarked) Active': np.sum,

                                                 'bu_c_Cash loan (non-earmarked) Closed': np.sum,

                                                 'bu_c_Cash loan (non-earmarked) Sold': np.sum,

                                                 'bu_c_Consumer credit Active': np.sum,

                                                 'bu_c_Consumer credit Bad debt': np.sum,

                                                 'bu_c_Consumer credit Closed': np.sum,

                                                 'bu_c_Consumer credit Sold': np.sum,

                                                 'bu_c_Credit card Active': np.sum,

                                                 'bu_c_Credit card Bad debt': np.sum,

                                                 'bu_c_Credit card Closed': np.sum,

                                                 'bu_c_Credit card Sold': np.sum,

                                                 'bu_c_Interbank credit Closed': np.sum,

                                                 'bu_c_Loan for business development Active': np.sum,

                                                 'bu_c_Loan for business development Closed': np.sum,

                                                 'bu_c_Loan for business development Sold': np.sum,

                                                 'bu_c_Loan for purchase of shares (margin lending) Active': np.sum,

                                                 'bu_c_Loan for purchase of shares (margin lending) Closed': np.sum,

                                                 'bu_c_Loan for the purchase of equipment Active': np.sum,

                                                 'bu_c_Loan for the purchase of equipment Closed': np.sum,

                                                 'bu_c_Loan for working capital replenishment Active': np.sum,

                                                 'bu_c_Loan for working capital replenishment Closed': np.sum,

                                                 'bu_c_Loan for working capital replenishment Sold': np.sum,

                                                 'bu_c_Microloan Active': np.sum,

                                                 'bu_c_Microloan Closed': np.sum,

                                                 'bu_c_Microloan Sold': np.sum,

                                                 'bu_c_Mobile operator loan Active': np.sum,

                                                 'bu_c_Mortgage Active': np.sum,

                                                 'bu_c_Mortgage Closed': np.sum,

                                                 'bu_c_Mortgage Sold': np.sum,

                                                 'bu_c_Real estate loan Active': np.sum,

                                                 'bu_c_Real estate loan Closed': np.sum,

                                                 'bu_c_Real estate loan Sold': np.sum,

                                                 'bu_c_Unknown type of loan Active': np.sum,

                                                 'bu_c_Unknown type of loan Closed': np.sum,

                                                 'bu_s_Remaining_days': np.sum,

                                                 'bu_s_Amt_overdue': np.sum,

                                                 'bub_m_DPD_DEFcat': np.mean,

                                                 'bub_f_Default': np.sum

                                                 })

w1_bu_extra_features
w1_hc_extra_features
w1_bu_extra_features
w1_extra_features = pd.concat([w1_hc_extra_features,w1_bu_extra_features], axis=1, join='outer', sort=False)

w1_extra_features
w1_extra_features.sort_index()
nmd.df_val_nan(w1_extra_features)
w1_extra_features.to_csv('../working/w1_extra_features.csv', sep=',', encoding='utf-8', index_label='SK_ID_CURR')
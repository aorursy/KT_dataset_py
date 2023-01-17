import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import gc

bureau_balance = pd.read_csv('../input/bureau_balance.csv')

# bb feature
bureau_balance['STATUS_mod'] = bureau_balance.STATUS.map({'0':0, '1':1, '2':2, '3':3, '4':4, '5':5, 'X':np.nan, 'C':0}).map(lambda x: 0 if x=='C' else x).interpolate(method = 'linear')
bureau_balance['write_off'] = bureau_balance.STATUS.map(lambda x: 1 if x=='5' else 0)
bureau_balance['adj_score'] = (bureau_balance.MONTHS_BALANCE-bureau_balance.MONTHS_BALANCE.min()+1)*bureau_balance.STATUS_mod

bb_month_count = bureau_balance.groupby('SK_ID_BUREAU')['MONTHS_BALANCE'].count()
bb_dpd_sum = bureau_balance.groupby('SK_ID_BUREAU')['STATUS_mod'].sum()
bb_write_off = bureau_balance.groupby('SK_ID_BUREAU')['write_off'].sum()
bb_dpd_sum_2_year = bureau_balance.loc[bureau_balance.MONTHS_BALANCE>=-24].groupby('SK_ID_BUREAU')['STATUS_mod'].sum()
bb_write_off_2_year = bureau_balance.loc[bureau_balance.MONTHS_BALANCE>=-24].groupby('SK_ID_BUREAU')['write_off'].sum()
bb_adj_score = bureau_balance.groupby('SK_ID_BUREAU')['adj_score'].sum()

bb_feature = pd.DataFrame({'bb_month_count':bb_month_count, 'bb_dpd_sum':bb_dpd_sum, 'bb_write_off':bb_write_off, 'bb_dpd_sum_2_year':bb_dpd_sum_2_year, 
                          'bb_write_off_2_year':bb_write_off_2_year, 'bb_adj_score': bb_adj_score}).reset_index().fillna(0)
del bb_month_count, bb_dpd_sum, bb_write_off, bb_dpd_sum_2_year, bb_write_off_2_year, bb_adj_score, bureau_balance
gc.collect()

# bureau
bureau = pd.read_csv('../input/bureau.csv')
bureau = bureau.sort_values(['SK_ID_CURR', 'DAYS_CREDIT'])

bureau['ADJ_DAYS'] = (bureau.DAYS_CREDIT - bureau.DAYS_CREDIT.min()) / (bureau.DAYS_CREDIT.max() - bureau.DAYS_CREDIT.min()) + 0.5 # more recent, more effecitve
# application count
bur_ncount = bureau.groupby('SK_ID_CURR')['SK_ID_BUREAU'].count()
bur_act_count = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['SK_ID_BUREAU'].count() # fillna: 0
bur_bad_count = bureau.loc[bureau.CREDIT_ACTIVE=='Bad debt'].groupby('SK_ID_CURR')['SK_ID_BUREAU'].count() # fillna: 0
bur_sold_count = bureau.loc[bureau.CREDIT_ACTIVE=='Sold out'].groupby('SK_ID_CURR')['SK_ID_BUREAU'].count() # fillna: 0
# application date
bur_recent_application = -bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].max()
bur_eariliest_application = -bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].min()
bur_max_enddate = -bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].max()
# application itervel
bureau['application_interval'] = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].diff(-1)
missing_iter = iter(bureau.groupby('SK_ID_CURR')['DAYS_CREDIT'].max())
bureau.application_interval = bureau.application_interval.map(lambda x: -next(missing_iter) if np.isnan(x) else -x)
bur_avg_intervel = bureau.groupby('SK_ID_CURR')['application_interval'].mean()
bur_sd_intervel = bureau.groupby('SK_ID_CURR')['application_interval'].agg('std').fillna(0)
# overdue days
bur_max_overdue_days = bureau.groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].max()
bur_active_total_overdue_days = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].sum()
bur_active_max_overdue_days = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['CREDIT_DAY_OVERDUE'].max()
bureau['DAYS_CREDIT_ISPAST'] = bureau.DAYS_CREDIT_ENDDATE.map(lambda x: 1 if x < 0 else 0)
bur_avg_remaining_days = bureau.loc[bureau.DAYS_CREDIT_ENDDATE>0].groupby('SK_ID_CURR')['DAYS_CREDIT_ENDDATE'].mean()
# overdue amount
bureau['ADJ_AMT_CREDIT_MAX_OVERDUE'] = bureau.ADJ_DAYS * bureau.AMT_CREDIT_MAX_OVERDUE
bur_total_max_overdue_adj = bureau.groupby('SK_ID_CURR')['ADJ_AMT_CREDIT_MAX_OVERDUE'].sum() # use adj days
bur_avg_max_overdue = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].mean()
bur_overall_max_overdue = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_MAX_OVERDUE'].max()
# adj prelong
bureau['ADJ_CNT_CREDIT_PROLONG'] = bureau.ADJ_DAYS * bureau.CNT_CREDIT_PROLONG
bur_avg_prelonged = bureau.groupby('SK_ID_CURR')['ADJ_CNT_CREDIT_PROLONG'].mean().fillna(0) # use adj days
bur_max_prelonged = bureau.groupby('SK_ID_CURR')['CNT_CREDIT_PROLONG'].max().fillna(0)
bur_total_prelonged_adj = bureau.groupby('SK_ID_CURR')['ADJ_CNT_CREDIT_PROLONG'].sum().fillna(0)
# historical amount
bureau['ADJ_AMT_CREDIT_SUM_DEBT'] = bureau.ADJ_DAYS * bureau.AMT_CREDIT_SUM
bur_total_amount_adj = bureau.groupby('SK_ID_CURR')['ADJ_AMT_CREDIT_SUM_DEBT'].sum()
bur_avg_amount = bureau.groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].mean()
# current amount
bur_active_total_amount = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].sum() # fillna 0
bur_active_avg_amount = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM'].mean() # fillna 0
bur_active_total_debt = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].sum() # fillna 0
bur_active_avg_debt = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_DEBT'].mean() # fillna 0
bur_active_total_limit  = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_LIMIT'].sum() # fillna 0
bur_active_avg_limit  = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_LIMIT'].mean() # fillna 0
bur_active_total_overdue = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].sum() # fillna 0
bur_active_avg_overdue = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_CREDIT_SUM_OVERDUE'].mean() # fillna 0
bur_active_ratio_debt_credit = (bur_active_total_debt / bur_active_total_amount.map(lambda x: x+0.1)) # fillna 0
bur_active_ratio_overdue_debt = (bur_active_total_overdue / bur_active_total_debt.map(lambda x: x+0.1)) # fillna 0
# credit update
bur_avg_update = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_UPDATE'].mean()
bur_recent_update = bureau.groupby('SK_ID_CURR')['DAYS_CREDIT_UPDATE'].max()
# annuity
bur_avg_annuity = bureau.groupby('SK_ID_CURR')['AMT_ANNUITY'].mean() # can't fillna 0
bur_total_annuity = bureau.groupby('SK_ID_CURR')['AMT_ANNUITY'].sum() # can't fillna 0
bur_active_total_annuity = bureau.loc[bureau.CREDIT_ACTIVE=='Active'].groupby('SK_ID_CURR')['AMT_ANNUITY'].sum()
bureau['term'] = bureau.AMT_CREDIT_SUM / bureau.AMT_ANNUITY
bur_avg_term = bureau.loc[bureau.term < float('inf')].groupby('SK_ID_CURR')['term'].mean() # can't fillna 0

bureau_num_feature = pd.DataFrame({'bur_ncount':bur_ncount, 'bur_act_count':bur_act_count, 'bur_bad_count':bur_bad_count, 'bur_sold_count':bur_sold_count,
            'bur_recent_application':bur_recent_application,'bur_eariliest_application':bur_eariliest_application, 'bur_max_enddate':bur_max_enddate,
            'bur_avg_intervel':bur_avg_intervel, 'bur_sd_intervel':bur_sd_intervel,
            'bur_max_overdue_days':bur_max_overdue_days, 'bur_active_total_overdue_days':bur_active_total_overdue_days, 'bur_active_max_overdue_days':bur_active_max_overdue_days,
            'bur_avg_remaining_days':bur_avg_remaining_days, 'bur_total_max_overdue_adj':bur_total_max_overdue_adj, 'bur_avg_max_overdue':bur_avg_max_overdue, 'bur_overall_max_overdue':bur_overall_max_overdue, 
            'bur_avg_prelonged':bur_avg_prelonged, 'bur_max_prelonged':bur_max_prelonged, 'bur_total_prelonged_adj':bur_total_prelonged_adj,
            'bur_total_amount_adj':bur_total_amount_adj, 'bur_avg_amount':bur_avg_amount,
            'bur_active_total_amount':bur_active_total_amount, 'bur_active_avg_amount':bur_active_avg_amount, 'bur_active_total_debt':bur_active_total_debt, 'bur_active_avg_debt':bur_active_avg_debt, 
            'bur_active_total_limit':bur_active_total_limit, 'bur_active_avg_limit':bur_active_avg_limit, 'bur_active_total_overdue':bur_active_total_overdue, 'bur_active_avg_overdue':bur_active_avg_overdue,
            'bur_active_ratio_debt_credit':bur_active_ratio_debt_credit, 'bur_active_ratio_overdue_debt':bur_active_ratio_overdue_debt,
            'bur_avg_update':bur_avg_update, 'bur_recent_update':bur_recent_update,
            'bur_avg_annuity':bur_avg_annuity, 'bur_total_annuity':bur_total_annuity, 'bur_active_total_annuity':bur_active_total_annuity, 'bur_avg_term':bur_avg_term}).reset_index()
fill0_list = ['bur_act_count', 'bur_bad_count', 'bur_sold_count', 'bur_active_total_overdue_days', 'bur_active_max_overdue_days', 'bur_active_total_amount', 'bur_active_avg_amount',
             'bur_active_total_debt', 'bur_active_avg_debt', 'bur_active_total_limit', 'bur_active_avg_limit', 'bur_active_total_overdue', 'bur_active_avg_overdue', 
              'bur_active_ratio_debt_credit', 'bur_active_ratio_overdue_debt', 'bur_active_total_annuity']
bureau_num_feature[fill0_list] = bureau_num_feature[fill0_list] .fillna(0)

bureau_cat = pd.get_dummies(bureau[['SK_ID_CURR','CREDIT_ACTIVE', 'CREDIT_CURRENCY', 'CREDIT_TYPE']], prefix='bur')
bureau_cat_feature = bureau_cat.groupby('SK_ID_CURR').mean().reset_index()
del bureau_cat
gc.collect()

bureau_bb = bureau[['SK_ID_CURR','SK_ID_BUREAU']].merge(bb_feature, on='SK_ID_BUREAU', how='left')

bb_avg_month = bureau_bb.groupby('SK_ID_CURR')['bb_month_count'].mean()
bb_total_overdue_month = bureau_bb.groupby('SK_ID_CURR')['bb_dpd_sum'].sum()
bb_total_writeoff = bureau_bb.groupby('SK_ID_CURR')['bb_write_off'].sum()
bb_max_overdue_month = bureau_bb.groupby('SK_ID_CURR')['bb_dpd_sum'].max()
bb_max_writeoff = bureau_bb.groupby('SK_ID_CURR')['bb_write_off'].max()

bb_total_overdue_month_2year = bureau_bb.groupby('SK_ID_CURR')['bb_dpd_sum_2_year'].sum()
bb_max_overdue_month_2year= bureau_bb.groupby('SK_ID_CURR')['bb_dpd_sum_2_year'].max()
bb_total_writeoff_2year = bureau_bb.groupby('SK_ID_CURR')['bb_write_off_2_year'].sum()
bb_max_writeoff_2year = bureau_bb.groupby('SK_ID_CURR')['bb_write_off_2_year'].max()

bb_max_score = bureau_bb.groupby('SK_ID_CURR')['bb_adj_score'].max()
bb_total_score = bureau_bb.groupby('SK_ID_CURR')['bb_adj_score'].sum()
bb_avg_score = bureau_bb.groupby('SK_ID_CURR')['bb_adj_score'].mean()

bureau_bb_feature = pd.DataFrame({'bb_avg_month':bb_avg_month, 
            'bb_total_overdue_month':bb_total_overdue_month, 'bb_total_writeoff':bb_total_writeoff, 'bur_sold_count':bb_max_overdue_month,'bb_max_writeoff':bb_max_writeoff,
            'bb_total_overdue_month_2year':bb_total_overdue_month_2year, 'bb_total_writeoff_2year':bb_total_writeoff_2year, 
            'bb_max_overdue_month_2year':bb_max_overdue_month_2year,'bb_max_writeoff_2year':bb_max_writeoff_2year, 
            'bb_max_score':bb_max_score, 'bb_total_score':bb_total_score, 'bb_avg_score':bb_avg_score}).reset_index()
			
bureau_feature = bureau_num_feature.merge(bureau_cat_feature, on='SK_ID_CURR').merge(bureau_bb_feature, on='SK_ID_CURR', how='left')
print(bureau_feature.shape)
bureau_feature.to_csv('bureau_feature.csv', index=False)
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import gc
import warnings
warnings.filterwarnings("ignore")

application_train = pd.read_csv('../input/application_train.csv')
credit_card_balance = pd.read_csv('../input/credit_card_balance.csv')
credit_card_balance = credit_card_balance.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

# count cards
ccb_prev_count = credit_card_balance.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
# INSTALMENTS
credit_card_balance['PAY_MONTH'] = credit_card_balance.CNT_INSTALMENT_MATURE_CUM.map(lambda x: 1 if x > 0 else 0)
ccb_temp = credit_card_balance.groupby(['SK_ID_CURR','SK_ID_PREV'])['PAY_MONTH'].sum().reset_index()
ccb_avg_inst_card = ccb_temp.groupby('SK_ID_CURR')['PAY_MONTH'].mean()
ccb_total_inst_card = ccb_temp.groupby('SK_ID_CURR')['PAY_MONTH'].sum()
# limit
ccb_temp = credit_card_balance.groupby(['SK_ID_CURR','SK_ID_PREV'])['AMT_CREDIT_LIMIT_ACTUAL'].mean().reset_index()
ccb_avg_limit_card = ccb_temp.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].mean()
ccb_max_limit_card = credit_card_balance.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].max()
ccb_total_limit_card = ccb_temp.groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].sum()
# avg drawing amount
ccb_temp = credit_card_balance.loc[credit_card_balance.CNT_DRAWINGS_CURRENT>0].groupby(['SK_ID_CURR','MONTHS_BALANCE'])['AMT_DRAWINGS_CURRENT', 'CNT_DRAWINGS_CURRENT'].sum().reset_index()
ccb_temp['avg_drawing_amount'] = (ccb_temp.AMT_DRAWINGS_CURRENT / ccb_temp.CNT_DRAWINGS_CURRENT).fillna(0)
ccb_avg_drawing_amount = ccb_temp.groupby('SK_ID_CURR')['avg_drawing_amount'].mean().fillna(0)
# count Refused
ccb_count_rej = credit_card_balance.groupby(['SK_ID_CURR'])['NAME_CONTRACT_STATUS'].agg(lambda x: np.sum(x=='Refused'))
last_month_credit = credit_card_balance.groupby(['SK_ID_CURR','SK_ID_PREV'])['MONTHS_BALANCE'].max().reset_index()
last_month_credit = last_month_credit.merge(credit_card_balance, on=['SK_ID_CURR','SK_ID_PREV', 'MONTHS_BALANCE'])
# current credit card situation
ccb_cur_total_receivable = last_month_credit.groupby('SK_ID_CURR')['AMT_TOTAL_RECEIVABLE'].sum()
ccb_cur_total_limit = last_month_credit.loc[last_month_credit.NAME_CONTRACT_STATUS == 'Active'].groupby('SK_ID_CURR')['AMT_CREDIT_LIMIT_ACTUAL'].sum() # fillna: 0
ccb_cur_total_payment = last_month_credit.groupby('SK_ID_CURR')['AMT_INST_MIN_REGULARITY'].sum()
ccb_cur_total_balance = last_month_credit.groupby('SK_ID_CURR')['AMT_BALANCE'].sum()
# drawing in 1y
ccb_temp = credit_card_balance.loc[credit_card_balance.MONTHS_BALANCE>=-12]
ccb_drawing_amount_1y = ccb_temp.groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].sum()
ccb_drawing_times_1y = ccb_temp.groupby('SK_ID_CURR')['CNT_DRAWINGS_CURRENT'].sum()
# drawing in 6m
ccb_temp = credit_card_balance.loc[credit_card_balance.MONTHS_BALANCE>=-6]
ccb_drawing_amount_6m = ccb_temp.groupby('SK_ID_CURR')['AMT_DRAWINGS_CURRENT'].sum()
ccb_drawing_times_6m = ccb_temp.groupby('SK_ID_CURR')['CNT_DRAWINGS_CURRENT'].sum()
# DPD
ccb_temp = credit_card_balance[['SK_ID_CURR', 'SK_ID_PREV', 'SK_DPD', 'SK_DPD_DEF']].groupby(['SK_ID_CURR','SK_ID_PREV'])['SK_DPD','SK_DPD_DEF'].max().reset_index()
ccb_max_dpd_days = ccb_temp.groupby('SK_ID_CURR')['SK_DPD'].max()
ccb_total_dpd_days = ccb_temp.groupby('SK_ID_CURR')['SK_DPD'].sum()
ccb_max_largedpd_days = ccb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
ccb_total_largedpd_days = ccb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].sum()

ccb_feature = pd.DataFrame({'ccb_prev_count':ccb_prev_count, 'ccb_avg_inst_card':ccb_avg_inst_card, 'ccb_avg_limit_card':ccb_avg_limit_card, 'ccb_total_inst_card':ccb_total_inst_card, 'ccb_count_rej': ccb_count_rej,
                        'ccb_avg_limit_card':ccb_avg_limit_card, 'ccb_max_limit_card':ccb_max_limit_card, 'ccb_total_limit_card':ccb_total_limit_card, 'ccb_avg_drawing_amount':ccb_avg_drawing_amount,
                        'ccb_cur_total_receivable':ccb_cur_total_receivable, 'ccb_cur_total_limit':ccb_cur_total_limit, 'ccb_cur_total_payment':ccb_cur_total_payment, 'ccb_cur_total_balance':ccb_cur_total_balance,
                        'ccb_drawing_amount_1y':ccb_drawing_amount_1y, 'ccb_drawing_times_1y':ccb_drawing_times_1y, 'ccb_drawing_amount_6m':ccb_drawing_amount_6m, 'ccb_drawing_times_6m':ccb_drawing_times_6m,
                        'ccb_max_dpd_days':ccb_max_dpd_days, 'ccb_total_dpd_days':ccb_total_dpd_days, 'ccb_max_largedpd_days':ccb_max_largedpd_days, 'ccb_total_largedpd_days':ccb_total_largedpd_days}).reset_index()
ccb_feature.to_csv('ccb_feature.csv', index=False)
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import gc
import warnings
warnings.filterwarnings("ignore")

installments_payments = pd.read_csv('../input/installments_payments.csv')
installments_payments = installments_payments.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'NUM_INSTALMENT_NUMBER'])
previous_application = pd.read_csv("../input/previous_application.csv")
previous_application = previous_application.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])

# recent application
recent_record = installments_payments.merge(installments_payments.groupby('SK_ID_CURR')['SK_ID_PREV'].max().reset_index().drop('SK_ID_CURR', axis=1), on='SK_ID_PREV')
ip_recent_term = recent_record.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].max()
ip_recent_total_actual_payment = recent_record.groupby('SK_ID_CURR')['AMT_PAYMENT'].sum()
ip_recent_total_required_payment = recent_record.groupby('SK_ID_CURR')['AMT_INSTALMENT'].sum()
# recent late & less time 
recent_record['is_late'] = (recent_record.DAYS_INSTALMENT < recent_record.DAYS_ENTRY_PAYMENT).map(lambda x: 1 if x==True else 0).fillna(0)
recent_record['is_less'] = (recent_record.AMT_INSTALMENT > recent_record.AMT_PAYMENT).map(lambda x: 1 if x==True else 0).fillna(0)
ip_recent_total_late_times = recent_record.groupby('SK_ID_CURR')['is_late'].sum()
ip_recent_total_less_times = recent_record.groupby('SK_ID_CURR')['is_less'].sum()
# recent late & less amount
ip_temp1 = recent_record.loc[recent_record.is_late==1]
ip_temp2 = recent_record.loc[recent_record.is_less==1]
ip_temp1['total_late'] = ip_temp1.DAYS_ENTRY_PAYMENT - ip_temp1.DAYS_INSTALMENT
ip_temp2['total_less'] = ip_temp2.AMT_INSTALMENT - ip_temp2.AMT_PAYMENT
ip_recent_total_late_days = ip_temp1.groupby('SK_ID_CURR')['total_late'].sum()
ip_recent_total_less_amount = ip_temp2.groupby('SK_ID_CURR')['total_less'].sum()
del ip_temp1, ip_temp2
gc.collect()

# previous application times
ip_prev_count = installments_payments.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
ip_payment_count = installments_payments.groupby('SK_ID_CURR')['SK_ID_PREV'].count()
# credit card
installments_payments['IS_CREDIT'] = installments_payments.NUM_INSTALMENT_VERSION.map(lambda x: 1 if x==0 else 0)
ip_creditcard_user = installments_payments.groupby('SK_ID_CURR')['IS_CREDIT'].sum().map(lambda x: 1 if x>0 else 0)
ip_creditcard_count = installments_payments.groupby(['SK_ID_CURR','SK_ID_PREV'])['IS_CREDIT'].sum().map(lambda x: 1 if x>0 else 0).reset_index().groupby('SK_ID_CURR')['IS_CREDIT'].sum()
# change times
ip_temp = (installments_payments.groupby(['SK_ID_CURR','SK_ID_PREV'])['NUM_INSTALMENT_VERSION'].nunique()-1).reset_index()
ip_total_change_times = ip_temp.groupby('SK_ID_CURR')['NUM_INSTALMENT_VERSION'].sum()
ip_avg_change_times = ip_temp.groupby('SK_ID_CURR')['NUM_INSTALMENT_VERSION'].mean()
# avg instl
ip_temp = installments_payments.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
ip_avg_instl = ip_temp.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].mean()
ip_max_instl = ip_temp.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].max()
# total late & less time 
installments_payments['is_late'] = (installments_payments.DAYS_INSTALMENT < installments_payments.DAYS_ENTRY_PAYMENT).map(lambda x: 1 if x==True else 0).fillna(0)
installments_payments['is_less'] = (installments_payments.AMT_INSTALMENT > installments_payments.AMT_PAYMENT).map(lambda x: 1 if x==True else 0).fillna(0)
ip_total_late_times = installments_payments.groupby('SK_ID_CURR')['is_late'].sum()
ip_total_less_times = installments_payments.groupby('SK_ID_CURR')['is_less'].sum()
# total late & less amount
ip_temp1 = installments_payments.loc[installments_payments.is_late==1]
ip_temp2 = installments_payments.loc[installments_payments.is_less==1]
ip_temp1['total_late'] = ip_temp1.DAYS_ENTRY_PAYMENT - ip_temp1.DAYS_INSTALMENT
ip_temp2['total_less'] = ip_temp2.AMT_INSTALMENT - ip_temp2.AMT_PAYMENT
ip_total_late_days = ip_temp1.groupby('SK_ID_CURR')['total_late'].sum()
ip_total_less_amount = ip_temp2.groupby('SK_ID_CURR')['total_less'].sum()
del ip_temp1, ip_temp2
gc.collect()
# total payment
ip_total_actual_payment = installments_payments.groupby('SK_ID_CURR')['AMT_PAYMENT'].sum()
ip_total_required_payment = installments_payments.groupby('SK_ID_CURR')['AMT_INSTALMENT'].sum()

# total late & less time in recent 1 year
ip_1y = installments_payments.loc[installments_payments.DAYS_ENTRY_PAYMENT>-365]
# payment 1 year
ip_payment_count_1y = ip_1y.groupby('SK_ID_CURR')['SK_ID_PREV'].count()
ip_creditcard_count_1y = ip_1y.groupby(['SK_ID_CURR','SK_ID_PREV'])['IS_CREDIT'].sum().map(lambda x: 1 if x>0 else 0).reset_index().groupby('SK_ID_CURR')['IS_CREDIT'].sum()
ip_total_late_times_1y = ip_1y.groupby('SK_ID_CURR')['is_late'].sum()
ip_total_less_times_1y = ip_1y.groupby('SK_ID_CURR')['is_less'].sum()
# avg instl
ip_temp = ip_1y.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
ip_avg_instl_1y = ip_temp.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].mean()
# total late & less amount
ip_temp1 = ip_1y.loc[ip_1y.is_late==1]
ip_temp2 = ip_1y.loc[ip_1y.is_less==1]
ip_temp1['total_late'] = ip_temp1.DAYS_ENTRY_PAYMENT - ip_temp1.DAYS_INSTALMENT
ip_temp2['total_less'] = ip_temp2.AMT_INSTALMENT - ip_temp2.AMT_PAYMENT
ip_total_late_days_1y = ip_temp1.groupby('SK_ID_CURR')['total_late'].sum()
ip_total_less_amount_1y = ip_temp2.groupby('SK_ID_CURR')['total_less'].sum()
del ip_temp1, ip_temp2
gc.collect()
# total payment
ip_total_actual_payment_1y = ip_1y.groupby('SK_ID_CURR')['AMT_PAYMENT'].sum()
ip_total_required_payment_1y = ip_1y.groupby('SK_ID_CURR')['AMT_INSTALMENT'].sum()

# total late & less time in recent 6 months
ip_6m = installments_payments.loc[installments_payments.DAYS_ENTRY_PAYMENT>-180]
ip_payment_count_6m = ip_6m.groupby('SK_ID_CURR')['SK_ID_PREV'].count()
ip_creditcard_count_6m = ip_6m.groupby(['SK_ID_CURR','SK_ID_PREV'])['IS_CREDIT'].sum().map(lambda x: 1 if x>0 else 0).reset_index().groupby('SK_ID_CURR')['IS_CREDIT'].sum()
ip_total_late_times_6m = ip_6m.groupby('SK_ID_CURR')['is_late'].sum()
ip_total_less_times_6m = ip_6m.groupby('SK_ID_CURR')['is_less'].sum()
# avg instl
ip_temp = ip_6m.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['NUM_INSTALMENT_NUMBER'].max().reset_index()
ip_avg_instl_6m = ip_temp.groupby('SK_ID_CURR')['NUM_INSTALMENT_NUMBER'].mean()
# total late & less amount
ip_temp1 = ip_6m.loc[ip_6m.is_late==1]
ip_temp2 = ip_6m.loc[ip_6m.is_less==1]
ip_temp1['total_late'] = ip_temp1.DAYS_ENTRY_PAYMENT - ip_temp1.DAYS_INSTALMENT
ip_temp2['total_less'] = ip_temp2.AMT_INSTALMENT - ip_temp2.AMT_PAYMENT
ip_total_late_days_6m = ip_temp1.groupby('SK_ID_CURR')['total_late'].sum()
ip_total_less_amount_6m = ip_temp2.groupby('SK_ID_CURR')['total_less'].sum()
del ip_temp1, ip_temp2
gc.collect()
# total payment
ip_total_actual_payment_6m = ip_6m.groupby('SK_ID_CURR')['AMT_PAYMENT'].sum()
ip_total_required_payment_6m = ip_6m.groupby('SK_ID_CURR')['AMT_INSTALMENT'].sum()

ip_feature = pd.DataFrame({'ip_prev_count':ip_prev_count, 'ip_payment_count':ip_payment_count, 'ip_creditcard_user':ip_creditcard_user, 'ip_creditcard_count':ip_creditcard_count,
                           'ip_total_change_times':ip_total_change_times, 'ip_avg_change_times':ip_avg_change_times, 'ip_avg_instl':ip_avg_instl, 'ip_max_instl':ip_max_instl,
                           'ip_total_late_times':ip_total_late_times, 'ip_total_less_times':ip_total_less_times, 'ip_total_late_days':ip_total_late_days, 'ip_total_less_amount':ip_total_less_amount,
                           'ip_total_actual_payment':ip_total_actual_payment, 'ip_total_required_payment':ip_total_required_payment,
                           'ip_recent_term':ip_recent_term, 'ip_recent_total_actual_payment':ip_recent_total_actual_payment, 'ip_recent_total_required_payment':ip_recent_total_required_payment,
                           'ip_recent_total_late_times':ip_recent_total_late_times, 'ip_recent_total_less_times':ip_recent_total_less_times, 
                           'ip_recent_total_late_days':ip_recent_total_late_days, 'ip_recent_total_less_amount':ip_recent_total_less_amount}).fillna(0)
ip_1y = pd.DataFrame({'ip_payment_count_1y':ip_payment_count_1y, 'ip_creditcard_count_1y': ip_creditcard_count_1y, 'ip_avg_instl_1y':ip_avg_instl_1y,
                      'ip_total_late_times_1y':ip_total_late_times_1y, 'ip_total_less_times_1y':ip_total_less_times_1y, 'ip_total_late_days_1y':ip_total_late_days_1y,
                      'ip_total_less_amount_1y':ip_total_less_amount_1y, 'ip_total_actual_payment_1y':ip_total_actual_payment_1y, 'ip_total_required_payment_1y':ip_total_required_payment_1y}).reset_index().fillna(0)
ip_6m = pd.DataFrame({'ip_payment_count_6m':ip_payment_count_6m, 'ip_creditcard_count_6m': ip_creditcard_count_6m, 'ip_avg_instl_6m':ip_avg_instl_6m,
                      'ip_total_late_times_6m':ip_total_late_times_6m, 'ip_total_less_times_6m':ip_total_less_times_6m, 'ip_total_late_days_6m':ip_total_late_days_6m,
                      'ip_total_less_amount_6m':ip_total_less_amount_6m, 'ip_total_actual_payment_6m':ip_total_actual_payment_6m, 'ip_total_required_payment_6m':ip_total_required_payment_6m}).reset_index().fillna(0)
ip_feature = ip_feature.merge(ip_1y, on='SK_ID_CURR', how='left').merge(ip_6m, on='SK_ID_CURR', how='left')
ip_feature['ip_active_1y'] = ip_feature.ip_total_late_times_1y.notnull().map(lambda x: 1 if x==True else 0)
ip_feature['ip_active_6m'] = ip_feature.ip_total_late_times_6m.notnull().map(lambda x: 1 if x==True else 0)

# active account
ACCOUNT_1Y = installments_payments.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].max().map(lambda x: 1 if x>=-365 else 0)
ACCOUNT_6M = installments_payments.groupby('SK_ID_PREV')['DAYS_ENTRY_PAYMENT'].max().map(lambda x: 1 if x>=-180 else 0)
ACTIVE_ACCOUNT = pd.DataFrame({'ACCOUNT_1Y':ACCOUNT_1Y, 'ACCOUNT_6M':ACCOUNT_6M}).reset_index()
pa_ip = previous_application[['SK_ID_CURR', 'SK_ID_PREV', 'NAME_CONTRACT_TYPE']].merge(ACTIVE_ACCOUNT, on='SK_ID_PREV', how='left')
COUNT_1Y = pd.get_dummies(pa_ip.loc[pa_ip.ACCOUNT_1Y==1, ['SK_ID_CURR', 'NAME_CONTRACT_TYPE']], prefix='ip_count_1y_').groupby('SK_ID_CURR').sum().reset_index()
COUNT_6M = pd.get_dummies(pa_ip.loc[pa_ip.ACCOUNT_6M==1, ['SK_ID_CURR', 'NAME_CONTRACT_TYPE']], prefix='ip_count_6m_').groupby('SK_ID_CURR').sum().reset_index()
ip_feature = ip_feature.merge(COUNT_1Y, on='SK_ID_CURR', how='left').merge(COUNT_6M, on='SK_ID_CURR', how='left')
for i in range(-6, 0):
    ip_feature.iloc[:, i] = ip_feature.iloc[:, i].fillna(0)
ip_feature.shape

ip_feature.to_csv('ip_feature.csv', index=False)
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import gc
import warnings
warnings.filterwarnings("ignore")

application_train = pd.read_csv('../input/application_train.csv')
POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
POS_CASH_balance = POS_CASH_balance.sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])

pcb_prev_count = POS_CASH_balance.groupby('SK_ID_CURR')['SK_ID_PREV'].nunique()
pcb_avg_month = POS_CASH_balance.groupby('SK_ID_CURR')['MONTHS_BALANCE'].count()/pcb_prev_count
pcb_recent_active = POS_CASH_balance.groupby('SK_ID_CURR')['MONTHS_BALANCE'].max()

# times of INSTALMENT change
pcb_temp_inst_change_time = POS_CASH_balance[['SK_ID_PREV', 'CNT_INSTALMENT']].groupby('SK_ID_PREV')['CNT_INSTALMENT'].nunique().map(lambda x: x - 1).reset_index().rename(columns={'CNT_INSTALMENT':'pcb_prev_inst_change_time'})
pcb_temp = POS_CASH_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['MONTHS_BALANCE'].count().reset_index()
pcb_temp = pcb_temp.merge(pcb_temp_inst_change_time, on='SK_ID_PREV')
pcb_inst_change_time = pcb_temp.groupby('SK_ID_CURR')['pcb_prev_inst_change_time'].sum()

# avg INSTALMENT
pcb_temp = POS_CASH_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT'].mean().reset_index()
pcb_avg_inst = pcb_temp.groupby(['SK_ID_CURR'])['CNT_INSTALMENT'].mean()

# active INSTALMENT
pcb_temp = POS_CASH_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['MONTHS_BALANCE'].max().reset_index()
pcb_temp = pcb_temp.merge(POS_CASH_balance[['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'NAME_CONTRACT_STATUS']], on=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
pcb_temp['active_1'] = pcb_temp.MONTHS_BALANCE.map(lambda x: 1 if x>=-4 else 0)
pcb_temp['active_2'] = pcb_temp.NAME_CONTRACT_STATUS.map(lambda x: 1 if x=='Active' else 0)
pcb_temp['active'] = pcb_temp.active_1 * pcb_temp.active_2
pcb_active_inst = pcb_temp.groupby('SK_ID_CURR')['active'].count()

# DPD
pcb_temp = POS_CASH_balance[['SK_ID_CURR', 'SK_ID_PREV', 'SK_DPD', 'SK_DPD_DEF']].groupby(['SK_ID_CURR','SK_ID_PREV'])['SK_DPD','SK_DPD_DEF'].max().reset_index()
pcb_max_dpd_days = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].max()
pcb_total_dpd_days = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].sum()
pcb_max_largedpd_days = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
pcb_total_largedpd_days = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].sum()

pcb_temp = POS_CASH_balance.loc[POS_CASH_balance.MONTHS_BALANCE>=-12, ['SK_ID_CURR', 'SK_ID_PREV', 'SK_DPD', 'SK_DPD_DEF']].groupby(['SK_ID_CURR','SK_ID_PREV'])['SK_DPD','SK_DPD_DEF'].max().reset_index()
pcb_max_dpd_days_1y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].max()
pcb_total_dpd_days_1y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].sum()
pcb_max_largedpd_days_1y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
pcb_total_largedpd_days_1y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].sum()

pcb_temp = POS_CASH_balance.loc[POS_CASH_balance.MONTHS_BALANCE>=-24, ['SK_ID_CURR', 'SK_ID_PREV', 'SK_DPD', 'SK_DPD_DEF']].groupby(['SK_ID_CURR','SK_ID_PREV'])['SK_DPD','SK_DPD_DEF'].max().reset_index()
pcb_max_dpd_days_2y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].max()
pcb_total_dpd_days_2y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD'].sum()
pcb_max_largedpd_days_2y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].max()
pcb_total_largedpd_days_2y = pcb_temp.groupby('SK_ID_CURR')['SK_DPD_DEF'].sum()

pcb_num = pd.DataFrame({'pcb_prev_count':pcb_prev_count, 'pcb_avg_month':pcb_avg_month, 'pcb_recent_active':pcb_recent_active, 'pcb_inst_change_time':pcb_inst_change_time,
                       'pcb_avg_inst':pcb_avg_inst, 'pcb_active_inst':pcb_active_inst, 
                       'pcb_max_dpd_days':pcb_max_dpd_days, 'pcb_total_dpd_days':pcb_total_dpd_days, 'pcb_max_largedpd_days':pcb_max_largedpd_days, 'pcb_total_largedpd_days':pcb_total_largedpd_days,
                       'pcb_max_dpd_days_1y':pcb_max_dpd_days_1y, 'pcb_total_dpd_days_1y':pcb_total_dpd_days_1y, 'pcb_max_largedpd_days_1y':pcb_max_largedpd_days_1y, 
                       'pcb_total_largedpd_days_1y':pcb_total_largedpd_days_1y, 'pcb_max_dpd_days_2y':pcb_max_dpd_days_2y, 'pcb_total_dpd_days_2y':pcb_total_dpd_days_2y, 
                       'pcb_max_largedpd_days_2y':pcb_max_largedpd_days_2y, 'pcb_total_largedpd_days_2y':pcb_total_largedpd_days_2y,}).reset_index()
					   
# INSTALMENT: ratio of end status of each type
pcb_temp = POS_CASH_balance.groupby(['SK_ID_CURR', 'SK_ID_PREV'])['MONTHS_BALANCE'].max().reset_index()
pcb_temp = pcb_temp.merge(POS_CASH_balance[['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE', 'NAME_CONTRACT_STATUS']], on=['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
pcb_temp_dummy = pd.get_dummies(pcb_temp, prefix='pcb_end_as')
pcb_end_as_dummy = pcb_temp_dummy.loc[pcb_temp_dummy.pcb_end_as_Active!=1].drop(['SK_ID_PREV','MONTHS_BALANCE', 'pcb_end_as_Active'], axis=1).groupby('SK_ID_CURR').sum().reset_index()
del pcb_temp_dummy
gc.collect()

pcb_feature = pcb_num.merge(pcb_end_as_dummy, on='SK_ID_CURR', how='left')

pcb_feature['pcb_no_dpd'] = pcb_feature.pcb_total_dpd_days.map(lambda x: 0 if x== 0 else 1)
pcb_feature['pcb_no_largedpd'] = pcb_feature.pcb_total_largedpd_days.map(lambda x: 0 if x== 0 else 1)
pcb_feature['pcb_no_dpd_1y'] = pcb_feature.pcb_total_dpd_days_1y.map(lambda x: 0 if x== 0 else 1)
pcb_feature['pcb_no_largedpd_1y'] = pcb_feature.pcb_total_largedpd_days_1y.map(lambda x: 0 if x== 0 else 1)
pcb_feature['pcb_no_dpd_2y'] = pcb_feature.pcb_total_dpd_days_2y.map(lambda x: 0 if x== 0 else 1)
pcb_feature['pcb_no_largedpd_2y'] = pcb_feature.pcb_total_largedpd_days_2y.map(lambda x: 0 if x== 0 else 1)
pcb_feature.to_csv('pcb_feature.csv', index=False)
import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import gc
import warnings
warnings.filterwarnings("ignore")

previous_application = pd.read_csv("../input/previous_application.csv")
previous_application = previous_application.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])

# cat
cat_col = []
for i in range(len(previous_application.columns)):
    if previous_application.iloc[:, i].dtype == 'object':
        cat_col.append(i)
cat_pa = previous_application.iloc[:, cat_col]
cat_pa = pd.concat([previous_application[['HOUR_APPR_PROCESS_START', 'NFLAG_LAST_APPL_IN_DAY', 'NFLAG_INSURED_ON_APPROVAL']],cat_pa], axis=1).fillna('XNA')
cat_pa.HOUR_APPR_PROCESS_START = cat_pa.HOUR_APPR_PROCESS_START.astype('object')
cat_pa.NFLAG_LAST_APPL_IN_DAY = cat_pa.NFLAG_LAST_APPL_IN_DAY.astype('object')
cat_pa.NFLAG_INSURED_ON_APPROVAL = cat_pa.NFLAG_INSURED_ON_APPROVAL.astype('object')
# selected version
cat_NAME_CONTRACT_TYPE = pd.get_dummies(cat_pa.NAME_CONTRACT_TYPE, prefix='pa_NAME_CONTRACT_TYPE').drop('pa_NAME_CONTRACT_TYPE_XNA', axis=1)
cat_NAME_CASH_LOAN_PURPOSE = pd.get_dummies(cat_pa.NAME_CASH_LOAN_PURPOSE, prefix='pa_NAME_CASH_LOAN_PURPOSE').drop('pa_NAME_CASH_LOAN_PURPOSE_XNA', axis=1)
cat_NAME_CONTRACT_STATUS = pd.get_dummies(cat_pa.NAME_CONTRACT_STATUS, prefix='pa_NAME_CONTRACT_STATUS')
cat_NAME_PAYMENT_TYPE = pd.get_dummies(cat_pa.NAME_PAYMENT_TYPE, prefix='pa_NAME_PAYMENT_TYPE').drop('pa_NAME_PAYMENT_TYPE_XNA', axis=1)
cat_CODE_REJECT_REASON = pd.get_dummies(cat_pa.CODE_REJECT_REASON, prefix='pa_CODE_REJECT_REASON').drop('pa_CODE_REJECT_REASON_XNA', axis=1)
cat_NAME_CLIENT_TYPE = pd.get_dummies(cat_pa.NAME_CLIENT_TYPE, prefix='pa_NAME_CLIENT_TYPE').drop('pa_NAME_CLIENT_TYPE_XNA', axis=1)
cat_NAME_PORTFOLIO = pd.get_dummies(cat_pa.NAME_PORTFOLIO, prefix='pa_NAME_PORTFOLIO').drop('pa_NAME_PORTFOLIO_XNA', axis=1)
cat_NAME_PRODUCT_TYPE = pd.get_dummies(cat_pa.NAME_PRODUCT_TYPE, prefix='pa_NAME_PRODUCT_TYPE').drop('pa_NAME_PRODUCT_TYPE_XNA', axis=1)
cat_NAME_YIELD_GROUP = pd.get_dummies(cat_pa.NAME_YIELD_GROUP, prefix='pa_NAME_YIELD_GROUP').drop('pa_NAME_YIELD_GROUP_XNA', axis=1)
cat_PRODUCT_COMBINATION = pd.get_dummies(cat_pa.PRODUCT_COMBINATION, prefix='pa_PRODUCT_COMBINATION').drop('pa_PRODUCT_COMBINATION_XNA', axis=1)
cat_NFLAG_INSURED_ON_APPROVAL = pd.get_dummies(cat_pa.NFLAG_INSURED_ON_APPROVAL, prefix='pa_NFLAG_INSURED_ON_APPROVAL').drop('pa_NFLAG_INSURED_ON_APPROVAL_XNA', axis=1)
cat_pa_dummy = pd.concat([previous_application[['SK_ID_PREV', 'SK_ID_CURR']], cat_NAME_CONTRACT_TYPE, cat_NAME_CASH_LOAN_PURPOSE, cat_NAME_CONTRACT_STATUS, cat_NAME_PAYMENT_TYPE, cat_CODE_REJECT_REASON, 
                          cat_NAME_CLIENT_TYPE, cat_NAME_PORTFOLIO, cat_NAME_PRODUCT_TYPE, cat_NAME_YIELD_GROUP, cat_PRODUCT_COMBINATION, cat_NFLAG_INSURED_ON_APPROVAL], axis=1)
del cat_NAME_CONTRACT_TYPE, cat_NAME_CASH_LOAN_PURPOSE, cat_NAME_CONTRACT_STATUS, cat_NAME_PAYMENT_TYPE, cat_CODE_REJECT_REASON, cat_NAME_CLIENT_TYPE, 
cat_NAME_PORTFOLIO, cat_NAME_PRODUCT_TYPE, cat_NAME_YIELD_GROUP, cat_PRODUCT_COMBINATION, cat_NFLAG_INSURED_ON_APPROVAL, cat_pa
gc.collect()
pa_cat = cat_pa_dummy.drop('SK_ID_PREV', axis=1).groupby('SK_ID_CURR').mean().reset_index()

# num
num_col = []
for i in range(len(previous_application.columns)):
    if (previous_application.iloc[:, i].dtype == 'int64') or (previous_application.iloc[:, i].dtype == 'float64'):
        num_col.append(i)
num_pa = previous_application.iloc[:, num_col].drop(['HOUR_APPR_PROCESS_START', 'NFLAG_LAST_APPL_IN_DAY', 'NFLAG_INSURED_ON_APPROVAL'], axis=1)
for i in range(-5, 0):
    num_pa.iloc[:, i] = num_pa.iloc[:, i].map(lambda x: np.nan if x==365243.0 else x)
# create new variables
num_pa['app_vs_actual_less'] = (num_pa.AMT_APPLICATION < num_pa.AMT_CREDIT).map(lambda x: 1 if x==True else 0)
num_pa['app_vs_actual_more'] = (num_pa.AMT_APPLICATION > num_pa.AMT_CREDIT).map(lambda x: 1 if x==True else 0)
a = num_pa.DAYS_DECISION - num_pa.DAYS_DECISION.min() + 1
num_pa['ADJ_SCORE'] = (a-a.min()) / (a.max()-a.min()) + 0.5
num_pa['ADJ_AMT_ANNUITY'] = num_pa['ADJ_SCORE']*num_pa['AMT_ANNUITY']
num_pa['ADJ_AMT_APPLICATION'] = num_pa['ADJ_SCORE']*num_pa['AMT_APPLICATION']
num_pa['ADJ_AMT_CREDIT'] = num_pa['ADJ_SCORE']*num_pa['AMT_CREDIT']
num_pa['FREQ'] = (num_pa['DAYS_LAST_DUE']-num_pa['DAYS_FIRST_DUE'])/num_pa['CNT_PAYMENT']
pa_prev_count = num_pa.groupby('SK_ID_CURR')['SK_ID_PREV'].count()
pa_avg_annuity = num_pa.groupby('SK_ID_CURR')['AMT_ANNUITY'].mean()
pa_avg_application = num_pa.groupby('SK_ID_CURR')['AMT_APPLICATION'].mean()
pa_avg_actual_credit = num_pa.groupby('SK_ID_CURR')['AMT_CREDIT'].mean()
pa_max_application = num_pa.groupby('SK_ID_CURR')['AMT_APPLICATION'].max()
pa_max_actual_credit = num_pa.groupby('SK_ID_CURR')['AMT_CREDIT'].max()
pa_total_annuity = num_pa.groupby('SK_ID_CURR')['AMT_ANNUITY'].sum()
pa_total_application = num_pa.groupby('SK_ID_CURR')['AMT_APPLICATION'].sum()
pa_total_actual_credit = num_pa.groupby('SK_ID_CURR')['AMT_CREDIT'].sum()
# compare application and actual credit
pa_not_full_credit_times = num_pa.groupby('SK_ID_CURR')['app_vs_actual_less'].sum()
pa_not_full_credit_rate = num_pa.groupby('SK_ID_CURR')['app_vs_actual_less'].mean()
pa_get_more_credit_times = num_pa.groupby('SK_ID_CURR')['app_vs_actual_more'].sum()
pa_get_more_credit_rate = num_pa.groupby('SK_ID_CURR')['app_vs_actual_more'].mean()
# adjusted: more recent, more important
pa_total_annuity_adj = num_pa.groupby('SK_ID_CURR')['ADJ_AMT_ANNUITY'].sum()
pa_total_application_adj = num_pa.groupby('SK_ID_CURR')['ADJ_AMT_APPLICATION'].sum()
pa_total_actual_credit_adj = num_pa.groupby('SK_ID_CURR')['ADJ_AMT_CREDIT'].sum()
# down payment
pa_avg_down_payment = num_pa.groupby('SK_ID_CURR')['AMT_DOWN_PAYMENT'].mean()
pa_max_down_payment = num_pa.groupby('SK_ID_CURR')['AMT_DOWN_PAYMENT'].max()
pa_total_down_payment = num_pa.groupby('SK_ID_CURR')['AMT_DOWN_PAYMENT'].sum()
# goods price
pa_avg_goods_price = num_pa.groupby('SK_ID_CURR')['AMT_GOODS_PRICE'].mean()
pa_max_goods_price = num_pa.groupby('SK_ID_CURR')['AMT_GOODS_PRICE'].max()
pa_total_goods_price = num_pa.groupby('SK_ID_CURR')['AMT_GOODS_PRICE'].sum()
# down payment rate
pa_avg_down_payment_rate = num_pa.groupby('SK_ID_CURR')['RATE_DOWN_PAYMENT'].mean()
pa_max_down_payment_rate = num_pa.groupby('SK_ID_CURR')['RATE_DOWN_PAYMENT'].max()
pa_total_down_payment_rate = num_pa.groupby('SK_ID_CURR')['RATE_DOWN_PAYMENT'].sum()
# selling area
pa_avg_selling_area = num_pa.groupby('SK_ID_CURR')['SELLERPLACE_AREA'].mean()
pa_max_selling_area = num_pa.groupby('SK_ID_CURR')['SELLERPLACE_AREA'].max()
pa_total_selling_area = num_pa.groupby('SK_ID_CURR')['SELLERPLACE_AREA'].sum()
# term
pa_avg_term = num_pa.groupby('SK_ID_CURR')['CNT_PAYMENT'].mean()
pa_max_term = num_pa.groupby('SK_ID_CURR')['CNT_PAYMENT'].max()
pa_total_term = num_pa.groupby('SK_ID_CURR')['CNT_PAYMENT'].sum()
pa_most_frequent_term = num_pa.groupby('SK_ID_CURR')['CNT_PAYMENT'].agg(pd.Series.mode).map(lambda x: np.mean(x))
# recent decision
pa_recent_decision_day = num_pa.groupby('SK_ID_CURR')['DAYS_DECISION'].max()
pa_earliest_decision_day = num_pa.groupby('SK_ID_CURR')['DAYS_DECISION'].min()
pa_usage_length = pa_recent_decision_day - pa_earliest_decision_day
# application interval
num_pa['application_interval'] = num_pa.groupby('SK_ID_CURR')['DAYS_DECISION'].diff(-1)
missing_iter = iter(num_pa.groupby('SK_ID_CURR')['DAYS_DECISION'].max())
num_pa.application_interval = num_pa.application_interval.map(lambda x: -next(missing_iter) if np.isnan(x) else -x)
pa_avg_intervel = num_pa.groupby('SK_ID_CURR')['application_interval'].mean()
pa_sd_intervel = num_pa.groupby('SK_ID_CURR')['application_interval'].agg('std').fillna(0)
# days
pa_first_due_day = num_pa.groupby('SK_ID_CURR')['DAYS_FIRST_DUE'].min()
pa_last_due_day = num_pa.groupby('SK_ID_CURR')['DAYS_LAST_DUE'].max()
pa_last_termination_day = num_pa.groupby('SK_ID_CURR')['DAYS_TERMINATION'].max()
pa_lastdue_termination_range = pa_last_termination_day - pa_last_due_day
pa_avg_freq = num_pa.groupby('SK_ID_CURR')['FREQ'].mean()
pa_min_freq = num_pa.groupby('SK_ID_CURR')['FREQ'].min()
pa_num = pd.DataFrame({'pa_prev_count':pa_prev_count, 'pa_avg_annuity':pa_avg_annuity, 'pa_avg_application':pa_avg_application, 'pa_avg_actual_credit':pa_avg_actual_credit, 
'pa_max_application':pa_max_application, 'pa_max_actual_credit':pa_max_actual_credit, 'pa_total_annuity':pa_total_annuity, 'pa_total_application':pa_total_application, 'pa_total_actual_credit':pa_total_actual_credit,
'pa_not_full_credit_times':pa_not_full_credit_times, 'pa_not_full_credit_rate':pa_not_full_credit_rate, 'pa_get_more_credit_times':pa_get_more_credit_times, 'pa_get_more_credit_rate':pa_get_more_credit_rate,
'pa_total_annuity_adj':pa_total_annuity_adj, 'pa_total_application_adj':pa_total_application_adj, 'pa_total_actual_credit_adj':pa_total_actual_credit_adj,
'pa_avg_down_payment':pa_avg_down_payment, 'pa_max_down_payment':pa_max_down_payment, 'pa_total_down_payment':pa_total_down_payment,
'pa_avg_goods_price':pa_avg_goods_price, 'pa_max_goods_price':pa_max_goods_price, 'pa_total_goods_price':pa_total_goods_price,
'pa_avg_down_payment_rate':pa_avg_down_payment_rate, 'pa_max_down_payment_rate':pa_max_down_payment_rate, 'pa_total_down_payment_rate':pa_total_down_payment_rate,
'pa_avg_selling_area':pa_avg_selling_area, 'pa_max_selling_area':pa_max_selling_area, 'pa_total_selling_area':pa_total_selling_area,
'pa_avg_term':pa_avg_term, 'pa_max_term':pa_max_term, 'pa_total_term':pa_total_term, 'pa_most_frequent_term':pa_most_frequent_term,
'pa_recent_decision_day':pa_recent_decision_day, 'pa_earliest_decision_day':pa_earliest_decision_day, 'pa_usage_length': pa_usage_length,
'pa_avg_intervel':pa_avg_intervel, 'pa_sd_intervel':pa_sd_intervel, 'pa_first_due_day':pa_first_due_day, 'pa_last_due_day':pa_last_due_day, 
'pa_last_termination_day':pa_last_termination_day, 'pa_lastdue_termination_range':pa_lastdue_termination_range, 'pa_avg_freq':pa_avg_freq, 'pa_min_freq':pa_min_freq}).reset_index()
pa_feature = pa_num.merge(pa_cat, on='SK_ID_CURR')
del pa_num, pa_cat
gc.collect()
pa_feature.to_csv('pa_feature.csv', index=False)
import numpy as np
import pandas as pd
import gc

bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
med = bureau.AMT_CREDIT_SUM.median()
bureau.AMT_CREDIT_SUM = bureau.AMT_CREDIT_SUM.fillna(med)
med = bureau.AMT_CREDIT_SUM_DEBT.median()
bureau.AMT_CREDIT_SUM_DEBT = bureau.AMT_CREDIT_SUM_DEBT.fillna(med)
bureau['OVERDUE_DEBT_RATIO'] = bureau.AMT_CREDIT_SUM_OVERDUE/(bureau.AMT_CREDIT_SUM_DEBT+1)
bureau['DEBT_TOTAL_RATIO'] = bureau.AMT_CREDIT_SUM_DEBT/(bureau.AMT_CREDIT_SUM+1)
bureau_balance['INT_STATUS'] = bureau_balance.STATUS.replace('X', 0.1).replace('C', 0).astype('int64')
bur_max_bad_level = bureau_balance.groupby('SK_ID_BUREAU')['INT_STATUS'].max().reset_index()
cluter_bur = bureau[['SK_ID_BUREAU', 'CREDIT_DAY_OVERDUE', 'OVERDUE_DEBT_RATIO', 'DEBT_TOTAL_RATIO', 'CNT_CREDIT_PROLONG']].merge(bur_max_bad_level, on='SK_ID_BUREAU', how='left').fillna(0)

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import Normalizer
X = cluter_bur.drop(['SK_ID_BUREAU'], axis=1)
X = Normalizer().fit_transform(X)
gmm = GaussianMixture(n_components=2, verbose=5, max_iter=100, init_params='kmeans')
gmm.fit(X)
group_prob = gmm.predict_proba(X)
group_prob = np.round(group_prob, decimals=2)
bur_cluster = pd.concat([bureau[['SK_ID_CURR', 'SK_ID_BUREAU']], pd.DataFrame({'cluster':group_prob[:, 0]})], axis=1)
bur_cluster = bur_cluster.groupby('SK_ID_CURR')['cluster'].mean().reset_index()
bur_cluster.to_csv('bur_cluster.csv', index=False)

import numpy as np
import pandas as pd
import featuretools as ft
import gc
import warnings
import time
a = time.time()
row1=None
row2=None
row3=None
app_train = pd.read_csv('../input/application_train.csv', nrows=row1).sort_values('SK_ID_CURR')
app_test = pd.read_csv('../input/application_test.csv', nrows=row1).sort_values('SK_ID_CURR')
bureau = pd.read_csv('../input/bureau.csv', nrows=row2).sort_values(['SK_ID_CURR', 'SK_ID_BUREAU'])
bureau_balance = pd.read_csv('../input/bureau_balance.csv', nrows=row3).sort_values(['SK_ID_BUREAU', 'MONTHS_BALANCE'])
previous = pd.read_csv('../input/previous_application.csv', nrows=row3).sort_values(['SK_ID_CURR', 'SK_ID_PREV'])
cash = pd.read_csv('../input/POS_CASH_balance.csv', nrows=row3).sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
credit = pd.read_csv('../input/credit_card_balance.csv', nrows=row3).sort_values(['SK_ID_CURR', 'SK_ID_PREV', 'MONTHS_BALANCE'])
installments = pd.read_csv('../input/installments_payments.csv', nrows=row3).sort_values(['SK_ID_CURR', 'SK_ID_PREV'])

# data manipulation
app_train = app_train[['SK_ID_CURR']]
app_test = app_test[['SK_ID_CURR']]
bureau = bureau[['SK_ID_CURR', 'SK_ID_BUREAU', 'DAYS_CREDIT', 'AMT_CREDIT_MAX_OVERDUE', 'CNT_CREDIT_PROLONG', 
                 'AMT_CREDIT_SUM', 'AMT_CREDIT_SUM_DEBT', 'AMT_CREDIT_SUM_LIMIT', 'AMT_CREDIT_SUM_OVERDUE']]
bureau_balance.STATUS = bureau_balance.STATUS.map({'C':0, 'X':0.1, '1':1, '2':2, '3':3, '4':4, '5':5})
previous = previous[['SK_ID_CURR', 'SK_ID_PREV', 'AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'NAME_CONTRACT_STATUS', 'DAYS_DECISION', 
                     'CNT_PAYMENT', 'NAME_YIELD_GROUP', 'NFLAG_INSURED_ON_APPROVAL','SELLERPLACE_AREA']]
previous.NAME_CONTRACT_STATUS = previous.NAME_CONTRACT_STATUS.map(lambda x: 1 if x=='Refused' else 0)
previous.NAME_YIELD_GROUP = previous.NAME_YIELD_GROUP.map({'XNA':0, 'low_noraml':1, 'low_action':1, 'middle':2, 'high':3})
cash = cash[['SK_ID_PREV', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF']]
credit = credit[['SK_ID_PREV', 'MONTHS_BALANCE', 'SK_DPD', 'SK_DPD_DEF', 'AMT_BALANCE', 'AMT_CREDIT_LIMIT_ACTUAL', 'AMT_INST_MIN_REGULARITY', 'AMT_TOTAL_RECEIVABLE']]
installments['date_diff'] = installments.DAYS_INSTALMENT - installments.DAYS_ENTRY_PAYMENT
installments['amount_diff'] = installments.AMT_INSTALMENT - installments.AMT_PAYMENT
installments = installments[['SK_ID_PREV', 'DAYS_INSTALMENT', 'amount_diff', 'date_diff']]

# time
import re

def replace_day_outliers(df):
    """Replace 365243 with np.nan in any columns with DAYS"""
    for col in df.columns:
        if "DAYS" in col:
            df[col] = df[col].replace({365243: np.nan})

    return df

# Replace all the day outliers
app_train = replace_day_outliers(app_train)
app_test = replace_day_outliers(app_test)
bureau = replace_day_outliers(bureau)
bureau_balance = replace_day_outliers(bureau_balance)
previous = replace_day_outliers(previous)
cash = replace_day_outliers(cash)
credit = replace_day_outliers(credit)
installments = replace_day_outliers(installments)
start_date = pd.Timestamp("2018-01-01")
# bureau
for col in ['DAYS_CREDIT']:
    bureau[col] = pd.to_timedelta(bureau[col], 'D')
# Create the date columns
bureau['bureau_credit_application_date'] = start_date + bureau['DAYS_CREDIT']
# balance
bureau_balance['MONTHS_BALANCE'] = pd.to_timedelta(bureau_balance['MONTHS_BALANCE'], 'M')
# Make a date column
bureau_balance['bureau_balance_date'] = start_date + bureau_balance['MONTHS_BALANCE']
bureau = bureau_balance.drop(columns = ['DAYS_CREDIT'])
bureau_balance = bureau_balance.drop(columns = ['MONTHS_BALANCE'])
# Convert to timedeltas in days
for col in ['DAYS_DECISION']:
    previous[col] = pd.to_timedelta(previous[col], 'D')
    
# Make date columns
previous['previous_decision_date'] = start_date + previous['DAYS_DECISION']

# Drop the time offset columns
previous = previous.drop(columns = ['DAYS_DECISION'])

# # cash
cash['MONTHS_BALANCE'] = pd.to_timedelta(cash['MONTHS_BALANCE'], 'M')
cash['cash_balance_date'] = start_date + cash['MONTHS_BALANCE']
cash = cash.drop(columns = ['MONTHS_BALANCE'])

# # credit
credit['MONTHS_BALANCE'] = pd.to_timedelta(credit['MONTHS_BALANCE'], 'M')
credit['credit_balance_date'] = start_date + credit['MONTHS_BALANCE']
credit = credit.drop(columns = ['MONTHS_BALANCE'])

# installment
installments['DAYS_INSTALMENT'] = pd.to_timedelta(installments['DAYS_INSTALMENT'], 'D')
installments['installments_due_date'] = start_date + installments['DAYS_INSTALMENT']
installments = installments.drop(columns = ['DAYS_INSTALMENT'])
# Make an entityset
es = ft.EntitySet(id = 'clients')

es = es.entity_from_dataframe(entity_id = 'app_train', dataframe = app_train, 
                              index = 'SK_ID_CURR')

es = es.entity_from_dataframe(entity_id = 'app_test', dataframe = app_test, 
                              index = 'SK_ID_CURR')

# part 1
es = es.entity_from_dataframe(entity_id = 'bureau', dataframe = bureau, 
                              index = 'SK_ID_BUREAU', time_index='bureau_credit_application_date')

es = es.entity_from_dataframe(entity_id = 'bureau_balance', dataframe = bureau_balance, 
                              make_index = True, index = 'bb_index',
                              time_index = 'bureau_balance_date')

# part 2
es = es.entity_from_dataframe(entity_id = 'previous', dataframe = previous, 
                              index = 'SK_ID_PREV', time_index = 'previous_decision_date')


es = es.entity_from_dataframe(entity_id = 'cash', dataframe = cash, 
                              make_index = True, index = 'cash_index',
                              time_index = 'cash_balance_date')

es = es.entity_from_dataframe(entity_id = 'installments', dataframe = installments,
                              make_index = True, index = 'installments_index',
                              time_index = 'installments_due_date')

es = es.entity_from_dataframe(entity_id = 'credit', dataframe = credit,
                              make_index = True, index = 'credit_index',
                              time_index = 'credit_balance_date')
# Relationship between app and bureau
r_app_bureau = ft.Relationship(es['app_train']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])

# Test Relationship between app and bureau
r_test_app_bureau = ft.Relationship(es['app_test']['SK_ID_CURR'], es['bureau']['SK_ID_CURR'])

# Relationship between bureau and bureau balance
r_bureau_balance = ft.Relationship(es['bureau']['SK_ID_BUREAU'], es['bureau_balance']['SK_ID_BUREAU'])

# Relationship between current app and previous apps
r_app_previous = ft.Relationship(es['app_train']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

# Test Relationship between current app and previous apps
r_test_app_previous = ft.Relationship(es['app_test']['SK_ID_CURR'], es['previous']['SK_ID_CURR'])

# Relationships between previous apps and cash, installments, and credit
r_previous_cash = ft.Relationship(es['previous']['SK_ID_PREV'], es['cash']['SK_ID_PREV'])
r_previous_installments = ft.Relationship(es['previous']['SK_ID_PREV'], es['installments']['SK_ID_PREV'])
r_previous_credit = ft.Relationship(es['previous']['SK_ID_PREV'], es['credit']['SK_ID_PREV'])

# Add in the defined relationships
es = es.add_relationships([r_app_bureau, r_test_app_bureau, r_bureau_balance, 
						   r_app_previous, r_test_app_previous, r_previous_cash,
						   r_previous_installments, r_previous_credit])
						   

del app_train, app_test, bureau, bureau_balance, cash, credit, previous, installments
gc.collect()
print('prepare time:', str(time.time()-a))

# train features
a = time.time()
time_features, time_feature_names = ft.dfs(entityset = es, target_entity = 'app_train', 
                                           #trans_primitives = ['cum_sum'], 
                                           max_depth = 2,
                                           agg_primitives = ['trend'],
                                           features_only = False, verbose = True,
                                           chunk_size = 30000,
                                           ignore_entities = ['app_test'])
print('feature time:,', str(time.time()-a))
time_features.reset_index().to_csv('trend3_train.csv', index=False)
# test features
time_features_test, time_feature_names = ft.dfs(entityset = es, target_entity = 'app_test', 
                                           #trans_primitives = ['cum_sum', 'time_since_previous'], 
                                           max_depth = 2,
                                           agg_primitives = ['trend'],
                                           features_only = False, verbose = True,
                                           chunk_size = 25000,
                                           ignore_entities = ['app_train'])
time_features_test.reset_index().to_csv('trend4_test.csv', index=False)

import pandas as pd

full_df = pd.read_csv("../input/home-credit-default-risk/application_train.csv")
test_df = pd.read_csv("../input/home-credit-default-risk/application_test.csv")

# bureau_feature_df = pd.read_csv("../input/features/bureau_feature.csv")
# ccb_feature_df = pd.read_csv("../input/features/ccb_feature.csv")
# ip_feature_df = pd.read_csv("../input/features/ip_feature.csv")
# pcb_feature_df = pd.read_csv("../input/features/pcb_feature.csv")
# pa_feature_df = pd.read_csv("../input/features/pa_feature.csv")
# bur_cluster_df = pd.read_csv("../input/features/bur_cluster.csv")

# full_df = full_df.merge(bureau_feature_df, on = 'SK_ID_CURR', how = 'left')
# full_df = full_df.merge(ccb_feature_df, on = 'SK_ID_CURR', how = 'left')
# full_df = full_df.merge(ip_feature_df, on = 'SK_ID_CURR', how = 'left')
# full_df = full_df.merge(pcb_feature_df, on = 'SK_ID_CURR', how = 'left')
# full_df = full_df.merge(pa_feature_df, on = 'SK_ID_CURR', how = 'left')
# full_df = full_df.merge(bur_cluster_df, on = 'SK_ID_CURR', how = 'left')

# test_df = test_df.merge(bureau_feature_df, on = 'SK_ID_CURR', how = 'left')
# test_df = test_df.merge(ccb_feature_df, on = 'SK_ID_CURR', how = 'left')
# test_df = test_df.merge(ip_feature_df, on = 'SK_ID_CURR', how = 'left')
# test_df = test_df.merge(pcb_feature_df, on = 'SK_ID_CURR', how = 'left')
# test_df = test_df.merge(pa_feature_df, on = 'SK_ID_CURR', how = 'left')
# test_df = test_df.merge(bur_cluster_df, on = 'SK_ID_CURR', how = 'left')

test_df

# del bureau_feature_df
# del ccb_feature_df
# del ip_feature_df
# del pcb_feature_df 
# del pa_feature_df 
# del bur_cluster_df 

# one-hot encoding of categorical variables
full_df = pd.get_dummies(full_df)
test_df = pd.get_dummies(test_df)

full_df, test_df = full_df.align(test_df, join = 'inner', axis = 1)

test_df.head(10)
test_df.count()
# list(full_df.drop(columns=['TARGET']).columns)
full_df.columns
# import numpy as np

# from sklearn.preprocessing import StandardScaler
# from sklearn.impute import SimpleImputer


# full_df = full_df.replace([np.inf, -np.inf], np.nan)
# test_df = test_df.replace([np.inf, -np.inf], np.nan)

# test_df.head(10)

# # Feature names
# features = list(full_df.columns)

# # Median imputation of missing values
# imputer = SimpleImputer(strategy = 'median')

# # Scale each feature to 0-1
# scaler = StandardScaler()

# # Fit on the training data
# imputer.fit(full_df)

# # Transform both training and testing data
# train = imputer.transform(full_df)
# test = imputer.transform(test_df)

# # Repeat with the scaler
# scaler.fit(train)
# train = scaler.transform(train)
# test = scaler.transform(test)

# print('Training data shape: ', train.shape)
# print('Testing data shape: ', test.shape)


# full_df = pd.DataFrame(data=train, columns=full_df.columns)
# test_df = pd.DataFrame(data=test, columns=test_df.columns)



full_df.head()
test_df.head()
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc

def model(features, test_features, encoding = 'ohe', n_folds = 5):
    
    """Train and test a light gradient boosting model using
    cross validation. 
    
    Parameters
    --------
        features (pd.DataFrame): 
            dataframe of training features to use 
            for training a model. Must include the TARGET column.
        test_features (pd.DataFrame): 
            dataframe of testing features to use
            for making predictions with the model. 
        encoding (str, default = 'ohe'): 
            method for encoding categorical variables. Either 'ohe' for one-hot encoding or 'le' for integer label encoding
            n_folds (int, default = 5): number of folds to use for cross validation
        
    Return
    --------
        submission (pd.DataFrame): 
            dataframe with `SK_ID_CURR` and `TARGET` probabilities
            predicted by the model.
        feature_importances (pd.DataFrame): 
            dataframe with the feature importances from the model.
        valid_metrics (pd.DataFrame): 
            dataframe with training and validation metrics (ROC AUC) for each fold and overall.
        
    """
    
    # Extract the ids
    train_ids = features['SK_ID_CURR']
    test_ids = test_features['SK_ID_CURR']
    
    # Extract the labels for training
    labels = features['TARGET']
    
    # Remove the ids and target
    features = features.drop(columns = ['SK_ID_CURR', 'TARGET'])
    test_features = test_features.drop(columns = ['SK_ID_CURR'])
    
    
    # One Hot Encoding
    if encoding == 'ohe':
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        
        # Align the dataframes by the columns
        features, test_features = features.align(test_features, join = 'inner', axis = 1)
        
        # No categorical indices to record
        cat_indices = 'auto'
    
    # Integer label encoding
    elif encoding == 'le':
        
        # Create a label encoder
        label_encoder = LabelEncoder()
        
        # List for storing categorical indices
        cat_indices = []
        
        # Iterate through each column
        for i, col in enumerate(features):
            if features[col].dtype == 'object':
                # Map the categorical features to integers
                features[col] = label_encoder.fit_transform(np.array(features[col].astype(str)).reshape((-1,)))
                test_features[col] = label_encoder.transform(np.array(test_features[col].astype(str)).reshape((-1,)))

                # Record the categorical indices
                cat_indices.append(i)
    
    # Catch error if label encoding scheme is not valid
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
        
    print('Training Data Shape: ', features.shape)
    print('Testing Data Shape: ', test_features.shape)
    
    # Extract feature names
    feature_names = list(features.columns)
    
    # Convert to np arrays
    features = np.array(features)
    test_features = np.array(test_features)
    
    # Create the kfold object
    k_fold = KFold(n_splits = n_folds, shuffle = True, random_state = 50)
    
    # Empty array for feature importances
    feature_importance_values = np.zeros(len(feature_names))
    
    # Empty array for test predictions
    test_predictions = np.zeros(test_features.shape[0])
    
    # Empty array for out of fold validation predictions
    out_of_fold = np.zeros(features.shape[0])
    
    # Lists for recording validation and training scores
    valid_scores = []
    train_scores = []
    
    # Iterate through each fold
    for train_indices, valid_indices in k_fold.split(features):
        
        # Training data for the fold
        train_features, train_labels = features[train_indices], labels[train_indices]
        # Validation data for the fold
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        
        # Create the model
        model = lgb.LGBMClassifier(n_estimators=500, objective = 'binary', 
                                   class_weight = 'balanced', learning_rate = 0.5, 
                                   reg_alpha = 0.7, reg_lambda = 0.5, 
                                   subsample = 0.5, n_jobs = -1, random_state = 30)
        
        # Train the model
        model.fit(train_features, train_labels, eval_metric = 'auc',
                  eval_set = [(valid_features, valid_labels), (train_features, train_labels)],
                  eval_names = ['valid', 'train'], categorical_feature = cat_indices,
                  early_stopping_rounds = 25, verbose = 200)
        
        # Record the best iteration
        best_iteration = model.best_iteration_
        
        # Record the feature importances
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        
        # Make predictions
        test_predictions += model.predict_proba(test_features, num_iteration = best_iteration)[:, 1] / k_fold.n_splits
        
        # Record the out of fold predictions
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration = best_iteration)[:, 1]
        
        # Record the best score
        valid_score = model.best_score_['valid']['auc']
        train_score = model.best_score_['train']['auc']
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Clean up memory
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
        
    # Make the submission dataframe
    submission = pd.DataFrame({'SK_ID_CURR': test_ids, 'TARGET': test_predictions})
    
    # Make the feature importance dataframe
    feature_importances = pd.DataFrame({'feature': feature_names, 'importance': feature_importance_values})
    
    # Overall validation score
    valid_auc = roc_auc_score(labels, out_of_fold)
    
    # Add the overall scores to the metrics
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    
    # Needed for creating dataframe of validation scores
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    # Dataframe of validation scores
    metrics = pd.DataFrame({'fold': fold_names,
                            'train': train_scores,
                            'valid': valid_scores}) 
    
    return submission, feature_importances, metrics
import numpy as np
submission, fi, metrics = model(full_df, test_df)
print('Baseline metrics')
print(metrics)
submission.to_csv('baseline_lgb.csv', index = False)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt
import seaborn as sns

from imblearn.over_sampling import RandomOverSampler

import os
import gc
#print(os.listdir("../input"))
def get_combined_dataset() :
    application_train = pd.read_csv('../input/application_train.csv')
    application_test = pd.read_csv('../input/application_test.csv')
    application=application_train.append(application_test, ignore_index=True,sort=False)
    application.set_index('SK_ID_CURR')
    return (application)

def get_application_dataset():
    df = get_combined_dataset()
    filteredColList =['NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE',
                      'WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE'] 
    df = df[[x for x in list(df) if x not in filteredColList]]
    oheCols = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE']
    
    df.loc[df.CODE_GENDER == 'XNA' ,'CODE_GENDER'] = 'F'
    df.loc[(df.DAYS_EMPLOYED > 0),'DAYS_EMPLOYED'] = np.nan
    df.loc[(df.REGION_RATING_CLIENT_W_CITY < 0),'REGION_RATING_CLIENT_W_CITY'] = np.nan
    df.loc[(df.OBS_30_CNT_SOCIAL_CIRCLE > 10),'OBS_30_CNT_SOCIAL_CIRCLE'] = 10
    df.loc[(df.DEF_30_CNT_SOCIAL_CIRCLE > 10),'DEF_30_CNT_SOCIAL_CIRCLE'] = 10
    df.loc[(df.OBS_60_CNT_SOCIAL_CIRCLE > 10),'OBS_60_CNT_SOCIAL_CIRCLE'] = 10
    df.loc[(df.DEF_60_CNT_SOCIAL_CIRCLE > 10),'DEF_60_CNT_SOCIAL_CIRCLE'] = 10
    df.loc[(df.AMT_REQ_CREDIT_BUREAU_QRT > 10),'AMT_REQ_CREDIT_BUREAU_QRT'] = 10
    df = pd.get_dummies(df,columns=oheCols)
    
    # New features
    df['NEW_INCOME2Credit']=df['AMT_CREDIT']/df['AMT_INCOME_TOTAL']
    df['NEW_Credit2ANNUITY']=df['AMT_ANNUITY']/df['AMT_CREDIT']
    df['NEW_INCOME2ANNUITY']=df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL']
    df['NEW_DAYS_EMPLOYED2DAYS_BIRTH'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']
    df['NEW_AMT_INCOME_TOTAL2CNT_FAM_MEMBERS'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
    
    df['NEW_CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    df['NEW_CREDIT2GOODS'] = df['AMT_CREDIT'] / df['AMT_GOODS_PRICE']
    df['NEW_INC_PER_CHLD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
    df['NEW_SOURCES_PROD'] = df['EXT_SOURCE_1'] * df['EXT_SOURCE_2'] * df['EXT_SOURCE_3']
    df['NEW_EXT_SOURCES_MEAN'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    df['NEW_SCORES_STD'] = df[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    df['NEW_SCORES_STD'] = df['NEW_SCORES_STD'].fillna(df['NEW_SCORES_STD'].mean())
    df['NEW_OWN_CAR_AGE2DAYS_BIRTH'] = df['OWN_CAR_AGE'] / df['DAYS_BIRTH']
    df['NEW_OWN_CAR_AGE2DAYS_EMPLOYED'] = df['OWN_CAR_AGE'] / df['DAYS_EMPLOYED']
    df['NEW_DAYS_LAST_PHONE_CHANGE2DAYS_BIRTH'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_BIRTH']
    df['NEW_DAYS_LAST_PHONE_CHANGE2DAYS_EMPLOYED'] = df['DAYS_LAST_PHONE_CHANGE'] / df['DAYS_EMPLOYED']
    return(df)

def transform_application(df):
    
    logTransformation = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','NEW_AMT_INCOME_TOTAL2CNT_FAM_MEMBERS','NEW_INC_PER_CHLD']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.log(x+1),axis=1)
    
    sqrtTransformation = ['DAYS_BIRTH','DAYS_EMPLOYED','DAYS_REGISTRATION','DAYS_ID_PUBLISH','OWN_CAR_AGE','DAYS_LAST_PHONE_CHANGE','NEW_INCOME2Credit']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)

    return(df)
def bureau_balance():
    df = pd.read_csv('../input/bureau_balance.csv')
    # getting the furthest date attached to bureau_id
    df1 = df.groupby(['SK_ID_BUREAU']).agg(
            {'MONTHS_BALANCE': min,
            })
    # Status of bureau_id as per freshest month
    df2 = df.groupby(['SK_ID_BUREAU']).agg(
                {'MONTHS_BALANCE': max,
                }).reset_index()
    df2 = pd.merge(df2,df,on=['SK_ID_BUREAU','MONTHS_BALANCE'],how='inner')
    df2 = pd.crosstab(df2['SK_ID_BUREAU'], df2['STATUS'])

    df = pd.merge(df1,df2,on=['SK_ID_BUREAU'],how='left').reset_index()
    df.columns = ['SK_ID_BUREAU','MONTHS_BALANCE','BB_S_0','BB_S_1','BB_S_2','BB_S_3','BB_S_4','BB_S_5','BB_S_C','BB_S_X']
    return(df)

def get_bureau_dataset():
    b = pd.read_csv('../input/bureau.csv')
    bb = bureau_balance()
    df = pd.merge(b,bb,on='SK_ID_BUREAU',how='left')
    df.loc[(df.DAYS_CREDIT_ENDDATE < 0) | (df.DAYS_CREDIT_ENDDATE > 5000),'DAYS_CREDIT_ENDDATE'] = np.nan
    df.loc[(df.DAYS_ENDDATE_FACT < -5000),'DAYS_ENDDATE_FACT'] = np.nan
    df.loc[(df.AMT_CREDIT_MAX_OVERDUE > 40000),'AMT_CREDIT_MAX_OVERDUE'] = 40000
    df.loc[(df.DAYS_CREDIT_UPDATE < -3000),'DAYS_CREDIT_UPDATE'] = np.nan
    df.loc[(df.AMT_CREDIT_SUM_DEBT < 0),'AMT_CREDIT_SUM_DEBT'] = np.nan
    df.loc[(df.AMT_CREDIT_SUM_LIMIT < 0),'AMT_CREDIT_SUM_LIMIT'] = np.nan

    All = df.groupby(['SK_ID_CURR']).agg(
            {'DAYS_CREDIT': [min, max],
             'CREDIT_DAY_OVERDUE':max,
             'DAYS_CREDIT_ENDDATE':max,
             'DAYS_ENDDATE_FACT':[min,max],
             'AMT_CREDIT_MAX_OVERDUE':max,
             'CNT_CREDIT_PROLONG':max,
             'AMT_CREDIT_SUM':max,
             'AMT_CREDIT_SUM_DEBT':max,
             'AMT_CREDIT_SUM_LIMIT':max,
             'DAYS_CREDIT_UPDATE':min,
             'AMT_ANNUITY':max,
             'MONTHS_BALANCE':min,
             'BB_S_0':sum,
             'BB_S_1':sum,
             'BB_S_2':sum,
             'BB_S_3':sum,
             'BB_S_4':sum,
             'BB_S_5':sum,
             'BB_S_C':sum,
             'BB_S_X':sum
            })
    All.columns = ["_all_".join(x) for x in All.columns.ravel()]
    Active = df.query('CREDIT_ACTIVE == "Active"').groupby(['SK_ID_CURR']).agg(
            {'CREDIT_DAY_OVERDUE':max,
             'AMT_CREDIT_MAX_OVERDUE': max,
             'CNT_CREDIT_PROLONG':[max,sum],
             'AMT_CREDIT_SUM':sum,
             'AMT_CREDIT_SUM_DEBT':sum,
             'AMT_CREDIT_SUM_LIMIT':sum,
             'AMT_CREDIT_SUM_OVERDUE':sum,
             'DAYS_CREDIT_UPDATE':min,
             'AMT_ANNUITY':sum,
             'MONTHS_BALANCE':min,
             'BB_S_0':sum,
             'BB_S_1':sum,
             'BB_S_2':sum,
             'BB_S_3':sum,
             'BB_S_4':sum,
             'BB_S_5':sum,
             'BB_S_C':sum,
             'BB_S_X':sum
            })
    Active.columns = ["_act_".join(x) for x in Active.columns.ravel()]
    
    CREDIT_ACTIVE_ctab = pd.crosstab(df['SK_ID_CURR'], df['CREDIT_ACTIVE']).rename_axis(None, axis=1)
    from functools import reduce
    dfs = [All,Active,CREDIT_ACTIVE_ctab]

    df_final = reduce(lambda left,right: pd.merge(left,right,on='SK_ID_CURR',how='outer'), dfs)
    df_final.reset_index(inplace=True)
    return(df_final)

    CREDIT_ACTIVE_ctab = pd.crosstab(df['SK_ID_CURR'], df['CREDIT_ACTIVE']).rename_axis(None, axis=1)
    from functools import reduce
    dfs = [All,Active,CREDIT_ACTIVE_ctab]

    df_final = reduce(lambda left,right: pd.merge(left,right,on='SK_ID_CURR',how='outer'), dfs)
    df_final.reset_index(inplace=True)
    return(df_final)

def bureau_newFeature(df):
    df['AMT_CREDIT_SUM_sum2AMT_CREDIT_SUM_DEBT_sum'] = df['AMT_CREDIT_SUM_DEBT_act_sum']/df['AMT_CREDIT_SUM_act_sum']
    df['AMT_CREDIT_SUM_sum2AMT_ANNUITY_sum'] = df['AMT_CREDIT_SUM_act_sum']/df['AMT_ANNUITY_act_sum']
    df['AMT_CREDIT_SUM_DEBT_sum2AMT_ANNUITY_sum'] = df['AMT_CREDIT_SUM_DEBT_act_sum']/df['AMT_ANNUITY_act_sum']
    df.replace([np.inf, -np.inf], np.nan,inplace=True)
    df.loc[df.AMT_CREDIT_SUM_sum2AMT_CREDIT_SUM_DEBT_sum>2,'AMT_CREDIT_SUM_sum2AMT_CREDIT_SUM_DEBT_sum'] = np.nan
    df.loc[df.AMT_CREDIT_SUM_sum2AMT_ANNUITY_sum>120,'AMT_CREDIT_SUM_sum2AMT_ANNUITY_sum'] = np.nan
    df.loc[df.AMT_CREDIT_SUM_DEBT_sum2AMT_ANNUITY_sum>80,'AMT_CREDIT_SUM_DEBT_sum2AMT_ANNUITY_sum'] = np.nan
    return(df)
    
def transform_bureau(df):
    logTransformation = ['CREDIT_DAY_OVERDUE_all_max','AMT_CREDIT_MAX_OVERDUE_all_max','AMT_CREDIT_SUM_all_max','AMT_CREDIT_SUM_DEBT_all_max',
                         'AMT_CREDIT_SUM_LIMIT_all_max','AMT_ANNUITY_all_max','AMT_CREDIT_MAX_OVERDUE_act_max','AMT_CREDIT_SUM_act_sum',
                         'AMT_CREDIT_SUM_DEBT_act_sum','AMT_CREDIT_SUM_LIMIT_act_sum','AMT_CREDIT_SUM_OVERDUE_act_sum','AMT_ANNUITY_act_sum']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.log(x+1),axis=1)
    
    sartLogTransformation = ['CREDIT_DAY_OVERDUE_act_max','DAYS_CREDIT_UPDATE_act_min']
    df[sartLogTransformation] = df[sartLogTransformation].apply(lambda x : np.sqrt(np.log(np.abs(x+1))),axis=1)
    
    sqrtTransformation = ['DAYS_CREDIT_all_min','DAYS_CREDIT_all_max','DAYS_CREDIT_ENDDATE_all_max','DAYS_ENDDATE_FACT_all_min','DAYS_ENDDATE_FACT_all_max',
                         'DAYS_CREDIT_UPDATE_all_min','MONTHS_BALANCE_all_min','MONTHS_BALANCE_act_min']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)

    return(df)
def get_previous_application():
    df = pd.read_csv('../input/previous_application.csv')
    df.loc[df.DAYS_FIRST_DRAWING >0,'DAYS_FIRST_DRAWING'] = np.nan
    df.loc[df.DAYS_FIRST_DUE >0,'DAYS_FIRST_DUE'] = np.nan
    df.loc[df.DAYS_LAST_DUE_1ST_VERSION >2000,'DAYS_LAST_DUE_1ST_VERSION'] = np.nan
    df.loc[df.DAYS_LAST_DUE >3000,'DAYS_LAST_DUE'] = np.nan
    df.loc[df.DAYS_TERMINATION >3000,'DAYS_TERMINATION'] = np.nan
    
    # filtering the invalid contracts
    dff = df.query('FLAG_LAST_APPL_PER_CONTRACT == "Y" and NFLAG_LAST_APPL_IN_DAY == 1')
    # count of loan status
    NAME_CONTRACT_STATUS_ctab = pd.crosstab(df['SK_ID_CURR'], df['NAME_CONTRACT_STATUS'])
    

    # new features
    df['AMT_ANNUITY2AMT_CREDIT'] = df['AMT_ANNUITY']/df['AMT_CREDIT']
    df['AMT_APPLICATION2AMT_CREDIT'] = df['AMT_APPLICATION']/df['AMT_CREDIT']
    df['AMT_GOODS_PRICE2AMT_CREDIT'] = df['AMT_GOODS_PRICE']/df['AMT_CREDIT']
    
    
    NAME_CONTRACT_STATUS_ctab = pd.crosstab(df['SK_ID_CURR'], df['NAME_CONTRACT_STATUS'])
    df_grouped = df.query('NAME_CONTRACT_STATUS != "Refused" and FLAG_LAST_APPL_PER_CONTRACT == "Y" and NFLAG_LAST_APPL_IN_DAY == 1')\
                                                    .groupby(['SK_ID_CURR'])\
                                                    .agg(
                                                        {'AMT_ANNUITY':max,
                                                         'AMT_APPLICATION':max,
                                                         'AMT_CREDIT':max,
                                                         'AMT_DOWN_PAYMENT':max,
                                                         'AMT_GOODS_PRICE':max,
                                                         'RATE_DOWN_PAYMENT':[min, max,'mean'],
                                                         'RATE_INTEREST_PRIMARY':[min, max,'mean'],
                                                         'RATE_INTEREST_PRIVILEGED':[min, max,'mean'],
                                                         'DAYS_DECISION':[min, max,'mean'],
                                                         'CNT_PAYMENT':[min, max,'mean'],
                                                         'DAYS_FIRST_DRAWING':min,
                                                         'DAYS_FIRST_DUE':[min, max],
                                                         'DAYS_LAST_DUE_1ST_VERSION':[min, max],
                                                         'DAYS_LAST_DUE':[min, max],
                                                         'DAYS_TERMINATION':[min, max],
                                                         'NFLAG_INSURED_ON_APPROVAL':sum
                                                        })
    df_final = pd.merge(df_grouped,NAME_CONTRACT_STATUS_ctab,on='SK_ID_CURR',how='outer')
    df_final.reset_index(inplace=True)
    df_final.columns = ['SK_ID_CURR','AMT_ANNUITY_max','AMT_APPLICATION_max','AMT_CREDIT_max','AMT_DOWN_PAYMENT_max','AMT_GOODS_PRICE_max',
                        'RATE_DOWN_PAYMENT_min','RATE_DOWN_PAYMENT_max','RATE_INTEREST_PRIMARY_min','RATE_INTEREST_PRIMARY_max',
                        'RATE_INTEREST_PRIVILEGED_min','RATE_INTEREST_PRIVILEGED_max','DAYS_DECISION_min','DAYS_DECISION_max','CNT_PAYMENT_min',
                        'CNT_PAYMENT_max','DAYS_FIRST_DRAWING_min','DAYS_FIRST_DUE_min','DAYS_FIRST_DUE_max',
                        'DAYS_LAST_DUE_1ST_VERSION_min','DAYS_LAST_DUE_1ST_VERSION_max','DAYS_LAST_DUE_min','DAYS_LAST_DUE_max','DAYS_TERMINATION_min',
                        'DAYS_TERMINATION_max','NFLAG_INSURED_ON_APPROVAL_sum','Approved','Canceled','Refused','Unused_offer']
    df_final.head()
    return(df_final)

def transform_previous_application(df):
    logTransformation = ['AMT_ANNUITY_max','AMT_APPLICATION_max','AMT_CREDIT_max', 'AMT_DOWN_PAYMENT_max','AMT_GOODS_PRICE_max']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.log(x+1),axis=1)
    
    sqrtTransformation = ['DAYS_DECISION_min','DAYS_DECISION_max','DAYS_FIRST_DRAWING_min','DAYS_FIRST_DUE_min','DAYS_FIRST_DUE_max','DAYS_LAST_DUE_min',
                          'DAYS_LAST_DUE_max','DAYS_TERMINATION_min','DAYS_TERMINATION_max']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)
    return(df)
def get_previous_application():
    df = pd.read_csv('../input/previous_application.csv')
    df.loc[df.DAYS_FIRST_DRAWING >0,'DAYS_FIRST_DRAWING'] = np.nan
    df.loc[df.DAYS_FIRST_DUE >0,'DAYS_FIRST_DUE'] = np.nan
    df.loc[df.DAYS_LAST_DUE_1ST_VERSION >2000,'DAYS_LAST_DUE_1ST_VERSION'] = np.nan
    df.loc[df.DAYS_LAST_DUE >3000,'DAYS_LAST_DUE'] = np.nan
    df.loc[df.DAYS_TERMINATION >3000,'DAYS_TERMINATION'] = np.nan

    # new features
    df['AMT_ANNUITY2AMT_CREDIT'] = df['AMT_ANNUITY']/df['AMT_CREDIT']
    df['AMT_APPLICATION2AMT_CREDIT'] = df['AMT_APPLICATION']/df['AMT_CREDIT']
    df['AMT_GOODS_PRICE2AMT_CREDIT'] = df['AMT_GOODS_PRICE']/df['AMT_CREDIT']
    
    
    NAME_CONTRACT_STATUS_ctab = pd.crosstab(df['SK_ID_CURR'], df['NAME_CONTRACT_STATUS'])
    df_grouped = df.query('NAME_CONTRACT_STATUS != "Refused" and FLAG_LAST_APPL_PER_CONTRACT == "Y" and NFLAG_LAST_APPL_IN_DAY == 1')\
                                                    .groupby(['SK_ID_CURR'])\
                                                    .agg(
                                                        {'AMT_ANNUITY':max,
                                                         'AMT_APPLICATION':max,
                                                         'AMT_CREDIT':max,
                                                         'AMT_DOWN_PAYMENT':max,
                                                         'AMT_GOODS_PRICE':max,
                                                         'RATE_DOWN_PAYMENT':[min, max,'mean'],
                                                         'RATE_INTEREST_PRIMARY':[min, max,'mean'],
                                                         'RATE_INTEREST_PRIVILEGED':[min, max,'mean'],
                                                         'DAYS_DECISION':[min, max,'mean'],
                                                         'CNT_PAYMENT':[min, max,'mean'],
                                                         'DAYS_FIRST_DRAWING':min,
                                                         'DAYS_FIRST_DUE':[min, max],
                                                         'DAYS_LAST_DUE_1ST_VERSION':[min, max],
                                                         'DAYS_LAST_DUE':[min, max],
                                                         'DAYS_TERMINATION':[min, max],
                                                         'NFLAG_INSURED_ON_APPROVAL':sum
                                                        })
    df_final = pd.merge(df_grouped,NAME_CONTRACT_STATUS_ctab,on='SK_ID_CURR',how='outer')
    df_final.reset_index(inplace=True)
    return(df_final)

def transform_previous_application(df):
    logTransformation = ['AMT_ANNUITY_max','AMT_APPLICATION_max','AMT_CREDIT_max', 'AMT_DOWN_PAYMENT_max','AMT_GOODS_PRICE_max']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.log(x+1),axis=1)
    
    sqrtTransformation = ['DAYS_DECISION_min','DAYS_DECISION_max','DAYS_FIRST_DRAWING_min','DAYS_FIRST_DUE_min','DAYS_FIRST_DUE_max','DAYS_LAST_DUE_min',
                          'DAYS_LAST_DUE_max','DAYS_TERMINATION_min','DAYS_TERMINATION_max']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)
    return(df)
def get_POS_CASH_balance():
    POS_CASH_balance = pd.read_csv('../input/POS_CASH_balance.csv')
    Closed_Loans = POS_CASH_balance[POS_CASH_balance['SK_ID_PREV'].isin(POS_CASH_balance.query('NAME_CONTRACT_STATUS == "Completed"').SK_ID_PREV)]
    Active_Loans = POS_CASH_balance[~POS_CASH_balance['SK_ID_PREV'].isin(POS_CASH_balance.query('NAME_CONTRACT_STATUS == "Active" and MONTHS_BALANCE == -1').SK_ID_PREV)]

    Active = Active_Loans.groupby(['SK_ID_CURR']).agg(
                    {  'MONTHS_BALANCE':min,
                       'CNT_INSTALMENT':[min,max],
                       'CNT_INSTALMENT_FUTURE':[min,max]
                    })
    Closed = Closed_Loans.groupby(['SK_ID_CURR']).agg(
                    {  'MONTHS_BALANCE':[min,max],
                       'CNT_INSTALMENT':max
                    })
    NAME_CONTRACT_STATUS = POS_CASH_balance.query('(NAME_CONTRACT_STATUS == "Completed") or (NAME_CONTRACT_STATUS == "Active" and MONTHS_BALANCE == -1) ')[['SK_ID_PREV','SK_ID_CURR','NAME_CONTRACT_STATUS']].drop_duplicates()
    NAME_CONTRACT_STATUS_ctab = pd.crosstab(NAME_CONTRACT_STATUS['SK_ID_CURR'], NAME_CONTRACT_STATUS['NAME_CONTRACT_STATUS'])

    from functools import reduce
    dfs = [NAME_CONTRACT_STATUS_ctab,Active,Closed]
    df_final = reduce(lambda left,right: pd.merge(left,right,on='SK_ID_CURR',how='outer'), dfs)
    df_final.reset_index(inplace=True)
    df_final.columns = ['SK_ID_CURR','Active','Completed','MONTHS_BALANCE_A_min','CNT_INSTALMENT_A_min','CNT_INSTALMENT_A_max','CNT_INSTALMENT_FUTURE_A_min',
                        'CNT_INSTALMENT_FUTURE_max','MONTHS_BALANCE_C_min','MONTHS_BALANCE_C_max','CNT_INSTALMENT_C_max']
    return(df_final)

def transform_POS_CASH_balance(df):
    sqrtTransformation = ['CNT_INSTALMENT_A_min','CNT_INSTALMENT_A_max','CNT_INSTALMENT_FUTURE_A_min','CNT_INSTALMENT_FUTURE_max','CNT_INSTALMENT_C_max']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)
    return(df)
def get_installment_payments():
    instalment_payments = pd.read_csv('../input/installments_payments.csv')
    instalment_payments['MONTH']=(instalment_payments['DAYS_INSTALMENT']/30).astype(int)
    # features for last month active loans
    Active = instalment_payments.query('MONTH == -1').groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION':max,
        'NUM_INSTALMENT_NUMBER':max,
        'AMT_INSTALMENT':sum,
        'AMT_PAYMENT':sum
    })
    Closed = instalment_payments.groupby('SK_ID_CURR').agg({
        'NUM_INSTALMENT_VERSION':max,
        'NUM_INSTALMENT_NUMBER':max,
        'DAYS_INSTALMENT':min,
        'AMT_INSTALMENT':[max,min]
    })
    from functools import reduce
    df_final = pd.merge(Active,Closed,on='SK_ID_CURR',how='outer')
    df_final.reset_index(inplace=True)
    df_final.columns=['SK_ID_CURR','NUM_INSTALMENT_VERSION_A_max','NUM_INSTALMENT_NUMBER_A_max','AMT_INSTALMENT_A_sum','AMT_PAYMENT_A_sum',
                      'NUM_INSTALMENT_VERSION_C_max','NUM_INSTALMENT_NUMBER_C_max','DAYS_INSTALMENT_C_min','AMT_INSTALMENT_C_max','AMT_INSTALMENT_c_min']
    return(df_final)

def transform_installment_payments(df):
    logTransformation = ['AMT_INSTALMENT_A_sum','AMT_PAYMENT_A_sum','AMT_INSTALMENT_C_max','AMT_INSTALMENT_c_min']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.log(x+1),axis=1)
    
    sqrtTransformation = ['NUM_INSTALMENT_VERSION_A_max','NUM_INSTALMENT_NUMBER_A_max','NUM_INSTALMENT_VERSION_C_max','NUM_INSTALMENT_NUMBER_C_max',
                          'DAYS_INSTALMENT_C_min']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)
    return(df)
def get_credit_card_balance():
    df = pd.read_csv('../input/credit_card_balance.csv')
    dfa = df.query('NAME_CONTRACT_STATUS == "Active"').groupby(['SK_ID_CURR','MONTHS_BALANCE']).agg({
        'AMT_BALANCE':sum,
        'AMT_CREDIT_LIMIT_ACTUAL':sum,
        'AMT_DRAWINGS_ATM_CURRENT':sum,
        'AMT_DRAWINGS_CURRENT':sum,
        'AMT_DRAWINGS_OTHER_CURRENT':sum,
        'AMT_DRAWINGS_POS_CURRENT':sum,
        'AMT_INST_MIN_REGULARITY':sum,
        'AMT_PAYMENT_CURRENT':sum,
        'AMT_PAYMENT_TOTAL_CURRENT':sum,
        'AMT_RECEIVABLE_PRINCIPAL':sum,
        'AMT_RECIVABLE':sum,
        'AMT_TOTAL_RECEIVABLE':sum,
        'CNT_DRAWINGS_ATM_CURRENT':sum,
        'CNT_DRAWINGS_CURRENT':sum,
        'CNT_DRAWINGS_POS_CURRENT':sum,
        'CNT_DRAWINGS_OTHER_CURRENT':sum,
        'CNT_INSTALMENT_MATURE_CUM':sum,
        'SK_DPD':max,
        'SK_DPD_DEF':max
    })
    dfa = dfa.groupby(['SK_ID_CURR']).agg(['mean',max,min,'std'])
    dfa.columns = ["_".join(x) for x in dfa.columns.ravel()]
    dfp = pd.pivot_table(df.query('NAME_CONTRACT_STATUS == ["Active","Completed","Demand"]').groupby(['SK_ID_CURR','SK_ID_PREV']).agg({'MONTHS_BALANCE':max}).merge(df[['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE','NAME_CONTRACT_STATUS']],on=['SK_ID_CURR','SK_ID_PREV','MONTHS_BALANCE']),
                values='SK_ID_PREV',index=['SK_ID_CURR'],columns=['NAME_CONTRACT_STATUS'],aggfunc=np.size).reset_index().rename_axis(None, axis=1)
    dfp.fillna(0,inplace=True)
    dfcc = dfa.merge(dfp,on='SK_ID_CURR')
    dfcc[dfcc < 0] = 0
    return(dfcc)

def transform_credit_card_balance(df):
    logTransformation = ['AMT_BALANCE_mean','AMT_BALANCE_max','AMT_BALANCE_min','AMT_BALANCE_std',
                         'AMT_CREDIT_LIMIT_ACTUAL_mean','AMT_CREDIT_LIMIT_ACTUAL_max','AMT_CREDIT_LIMIT_ACTUAL_min','AMT_CREDIT_LIMIT_ACTUAL_std',
                         'AMT_DRAWINGS_ATM_CURRENT_mean','AMT_DRAWINGS_ATM_CURRENT_max','AMT_DRAWINGS_ATM_CURRENT_min','AMT_DRAWINGS_ATM_CURRENT_std',
                         'AMT_DRAWINGS_CURRENT_mean','AMT_DRAWINGS_CURRENT_max','AMT_DRAWINGS_CURRENT_min','AMT_DRAWINGS_CURRENT_std',
                         'AMT_DRAWINGS_OTHER_CURRENT_mean','AMT_DRAWINGS_OTHER_CURRENT_max','AMT_DRAWINGS_OTHER_CURRENT_std','AMT_DRAWINGS_POS_CURRENT_mean',
                         'AMT_DRAWINGS_POS_CURRENT_max','AMT_DRAWINGS_POS_CURRENT_min','AMT_DRAWINGS_POS_CURRENT_std','AMT_INST_MIN_REGULARITY_mean',
                         'AMT_INST_MIN_REGULARITY_max','AMT_INST_MIN_REGULARITY_min','AMT_INST_MIN_REGULARITY_std','AMT_PAYMENT_CURRENT_mean',
                         'AMT_PAYMENT_CURRENT_max','AMT_PAYMENT_CURRENT_min','AMT_PAYMENT_CURRENT_std','AMT_PAYMENT_TOTAL_CURRENT_mean','AMT_PAYMENT_TOTAL_CURRENT_max',
                         'AMT_PAYMENT_TOTAL_CURRENT_min','AMT_PAYMENT_TOTAL_CURRENT_std','AMT_RECEIVABLE_PRINCIPAL_mean','AMT_RECEIVABLE_PRINCIPAL_max','AMT_RECEIVABLE_PRINCIPAL_min',
                         'AMT_RECEIVABLE_PRINCIPAL_std','AMT_RECIVABLE_mean','AMT_RECIVABLE_max','AMT_RECIVABLE_min','AMT_RECIVABLE_std','AMT_TOTAL_RECEIVABLE_mean',
                         'AMT_TOTAL_RECEIVABLE_max','AMT_TOTAL_RECEIVABLE_min','AMT_TOTAL_RECEIVABLE_std']
    df[logTransformation] = df[logTransformation].apply(lambda x : np.log(x+1),axis=1)
    
    sqrtTransformation = ['CNT_DRAWINGS_ATM_CURRENT_mean','CNT_DRAWINGS_ATM_CURRENT_max','CNT_DRAWINGS_ATM_CURRENT_min',
                          'CNT_DRAWINGS_ATM_CURRENT_std','CNT_DRAWINGS_CURRENT_mean','CNT_DRAWINGS_CURRENT_max','CNT_DRAWINGS_CURRENT_min',
                          'CNT_DRAWINGS_CURRENT_std','CNT_DRAWINGS_POS_CURRENT_mean','CNT_DRAWINGS_POS_CURRENT_max','CNT_DRAWINGS_POS_CURRENT_min',
                          'CNT_DRAWINGS_POS_CURRENT_std','CNT_INSTALMENT_MATURE_CUM_mean','CNT_INSTALMENT_MATURE_CUM_max','CNT_INSTALMENT_MATURE_CUM_min',
                          'CNT_INSTALMENT_MATURE_CUM_std','SK_DPD_mean','SK_DPD_std','SK_DPD_DEF_mean','SK_DPD_DEF_max','SK_DPD_DEF_std']
    df[sqrtTransformation] = df[sqrtTransformation].apply(lambda x: np.sqrt(np.abs(x)),axis=1)
    return(df)
def getFinalDataSet():
    application = get_application_dataset()
    application = transform_application(get_application_dataset())
    bureau = transform_bureau(get_bureau_dataset())
    previous_application = get_previous_application()
    #previous_application = transform_previous_application(get_previous_application())
    POS_CASH_balance = transform_POS_CASH_balance(get_POS_CASH_balance())
    installment_payments = transform_installment_payments(get_installment_payments()) 
    credit_card_balance = transform_credit_card_balance(get_credit_card_balance())
    dfs = [application, bureau, previous_application, POS_CASH_balance, installment_payments, credit_card_balance]
    from functools import reduce
    df = reduce(lambda left,right: pd.merge(left,right,on='SK_ID_CURR',how='left'), dfs)
    return(df)
def scaleNfillna(df):
    df.replace([np.inf, -np.inf], np.nan,inplace=True)
    df.fillna(0,inplace=True)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    df = scaler.fit_transform(df)
    return(df)
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn import metrics
def kfold_lightgbm(df, num_folds, stratified = False, debug= False):
    from lightgbm import LGBMClassifier
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.model_selection import KFold, StratifiedKFold
    # Divide in training/validation and test data
    train_df = df[df['TARGET'].notnull()]
    test_df = df[df['TARGET'].isnull()]
    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))
    del df
    gc.collect()
    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=1001)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=1001)
    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in ['SK_ID_CURR','TARGET']]
    
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['TARGET'])):
        train_x, train_y= train_df[feats].iloc[train_idx], train_df['TARGET'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['TARGET'].iloc[valid_idx]

        # LightGBM parameters found by Bayesian optimization
        clf = LGBMClassifier(
            nthread=4,
            n_estimators=10000,
            learning_rate=0.02,
            num_leaves=34,
            colsample_bytree=0.9497036,
            subsample=0.8715623,
            max_depth=8,
            reg_alpha=0.041545473,
            reg_lambda=0.0735294,
            min_split_gain=0.0222415,
            min_child_weight=39.3259775,
            silent=-1,
            verbose=-1,
           # scale_pos_weight=1,
        )

        clf.fit(train_x, train_y, eval_set=[(train_x, train_y), (valid_x, valid_y)], 
            eval_metric= 'auc', verbose= 100, early_stopping_rounds= 200)

        oof_preds[valid_idx] = clf.predict_proba(valid_x, num_iteration=clf.best_iteration_)[:, 1]
        sub_preds += clf.predict_proba(test_df[feats], num_iteration=clf.best_iteration_)[:, 1] / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = clf.feature_importances_
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()
    if not debug:
        test_df['TARGET'] = sub_preds
    print('Full AUC score %.6f' % roc_auc_score(train_df['TARGET'], oof_preds))
    # Write submission file and plot feature importance
    display_importances(feature_importance_df)
    #return submission
    return(test_df[['SK_ID_CURR', 'TARGET']])

def display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]
    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances01.png')
def ANN(X_train,y_train,X_test,y_test,L_dim,num_epochs = 2):
    
    #model
    ann = Sequential()
    ann.add(Dense(L_dim[0], input_dim=X_train.shape[1], activation='relu'))
    ann.add(BatchNormalization())
    ann.add(Dropout(0.2))
    ann.add(Dense(L_dim[1], activation='relu'))
    ann.add(BatchNormalization())
    ann.add(Dropout(0.2))
    ann.add(Dense(L_dim[2], activation='relu'))
    ann.add(BatchNormalization())
    ann.add(Dropout(0.2))
    ann.add(Dense(L_dim[3], activation='relu'))
    ann.add(BatchNormalization())
    ann.add(Dropout(0.2))
    ann.add(Dense(L_dim[4], activation='relu'))
    ann.add(BatchNormalization())
    ann.add(Dense(1, activation='sigmoid'))

    ann.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    ann.summary()
    
    #training
    ann.fit(X_train, 
          y_train, 
          epochs=num_epochs,
          batch_size=32,
          validation_data=(X_test,y_test),
          shuffle=True,
          verbose=1)
    #accuracy
    from sklearn import metrics
    y_pred = ann.predict(X_test)
    cm = metrics.confusion_matrix(y_test, y_pred > 0.5)
    print(cm)
    #roc
    fpr, tpr, thresholds = metrics.roc_curve(y_test+1, y_pred, pos_label=2)
    print(metrics.auc(fpr, tpr))
    return(ann) 
def AE(X):
    input_data = Input(shape=(X.shape[1],))
    encoded = Dense(128, activation='relu')(input_data)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dense(16, activation='relu')(encoded)
    encoded = BatchNormalization(name='encoded_layer')(encoded)

    decoded = Dense(32, activation='relu')(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = BatchNormalization()(decoded)
    #decoded = Dense(237, activation='sigmoid')(encoded)
    decoded = Dense(X.shape[1], activation='linear')(encoded)

    autoencoder = Model(input_data, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    
    autoencoder.fit(X, X,epochs=10, batch_size=32,shuffle=True)
    return(autoencoder)
def submitLGBM(debug=True):
    df = getFinalDataSet()
    submission = kfold_lightgbm(df, 4, stratified = True)
    if not debug:
        print("writing the submission file")
        submission.to_csv('submission_1.csv', index=False)
def ANN_prediction(df):
    feats =[x for x in list(df) if x not in ['SK_ID_CURR','TARGET']]
    df[feats] = scaleNfillna(df[feats])
    X = df.loc[df['TARGET'].notnull(),feats].values
    y = df[df['TARGET'].notnull()].TARGET.values
    # train test splitting

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

    # up sampling
    ros = RandomOverSampler(random_state=0, sampling_strategy=0.4)
    X_resampled, y_resampled = ros.fit_sample(X_train, y_train)
    L_dim = (128,64,32,16,8)
    ann = ANN(X_resampled, y_resampled,X_test,y_test,L_dim,5)
    annPred = pd.DataFrame()
    annPred['SK_ID_CURR'] = df['SK_ID_CURR']
    annPred['annPred'] = ann.predict(df[feats].values)
    return(annPred)
def AE_prediction(df):
    feats =[x for x in list(df) if x not in ['SK_ID_CURR','TARGET']]
    df.loc[:,feats] = scaleNfillna(df.loc[:,feats])
    X = df[feats].values
    ae = AE(X)
    intermediate_layer_model = Model(inputs=ae.input, outputs=ae.get_layer('encoded_layer').output)
    aePred = pd.DataFrame(columns=['SK_ID_CURR']+['ae'+str(x) for x in range(1,17)])
    aePred['SK_ID_CURR'] = df['SK_ID_CURR']
    aePred.loc[:,'ae1':'ae16'] = intermediate_layer_model.predict(X)
    return(aePred)

def submit():
    df = getFinalDataSet()
    annPred = ANN_prediction(df)
    df = pd.merge(df,annPred,on='SK_ID_CURR',how='left')
    #aePred = AE_prediction(df)
    #df = pd.merge(df,aePred,on='SK_ID_CURR',how='left')
    submission = kfold_lightgbm(df, 4, stratified = True)
    print("writing the submission file")
    submission.to_csv('submission_1.csv', index=False)
    
def submitWeighted():
    df = getFinalDataSet()
    annPred = ANN_prediction(df)
    lgbPred = kfold_lightgbm(df, 4, stratified = True)
    weighted = pd.merge(lgbPred,annPred,on='SK_ID_CURR',how='left')
    weighted['TARGET_avg'] = (0.8)*weighted['TARGET'] + (0.2)*weighted['annPred']
    submission = weighted.loc[:,['SK_ID_CURR','TARGET_avg']]
    submission.rename(columns={'TARGET_avg': 'TARGET'}, inplace=True)
    submission.to_csv('submission_1.csv', index=False)
def ilo():
    df = getFinalDataSet()

    feats =[x for x in list(df) if x not in ['SK_ID_CURR','TARGET']]
    df[feats] = scaleNfillna(df[feats])
    X = df[feats].values
    ae = AE(X)
    intermediate_layer_model = Model(inputs=ae.input, outputs=ae.get_layer('encoded_layer').output)
    X = intermediate_layer_model.predict(X[df["TARGET"].notnull()])
    y = df[df['TARGET'].notnull()].TARGET.values

    kfold_lightgbm(X,y, 2, False, False)
#submit()
#submitLGBM(debug=False)
#submitWeighted()
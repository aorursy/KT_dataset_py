import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
print('\n')
print(os.listdir("../working"))
print(os.listdir("../"))
### set dataframe display settigns ###
pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.float_format','{:,.2f}'.format)
train = pd.read_csv('../input/application_train.csv')
test = pd.read_csv('../input/application_test.csv')
bureau = pd.read_csv('../input/bureau.csv')
bureau_balance = pd.read_csv('../input/bureau_balance.csv')
previous = pd.read_csv('../input/previous_application.csv')
pos = pd.read_csv('../input/POS_CASH_balance.csv')
installments = pd.read_csv('../input/installments_payments.csv')
cc = pd.read_csv('../input/credit_card_balance.csv')
### Check Data Shape ###
print('train:', train.shape)
print('test:', test.shape)
print('bureau:', bureau.shape)
print('bureau_balance:', bureau_balance.shape)
print('previous:', previous.shape)
print('pos:', pos.shape)
print('installments:', installments.shape)
print('cc:', cc.shape)
train.head()
### Append ###
df = train.append(test)
print(df.shape)
print('Check:', train.shape[0]+test.shape[0]==df.shape[0])
### Concat ###
df1 = pd.concat([train,test],axis=0)
print(df1.shape)
### Merge ###
df2 = train.merge(test,on='SK_ID_CURR',how='outer')
print(df2.shape)
df2.head()
del df1,df2
### Define Replace Function ###
def replace(df,col,pre_value,after_value):
    df[col].replace(pre_value,after_value,inplace=True)
### Define Absolute Function ###
def abs_func(df,col):
    df[col] = abs(df[col])
### Define Drop Function ###
def drop(df,col):
    df.drop(columns=[col],axis=1,inplace=True)
### Define KDE plot Function ###
def kde_plot(df,var_name):
    plt.figure(figsize=(10,6))
    sns.kdeplot(df[var_name])
    plt.xlabel(var_name);plt.ylabel('Density');plt.title('%s Distribution'%var_name)
### Defind Tukey IQR Function ###
def find_outliers_tukey(x):
    q1 = np.percentile(x,10)
    q3 = np.percentile(x,90)
    iqr = q3 - q1
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x < floor) | (x > ceiling)])
    outlier_values = list(x[outlier_indices])
    return outlier_indices, outlier_values
### CODE_GENDER ###
df['CODE_GENDER'].value_counts()
#replace(df,'CODE_GENDER','XNA',np.nan)
df['CODE_GENDER'].value_counts()
### AMT_INCOME_TOTAL ###
df['AMT_INCOME_TOTAL'].describe()
df.loc[df['AMT_INCOME_TOTAL']>10000000,'AMT_INCOME_TOTAL']
df.loc[df['AMT_INCOME_TOTAL']>100000000,'AMT_INCOME_TOTAL'] = np.nan
### REGION_POPULATION_RELATIVE ###
df['REGION_POPULATION_RELATIVE'].describe()
### DAYS_BIRTH ###
df['DAYS_BIRTH'].describe()
abs_func(df,'DAYS_BIRTH')
### DAYS_EMPLOYED ###
df['DAYS_EMPLOYED'].describe()
np.percentile(df['DAYS_EMPLOYED'],85)
print(df.shape)
print(len(df[df['DAYS_EMPLOYED']>0]))
print(len(df[df['DAYS_EMPLOYED']==365243]))
kde_plot(df,'DAYS_EMPLOYED')
replace(df,'DAYS_EMPLOYED',365243,np.nan)
df['DAYS_EMPLOYED'].describe()
abs_func(df,'DAYS_EMPLOYED')
df['DAYS_EMPLOYED'].describe()
### DAYS_REGISTRATION ###
abs_func(df,'DAYS_REGISTRATION')
df['DAYS_REGISTRATION'].describe()
### DAYS_ID_PUBLISH ###
abs_func(df,'DAYS_ID_PUBLISH')
### OWN_CAR_AGE ###
df['OWN_CAR_AGE'].describe() # 91 years old car!
kde_plot(df,'OWN_CAR_AGE')
### FLAG_PHONE ###
drop(df,'FLAG_PHONE')
### REGION_RATING_CLIENT_W_CITY ###
df['REGION_RATING_CLIENT_W_CITY'].value_counts()
### 不能删除这一行，是测试集！！！ ###
#df = df[df['REGION_RATING_CLIENT_W_CITY'] != -1]
### ORGANIZATION_TYPE ###
df['ORGANIZATION_TYPE'].value_counts()
### EXT_SOURCE_1 ###
df['EXT_SOURCE_1'].describe()
### APARTMENTS_AVG ###
df['APARTMENTS_AVG'].describe()
### OBS_30_CNT_SOCIAL_CIRCLE ###
df['OBS_30_CNT_SOCIAL_CIRCLE'].describe()
print(df.shape)
print(len(df[df['OBS_30_CNT_SOCIAL_CIRCLE']>100]))
df.loc[df['OBS_30_CNT_SOCIAL_CIRCLE']>100,'OBS_30_CNT_SOCIAL_CIRCLE'] = np.nan
df[df['OBS_30_CNT_SOCIAL_CIRCLE']>100]['OBS_30_CNT_SOCIAL_CIRCLE'] 
kde_plot(df,'OBS_30_CNT_SOCIAL_CIRCLE')
### OBS_60_CNT_SOCIAL_CIRCLE ###
df['OBS_60_CNT_SOCIAL_CIRCLE'].describe()
df.loc[df['OBS_60_CNT_SOCIAL_CIRCLE']>100,'OBS_60_CNT_SOCIAL_CIRCLE']
df.loc[df['OBS_60_CNT_SOCIAL_CIRCLE']>100,'OBS_60_CNT_SOCIAL_CIRCLE'] = np.nan
### DAYS_LAST_PHONE_CHANGE ###
abs_func(df,'DAYS_LAST_PHONE_CHANGE')
### AMT_REQ_CREDIT_BUREAU_QRT ###
df['AMT_REQ_CREDIT_BUREAU_QRT'].describe()
df.loc[df['AMT_REQ_CREDIT_BUREAU_QRT']>10,'AMT_REQ_CREDIT_BUREAU_QRT']
kde_plot(df,'AMT_REQ_CREDIT_BUREAU_QRT')
df.loc[df['AMT_REQ_CREDIT_BUREAU_QRT']>100,'AMT_REQ_CREDIT_BUREAU_QRT'] = np.nan
df.to_csv('df.csv',index=False)
### DAYS_CREDIT ###
abs_func(bureau,'DAYS_CREDIT')
### CREDIT_DAY_OVERDUE ###
bureau['CREDIT_DAY_OVERDUE'].describe()
kde_plot(bureau,'CREDIT_DAY_OVERDUE')
bureau.loc[bureau['CREDIT_DAY_OVERDUE']>1000,'CREDIT_DAY_OVERDUE']
### DAYS_CREDIT_ENDDATE ###
bureau['DAYS_CREDIT_ENDDATE'].describe()
kde_plot(bureau,'DAYS_CREDIT_ENDDATE')
indices, values = find_outliers_tukey(bureau['DAYS_CREDIT_ENDDATE'].fillna(bureau['DAYS_CREDIT_ENDDATE'].mean()))
print(bureau.shape)
print(len(values))
print(values)
len(bureau.loc[bureau['DAYS_CREDIT_ENDDATE']>30000,'DAYS_CREDIT_ENDDATE'])
len(bureau.loc[bureau['DAYS_CREDIT_ENDDATE']<-10000,'DAYS_CREDIT_ENDDATE'])
bureau.loc[bureau['DAYS_CREDIT_ENDDATE']<-20000,'DAYS_CREDIT_ENDDATE']
bureau.loc[bureau['DAYS_CREDIT_ENDDATE']<-20000,'DAYS_CREDIT_ENDDATE'] = np.nan
### DAYS_ENDDATE_FACT ###
bureau['DAYS_ENDDATE_FACT'].describe()
bureau.loc[bureau['DAYS_ENDDATE_FACT']<-4000,'DAYS_ENDDATE_FACT']
indices, values = find_outliers_tukey(bureau['DAYS_ENDDATE_FACT'].fillna(bureau['DAYS_ENDDATE_FACT'].mean()))
print(bureau.shape)
print(len(values))
print(values)
bureau.loc[bureau['DAYS_ENDDATE_FACT']<-4000,'DAYS_ENDDATE_FACT'] = np.nan
### AMT_CREDIT_MAX_OVERDUE ###
bureau['AMT_CREDIT_MAX_OVERDUE'].describe()
np.percentile(bureau['AMT_CREDIT_MAX_OVERDUE'].fillna(bureau['AMT_CREDIT_MAX_OVERDUE'].mean()),97)
len(bureau.loc[bureau['AMT_CREDIT_MAX_OVERDUE']>10000000,'AMT_CREDIT_MAX_OVERDUE'])
bureau.loc[bureau['AMT_CREDIT_MAX_OVERDUE']>10000000,'AMT_CREDIT_MAX_OVERDUE'] = np.nan
### AMT_CREDIT_SUM ###
bureau['AMT_CREDIT_SUM'].describe()
np.percentile(bureau['AMT_CREDIT_SUM'].fillna(bureau['AMT_CREDIT_SUM'].mean()),95)
len(bureau.loc[bureau['AMT_CREDIT_SUM']>10000000,'AMT_CREDIT_SUM'])
bureau.loc[bureau['AMT_CREDIT_SUM']>10000000,'AMT_CREDIT_SUM'] = np.nan
### AMT_CREDIT_SUM_DEBT ###
bureau['AMT_CREDIT_SUM_DEBT'].describe()
len(bureau.loc[bureau['AMT_CREDIT_SUM_DEBT']<0,'AMT_CREDIT_SUM_DEBT'])
bureau.loc[bureau['AMT_CREDIT_SUM_DEBT']<0,'AMT_CREDIT_SUM_DEBT'] = 0
np.percentile(bureau['AMT_CREDIT_SUM_DEBT'].fillna(bureau['AMT_CREDIT_SUM_DEBT'].mean()),95)
len(bureau.loc[bureau['AMT_CREDIT_SUM_DEBT']>50000000,'AMT_CREDIT_SUM_DEBT'])
bureau.loc[bureau['AMT_CREDIT_SUM_DEBT']>50000000,'AMT_CREDIT_SUM_DEBT'] = np.nan
### AMT_CREDIT_SUM_LIMIT ###
bureau['AMT_CREDIT_SUM_LIMIT'].describe()
### AMT_ANNUITY ###
bureau['AMT_ANNUITY'].describe()
len(bureau.loc[bureau['AMT_ANNUITY']>10000000,'AMT_ANNUITY'])
bureau.loc[bureau['AMT_ANNUITY']>10000000,'AMT_ANNUITY'] = np.nan
bureau.to_csv('bureau.csv',index=False)
### MONTHS_BALANCE ###
abs_func(bureau_balance,'MONTHS_BALANCE')
### Save DataFrame ###
bureau_balance.to_csv('bureau_balance.csv',index=False)
### AMT_DOWN_PAYMENT ###
previous['AMT_DOWN_PAYMENT'].describe()
previous.loc[previous['AMT_DOWN_PAYMENT']<0,'AMT_DOWN_PAYMENT'] = 0
### NFLAG_MICRO_CASH ###
# No such Variable
### NAME_CASH_LOAN_PURPOSE ###
previous['NAME_CASH_LOAN_PURPOSE'].value_counts()
### DAYS_DECISION ###
abs_func(previous,'DAYS_DECISION')
### CODE_REJECT_REASON ###
previous['CODE_REJECT_REASON'].value_counts()
### NAME_CLIENT_TYPE ###
previous['NAME_CLIENT_TYPE'].value_counts()
### SELLERPLACE_AREA ###
previous['SELLERPLACE_AREA'].describe()
print(len(previous.loc[previous['SELLERPLACE_AREA']>100000,'SELLERPLACE_AREA']))
print(len(previous.loc[previous['SELLERPLACE_AREA']==0,'SELLERPLACE_AREA']))
previous.loc[previous['SELLERPLACE_AREA']<0,'SELLERPLACE_AREA'] = 0
kde_plot(previous,'SELLERPLACE_AREA')
### DAYS_FIRST_DRAWING ###
previous['DAYS_FIRST_DRAWING'].describe()
len(previous.loc[previous['DAYS_FIRST_DRAWING']==365243,'DAYS_FIRST_DRAWING'])
previous.loc[previous['DAYS_FIRST_DRAWING']==365243,'DAYS_FIRST_DRAWING'] = np.nan
abs_func(previous,'DAYS_FIRST_DRAWING')
### DAYS_FIRST_DUE ###
previous['DAYS_FIRST_DUE'].describe()
print(len(previous.loc[previous['DAYS_FIRST_DUE']==365243,'DAYS_FIRST_DUE']))
print(len(previous.loc[(previous['DAYS_FIRST_DUE']<365243)&(previous['DAYS_FIRST_DUE']>0),'DAYS_FIRST_DUE']))
previous.loc[previous['DAYS_FIRST_DUE']==365243,'DAYS_FIRST_DUE'] = np.nan
abs_func(previous,'DAYS_FIRST_DUE')
### DAYS_LAST_DUE_1ST_VERSION ###
previous.loc[previous['DAYS_LAST_DUE_1ST_VERSION']==365243,'DAYS_LAST_DUE_1ST_VERSION'] = np.nan
abs_func(previous,'DAYS_LAST_DUE_1ST_VERSION')
### DAYS_LAST_DUE ###
previous['DAYS_LAST_DUE'].describe()
print(len(previous.loc[(previous['DAYS_LAST_DUE']<365243)&(previous['DAYS_LAST_DUE']>0),'DAYS_LAST_DUE']))
previous.loc[previous['DAYS_LAST_DUE']==365243,'DAYS_LAST_DUE'] = np.nan
abs_func(previous,'DAYS_LAST_DUE')
### DAYS_TERMINATION ###
previous['DAYS_TERMINATION'].describe()
previous.loc[previous['DAYS_TERMINATION']==365243,'DAYS_TERMINATION'] = np.nan
abs_func(previous,'DAYS_TERMINATION')
previous.to_csv('previous.csv',index=False)
### MONTHS_BALANCE ###
abs_func(pos,'MONTHS_BALANCE')
### Save DataFrame ###
pos.to_csv('pos.csv',index=False)
### NUM_INSTALMENT_VERSION ###
installments['NUM_INSTALMENT_VERSION'].describe()
np.percentile(installments['NUM_INSTALMENT_VERSION'],90)
len(installments.loc[installments['NUM_INSTALMENT_VERSION']>100,'NUM_INSTALMENT_VERSION'])
installments.loc[installments['NUM_INSTALMENT_VERSION']>50,'NUM_INSTALMENT_VERSION']
kde_plot(installments,'NUM_INSTALMENT_VERSION')
### DAYS_INSTALMENT ###
abs_func(installments,'DAYS_INSTALMENT')
### DAYS_ENTRY_PAYMENT ###
abs_func(installments,'DAYS_ENTRY_PAYMENT')
### AMT_INSTALMENT ###
installments['AMT_INSTALMENT'].describe()
kde_plot(installments,'AMT_INSTALMENT')
len(installments.loc[installments['AMT_INSTALMENT']>1000000,'AMT_INSTALMENT'])
### Save DataFrame ###
installments.to_csv('installments.csv',index=False)
### MONTHS_BALANCE ###
abs_func(cc,'MONTHS_BALANCE')
### AMT_BALANCE ###
cc['AMT_BALANCE'].describe()
len(cc.loc[cc['AMT_BALANCE']<0,'AMT_BALANCE'])
kde_plot(cc,'AMT_BALANCE')
### AMT_DRAWINGS_ATM_CURRENT ###
cc['AMT_DRAWINGS_ATM_CURRENT'].describe()
len(cc.loc[cc['AMT_DRAWINGS_ATM_CURRENT']<0,'AMT_DRAWINGS_ATM_CURRENT'])
cc.loc[cc['AMT_DRAWINGS_ATM_CURRENT']<0,'AMT_DRAWINGS_ATM_CURRENT'] = np.nan
kde_plot(cc,'AMT_DRAWINGS_ATM_CURRENT')
### AMT_DRAWINGS_CURRENT ###
cc['AMT_DRAWINGS_CURRENT'].describe()
len(cc.loc[cc['AMT_DRAWINGS_CURRENT']<0,'AMT_DRAWINGS_CURRENT'])
cc.loc[cc['AMT_DRAWINGS_CURRENT']<0,'AMT_DRAWINGS_CURRENT'] = np.nan
### AMT_RECEIVABLE_PRINCIPAL ###
cc['AMT_RECEIVABLE_PRINCIPAL'].describe()
len(cc.loc[cc['AMT_RECEIVABLE_PRINCIPAL']<0,'AMT_RECEIVABLE_PRINCIPAL'])
kde_plot(cc,'AMT_RECEIVABLE_PRINCIPAL')
#cc.loc[cc['AMT_RECEIVABLE_PRINCIPAL']<0,'AMT_RECEIVABLE_PRINCIPAL'] = np.nan
### AMT_RECIVABLE ###
print(len(cc.loc[cc['AMT_RECIVABLE']<0,'AMT_RECIVABLE']))
#cc.loc[cc['AMT_RECIVABLE']<0,'AMT_RECIVABLE'] = np.nan
### Sava DataFrame ###
cc.to_csv('cc.csv',index=False)
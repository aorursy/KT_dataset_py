import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
df = pd.read_csv('../input/application_train.csv')

sns.set(style="ticks", color_codes=True)
#plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
#Lets see defaulters by gender
sns.countplot(df[df.TARGET == 1].CODE_GENDER)
#Lets see defaulters by car ownership
sns.countplot(df[df.TARGET == 1].FLAG_OWN_CAR)

#Lets see defaulters by real estate ownership
sns.countplot(df[df.TARGET == 1].FLAG_OWN_REALTY)
#Lets see defaulters by loan type
sns.countplot(df[df.TARGET == 1].NAME_CONTRACT_TYPE)
#Is the same distribution obtained for non-defaulters as well?
sns.countplot(df[df.TARGET == 0].NAME_CONTRACT_TYPE)
#Lets see defaulters by suite type
plt.figure(figsize=(25, 25))
sns.countplot(df[df.TARGET == 1].NAME_TYPE_SUITE)
#Lets see defaulters by children-count
sns.countplot(df[df.TARGET == 1].CNT_CHILDREN)
#Is the same pattern obserbed for non-defaulters as well?
sns.countplot(df[df.TARGET == 0].CNT_CHILDREN)
#Histogram showing the age distribution of the clients
sns.distplot(df.DAYS_BIRTH)
#Defaulters by occupation
plt.figure(figsize=(25, 25))
sns.countplot(df[df.TARGET == 1].OCCUPATION_TYPE)
#Defaulters by family-size
plt.figure(figsize=(25, 25))
sns.countplot(df[df.TARGET == 1].CNT_FAM_MEMBERS)

#Is the same distribution observed for non-defaulters as well?
plt.figure(figsize=(25, 25))
sns.countplot(df[df.TARGET == 0].CNT_FAM_MEMBERS)

#Defaulters arranged by education type and credit range
sns.catplot(y="AMT_CREDIT", x="NAME_EDUCATION_TYPE", kind="box", data=df[df.TARGET == 1], height=6,aspect=2)
#Is the same pattern observed for non-defaulters as well?
sns.catplot(y="AMT_CREDIT", x="NAME_EDUCATION_TYPE", kind="box", data=df[df.TARGET == 0], height=6,aspect=2)
#'Businessman' never default on credit
plt.figure(figsize=(16,4))
sns.countplot(x="TARGET", data=df[df.NAME_INCOME_TYPE == "Businessman"])
#Defaulters arranged by income type and credit range
sns.catplot(y="AMT_CREDIT", x="NAME_INCOME_TYPE", kind="box", data=df[df.TARGET == 1], height=6,aspect=2)
#Overall distribution by education type
plt.figure(figsize=(16,8))
sns.countplot(x="NAME_EDUCATION_TYPE", data=df)
#Overall distribution by occupation type
plt.figure(figsize=(20,20))
sns.countplot(x="OCCUPATION_TYPE", data=df)
#Applications by day of the week(defaulters)
plt.figure(figsize=(20,20))
sns.countplot(x="WEEKDAY_APPR_PROCESS_START", data=df[df.TARGET == 1])
#Applications by time of the day(defaulters)
plt.figure(figsize=(10,10))
sns.distplot(df['HOUR_APPR_PROCESS_START'][df.TARGET == 1])
#Distribution by 'How many days before application did client change phone'(non-defaulters)
plt.figure(figsize=(10,10))
sns.distplot(df['DAYS_LAST_PHONE_CHANGE'][(df.TARGET == 0)].dropna())

#Is the same distribution observed for defaulters as well?
plt.figure(figsize=(10,10))
sns.distplot(df['DAYS_LAST_PHONE_CHANGE'][(df.TARGET == 1)].dropna())

#Defaulters by number of family members
plt.figure(figsize=(10,10))
sns.countplot(x="CNT_FAM_MEMBERS", data=df[df.TARGET == 1])
#Is the same pattern observed for non-defaulters as well?
plt.figure(figsize=(10,10))
sns.countplot(x="CNT_FAM_MEMBERS", data=df[df.TARGET == 0])
#Distribution of defaulters by the age of their cars
sns.distplot(df[df.TARGET == 1]['OWN_CAR_AGE'].dropna())
#Total distribution by total income
plt.figure(figsize=(10,10))
sns.distplot(df['AMT_INCOME_TOTAL'],bins = 50)

#Is the relationship between total income and credit amount linear?
sns.regplot(x="AMT_INCOME_TOTAL", y="AMT_CREDIT", data=df[df.TARGET == 0])
df_b = pd.read_csv('../input/bureau.csv')
df_b.columns
#Status of the Credit Bureau (CB) reported credits by active credit
sns.countplot(x="CREDIT_ACTIVE", data=df_b)
#Status of the Credit Bureau (CB) reported credits by credit currency
sns.countplot(x="CREDIT_CURRENCY", data=df_b)
#Status of the Credit Bureau (CB) by 'How many days before current application did client apply for Credit Bureau credit'
sns.distplot(df_b['DAYS_CREDIT'],bins =50)
#Status of the Credit Bureau (CB) by Credit type
plt.figure(figsize=(100,50))
sns.countplot(df_b.CREDIT_TYPE)
df_ccb = pd.read_csv('../input/credit_card_balance.csv')
#Distribution of credit card balance by 'Month of balance relative to application date'
plt.figure(figsize=(25,25))
sns.distplot(df_ccb.MONTHS_BALANCE)
#Distribution of credit card balance by Amount Balance
plt.figure(figsize=(25,25))
sns.distplot(df_ccb.AMT_BALANCE)
#Distribution of credit card balance by 'Amount drawing at ATM during the month of the previous credit'
plt.figure(figsize=(25,25))
sns.distplot(df_ccb.CNT_DRAWINGS_ATM_CURRENT.dropna())
#Distribution of credit card balance by 'Number of paid installments on the previous credit'
plt.figure(figsize=(25,25))
sns.distplot(df_ccb.CNT_INSTALMENT_MATURE_CUM.dropna())

#Distribution of credit card balance by 'DPD (Days past due) during the month on the previous credit' > 0
plt.figure(figsize=(25,25))
sns.distplot(df_ccb[df_ccb.SK_DPD > 0].SK_DPD)
df_pa = pd.read_csv('../input/previous_application.csv')
#Distribution of previous applications by 'application amount' < 3000000
plt.figure(figsize=(25,25))
sns.distplot(df_pa[df_pa.AMT_APPLICATION < 3000000].AMT_APPLICATION)
#Is the relationship between application amount and credit granted linear?
plt.figure(figsize=(10,10))
sns.scatterplot(x="AMT_APPLICATION", y="AMT_CREDIT", data=df_pa)
#Is the relationship between application amount and 'price of goods' linear?
plt.figure(figsize=(10,10))
sns.scatterplot(y="AMT_APPLICATION", x="AMT_GOODS_PRICE", data=df_pa)
#Distribution of application by days of the week
plt.figure(figsize=(25,25))
sns.countplot(df_pa.WEEKDAY_APPR_PROCESS_START)
df_ip = pd.read_csv('../input/installments_payments.csv')
#Is the relationship between application amount and credit granted linear?
plt.figure(figsize=(10,10))
sns.scatterplot(x="AMT_INSTALMENT", y="AMT_PAYMENT", data=df_ip)

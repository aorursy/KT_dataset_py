import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
df1=pd.read_csv('../input/bureau_balance.csv', chunksize=100)
for i in df1:
    df1=pd.DataFrame(i)
    break
print(df1)

sns.swarmplot(x="STATUS", y="MONTHS_BALANCE", data=df1)
df2=pd.read_csv('../input/installments_payments.csv', chunksize=20)
for i in df2:
    df2=pd.DataFrame(i)
    break
print(df2)
df2['days_difference']=df2['DAYS_ENTRY_PAYMENT']-df2['DAYS_INSTALMENT']
df2['AMT_DIFFERENCE']=df2['AMT_PAYMENT']-df2['AMT_INSTALMENT']
sns.barplot(x="AMT_DIFFERENCE", y="days_difference", hue='NUM_INSTALMENT_VERSION',data=df2)
plt.tight_layout()
df3 = pd.read_csv('../input/bureau.csv')
df3.groupby(['SK_ID_CURR','CREDIT_ACTIVE']).size()
df3.loc[df3['CNT_CREDIT_PROLONG']> 4 ]
df4 = df3.loc[df3['CNT_CREDIT_PROLONG'] >=1 ]


sns.barplot(x="CNT_CREDIT_PROLONG", y="AMT_CREDIT_SUM_DEBT", data=df4)
application = pd.read_csv('../input/application_train.csv', encoding='iso-8859-1')
# Selecting required columns
cols = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY', 
        'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON', 
        'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']

application.groupby("TARGET")[cols].max().transpose().plot(kind="barh", figsize=(10,5),width=.8)
plt.title("Maximum enquries made by defaulters and repayers")

application.groupby("TARGET")[cols].mean().transpose().plot(kind="barh", figsize=(10,5),width=.8)
plt.title("average enquries made by defaulters and repayers")

application.groupby("TARGET")[cols].std().transpose().plot(kind="barh", figsize=(10,5),width=.8)
plt.title("standard deviation in enquries made by defaulters and repayers")

plt.show() 
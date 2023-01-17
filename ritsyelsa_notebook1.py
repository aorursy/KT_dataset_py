import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
%matplotlib inline
master_data1=pd.read_csv('../input/application_train.csv',chunksize=10000)
for i in master_data1:
    master_data=pd.DataFrame(i)
    break
    
master_data.head()
master_data['NAME_CONTRACT_TYPE']=master_data['NAME_CONTRACT_TYPE'].astype('category')
target_type1=master_data[(master_data.TARGET==1)]
labels1=['Cash loans','Revolving loans']
sizes1=[target_type1['NAME_CONTRACT_TYPE'].value_counts()[0],
      target_type1['NAME_CONTRACT_TYPE'].value_counts()[1]]
explode = (0, 0.3)
plt.pie(sizes1, labels=labels1, explode=explode,
        autopct='%1.1f%%', shadow=True, startangle=90)
plt.title('DEFAULTERS V/S CONTRACT TYPES')
plt.axis('equal')
plt.show()
target_type0=master_data[(master_data.TARGET==0)]
labels0=['Cash loans','Revolving loans']
sizes0=[target_type0['NAME_CONTRACT_TYPE'].value_counts()[0],
      target_type0['NAME_CONTRACT_TYPE'].value_counts()[1]]
explode = (0, 0.3)
plt.pie(sizes0, labels=labels0, explode=explode,
        autopct='%1.3f%%', shadow=True, startangle=90)
plt.title('NON-DEFAULTERS V/S CONTRACT TYPES')
plt.axis('equal')
plt.show()

cont_type_rev=master_data[(master_data.NAME_CONTRACT_TYPE=='Revolving loans')]
labels_rev=['1','0']
sizes_rev=[cont_type_rev['TARGET'].value_counts()[0],
      cont_type_rev['TARGET'].value_counts()[1]]
explode = (0, 0.3)
plt.pie(sizes_rev, labels=labels_rev, explode=explode,
        autopct='%1.3f%%', shadow=True, startangle=90)
plt.title('REVOLVING LOANS V/S TARGET')
plt.axis('equal')
plt.show()
cont_type_cash=master_data[(master_data.NAME_CONTRACT_TYPE=='Cash loans')]
labels_cash=['1','0']
sizes_cash=[cont_type_cash['TARGET'].value_counts()[0],
      cont_type_cash['TARGET'].value_counts()[1]]
explode = (0, 0.3)
plt.pie(sizes_cash, labels=labels_cash, explode=explode,
        autopct='%1.3f%%', shadow=True, startangle=90)
plt.title('CASH LOANS V/S TARGET')
plt.axis('equal')
plt.show()
data=pd.read_csv('../input/bureau.csv')
beuro=pd.DataFrame(data)
beuro.fillna(0)
anuit=beuro.query('AMT_ANNUITY > 0')
plt.plot(anuit['AMT_ANNUITY'])
plt.show()
plt.pie([(beuro['AMT_CREDIT_MAX_OVERDUE']>0).sum(),(beuro['AMT_CREDIT_MAX_OVERDUE']==0).sum()],
        autopct='%.2f%%',shadow=True,
       startangle=90,labels=['Non-Defaulters','Defaulters'])
plt.axis('equal')
plt.show()
plt.pie([(beuro['DAYS_CREDIT']>-360).sum(),(beuro['DAYS_CREDIT']<-360).sum()],
        autopct='%.2f%%',shadow=True,
       startangle=90,labels=['Short term loans {<1 yr}','Long term loans'])
plt.axis('equal')
plt.show()
plt.pie([(beuro['AMT_ANNUITY']>0).sum(),(beuro['DAYS_CREDIT']<1).sum()],
        autopct='%.2f%%',shadow=True,
       startangle=90,labels=['Installment loans','Interest Only Loans'])
plt.axis('equal')
plt.show()

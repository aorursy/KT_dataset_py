import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

df1=pd.read_csv('../input/dataset/transactions_v2.csv')

df2=pd.read_excel('../input/datast2/members_v3.xlsx',index_col=0)
df3=pd.read_excel('../input/dataset/user_logs_v2.xlsx',index_col=0)
df3.to_csv('user_logs.csv', encoding='utf-8', index=False)

df4=pd.read_csv('user_logs.csv')
df2.to_csv('members.csv', encoding='utf-8', index=False)

df5=pd.read_csv('members.csv')
df4.head()
df7 = pd.concat([df1, df4,df5], axis=1)

df7.to_csv('file3.csv', index=False)
df8=pd.read_csv('file3.csv')
df8.head()
sns.countplot(df8['actual_amount_paid'])

import pandas as pd

import numpy as np

import seaborn as sns # data visualization library  

import matplotlib.pyplot as plt

import time

from subprocess import check_output

df8.head()
# feature names as a list

col = df8.columns       # .columns gives columns names in data 

print(col)
#Therefore, drop these unnecessary features. However do not forget this is not a feature selection. This is like a browse a pub, we do not choose our drink yet !!!

# y includes our labels and x includes our features

#list0 = ['is_cancel','is_auto_renew']

y = df8.is_auto_renew           # 0 or 1 

list1 = ['Unnamed: 9','payment_method_id.1','Unnamed: 10','msno.1','payment_plan_days.1','plan_list_price.1','actual_amount_paid.1','is_auto_renew.1','transaction_date.1','membership_expire_date.1', 'is_cancel.1',]

x = df8.drop(list1,axis = 1 )

x.shape
ax = sns.countplot(y,label="Count")       # 1 = 823454, 0 =225121 

print(y.value_counts())

x.describe()

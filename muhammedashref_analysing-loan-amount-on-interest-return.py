import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
df = pd.read_csv("/kaggle/input/lending-club-loan-data/loan.csv",low_memory=False)
data= df[df['loan_status'].apply(lambda x:x=='Charged Off' or x=='Fully Paid')][['loan_status','loan_amnt','int_rate','total_pymnt','issue_d','last_pymnt_d',]]

sns.countplot(data['loan_status'],order=data['loan_status'].value_counts().index,palette='cool_r')
data['loan_amnt_group'] = pd.cut(data['loan_amnt'],
                                 [0,5000,10000,15000,20000,25000,30000,35000,40000],
                                 labels=['below 5000','5001to10000','10001to15000','15001to20000','20001to25000',
                                         '25001to30000','30001to35000','35001to40000']) #grouped as per the loan amount
plt.figure(figsize=(12,8))
sns.countplot(data['loan_amnt_group'],order=data['loan_amnt_group'].value_counts().index,hue=data['loan_status'],palette='coolwarm_r')
plt.figure(figsize=(12,10))
sns.boxplot(x='loan_amnt_group',y='int_rate',data=data)
plt.figure(figsize=(12,10))
sns.violinplot(x='loan_amnt_group',y='int_rate',data=data)
meanrp= data.groupby('loan_amnt_group').mean()
plt.figure(figsize=(10,5))
chart=sns.barplot(x=meanrp.index,y='int_rate',data=meanrp,
                  order=meanrp['int_rate'].sort_values(ascending=False).index,
                  palette='winter')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart
meanfp=data[data['loan_status']=='Fully Paid'].groupby('loan_amnt_group').mean()
plt.figure(figsize=(10,5))
chart=sns.barplot(x=meanfp.index,y='int_rate',data=meanfp,
                  order=meanfp['int_rate'].sort_values(ascending=False).index,
                  palette='winter')
chart.set_xticklabels(chart.get_xticklabels(), rotation=45)
chart

del(data)
data= df[df['loan_status'].apply(lambda x:x=='Charged Off' or x=='Fully Paid' or x=='Current')][['loan_status','loan_amnt','int_rate','total_pymnt','issue_d','last_pymnt_d',]]
year_grp=['below 5000','5001to10000','10001to15000','15001to20000','20001to25000',
                                         '25001to30000','30001to35000','35001to40000']
data['loan_amnt_group'] = pd.cut(data['loan_amnt'],
                                 [0,5000,10000,15000,20000,25000,30000,35000,40000],
                                 labels=year_grp) #grouped as per the loan amount

data['issue_d']=pd.to_datetime(data['issue_d']) #Change data type of column issue_d from string to datetime.
data['issue_y']=data['issue_d'].apply(lambda x:x.year) #created a new column for year of issuing loan
plt.figure(figsize=(12,10))
sns.set_context('poster',font_scale=.5)
chart=sns.lineplot(x='issue_y',y='int_rate',hue='loan_amnt_group',
                   data=data,hue_order=year_grp,style='loan_amnt_group',
                   dashes=False,markers=True,palette='winter')
chart.set(xlabel='Year',ylabel='Avg. Interset rate')
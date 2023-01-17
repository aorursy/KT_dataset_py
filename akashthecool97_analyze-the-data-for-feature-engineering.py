

import numpy as np

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



hr = pd.read_csv("../input/HR_comma_sep.csv")

print(hr.head())
print('dept:')

print(set(hr['sales']))

print('salary:')

print(set(hr['salary']))

sales = ['management','RandD','hr','accounting','marketing','product_mng','IT','support','technical','sales']

salary = set(hr['salary'])

for i in range(hr.shape[0]):

    for j in range(len(sales)):

        if(hr['sales'][i]==sales[j]):

            hr.set_value(i,'sales',int(j))

    if(hr['salary'][i]=='low'):

        hr.set_value(i,'salary',int(3))

    if(hr['salary'][i]=='medium'):

        hr.set_value(i,'salary',int(2))

    if(hr['salary'][i]=='high'):

        hr.set_value(i,'salary',int(1))

hr[['sales']]=hr[['sales']].apply(pd.to_numeric)

hr[['salary']]=hr[['salary']].apply(pd.to_numeric)

print(hr.head())
hr['satisfaction_level']=100*hr['satisfaction_level']

hr['last_evaluation']=100*hr['last_evaluation']

hr['satisfaction_level']=hr['satisfaction_level'].astype(np.int32)

hr['last_evaluation']=hr['last_evaluation'].astype(np.int32)

print(hr.head())
plt.figure(figsize=(10,10))

h = sns.heatmap(hr.corr(),annot=True,square=True,annot_kws={"size":9})

plt.title('Correlation between features')

plt.show()
left = hr[hr['left']==1]

non_left = hr[hr['left']==0]

sns.kdeplot(left['last_evaluation'])

sns.kdeplot(non_left['last_evaluation'],shade=True)

plt.show()


sns.countplot('number_project',hue='left',data=hr)

plt.show()
sns.kdeplot(left['average_montly_hours'])

sns.kdeplot(non_left['average_montly_hours'],shade=True)

plt.show()
print(set(hr['promotion_last_5years']))
sns.countplot('promotion_last_5years',hue='left',data=hr)

plt.show()
print(hr.shape[0])

print(hr[hr['promotion_last_5years']==1].shape[0])

print(non_left[non_left['promotion_last_5years']==1].shape[0])

print(left[left['promotion_last_5years']==1].shape[0])
sns.countplot('sales',hue='left',data=hr)
m = []

for i in range(0,10):

    print(left[left['sales']==i].shape[0]/hr[hr['sales']==i].shape[0])

    m.append(left[left['sales']==i].shape[0]/hr[hr['sales']==i].shape[0])
n = m

n = sorted(n)

rename = []

for i in range(0,10):

    for j in range(0,10):

        if(m[i]==n[j]):

            rename.append(j)

hr['sales'].replace(range(0,10),rename,inplace=True)

sns.countplot('sales',hue='left',data=hr)
left = hr[hr['left']==1]

non_left = hr[hr['left']==0]

sns.countplot('sales',hue='salary',data=left)
sns.violinplot(x='sales',y='satisfaction_level',hue='left',data=hr,split=True)
sns.violinplot(x='sales',y='last_evaluation',hue='left',data=hr,split=True)
sns.countplot('sales',hue='number_project',data=left)
sns.violinplot(x='sales',y='average_montly_hours',hue='left',data=hr,split=True)
sns.countplot('sales',hue='time_spend_company',data=non_left,palette=sns.light_palette("green"))

sns.countplot('sales',hue='time_spend_company',data=left,palette=sns.cubehelix_palette(8))#violet


sns.countplot('Work_accident',hue='sales',data=non_left,palette=sns.cubehelix_palette(8))#violet

sns.countplot('Work_accident',hue='sales',data=left,palette=sns.light_palette("green"))
sns.countplot('promotion_last_5years',hue='sales',data=non_left,palette=sns.cubehelix_palette(8))#violet

sns.countplot('promotion_last_5years',hue='sales',data=left,palette=sns.light_palette("green"))
d_s=[]

d_a=[]

d_b=[]

d_c=[]

for i in range(hr.shape[0]):

    if(hr['last_evaluation'][i] in range(55,80)):

        d_s.append(1)

    else:

        d_s.append(0)

    if(hr['average_montly_hours'][i] in range(160,260)):

        d_a.append(1) 

    else:

        d_a.append(0)

    if(hr['time_spend_company'][i] in [2,3,4]):

        d_b.append(1)

    else:

        d_b.append(0)

    if(hr['number_project'][i] in [3,4,5]):

        d_c.append(0)

    else:

        d_c.append(1)

hr['discrete_last_evaluation']=pd.Series(d_s)

hr['expected_avg_hr']=pd.Series(d_a)

hr['shouldbe_time_spend_company']=pd.Series(d_b)

hr['expected_no_of_projects']=pd.Series(d_c)

hr.head()
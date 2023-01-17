import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns
df=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

df.head()
df_copy=df.copy()
df['sl_no'].unique().size
df.info()
df['salary'].fillna(0,inplace=True)

df['salary'].isna().any()
df_copy['salary'].median()
df['gender']=df['gender'].map({'M':0,'F':1})
df.head()
df['ssc_b'].value_counts()
df['ssc_b']=df['ssc_b'].map({'Central':1,'Others':0})
df['hsc_b']=df['hsc_b'].map({'Central':1,'Others':0})
df['hsc_s'].value_counts()
df_subjects=pd.get_dummies(df['hsc_s'])

df=df.merge(df_subjects,on=df.index)
df.drop('key_0',axis=1,inplace=True)

df.drop('hsc_s',axis=1,inplace=True)
df.columns
df=df[['sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'Arts', 'Commerce','Science','degree_p',

       'degree_t', 'workex', 'etest_p', 'specialisation', 'mba_p', 'status',

       'salary']]
df['degree_t'].value_counts()
df_deg=pd.get_dummies(df['degree_t'])

df=df.merge(df_deg,on=df.index)
df.drop('key_0',axis=1,inplace=True)

df.drop('degree_t',axis=1,inplace=True)
df.columns
df=df[['sl_no', 'gender', 'ssc_p', 'ssc_b', 'hsc_p', 'hsc_b', 'Arts',

       'Commerce', 'Science', 'degree_p','Comm&Mgmt', 'Others',

       'Sci&Tech','workex', 'etest_p',

       'specialisation', 'mba_p', 'status', 'salary']]
df
df['workex']=df['workex'].map({'Yes':1,'No':0})

df.head()
df['specialisation'].value_counts()
df['specialisation']=df['specialisation'].map({'Mkt&Fin':1,'Mkt&HR':0})
df['status'].value_counts()
df['status']=df['status'].map({'Placed':1,'Not Placed':0})
df.head()


df.loc[df['ssc_p']<=60,'ssc_p_c']=3

df.loc[(df['ssc_p']>60) & (df['ssc_p']<81),'ssc_p_c']=2

df.loc[(df['ssc_p']>80)& (df['ssc_p']<101),'ssc_p_c']=1



df.loc[df['hsc_p']<=60,'hsc_p_c']=3

df.loc[(df['hsc_p']>60) & (df['hsc_p']<81),'hsc_p_c']=2

df.loc[(df['hsc_p']>80)& (df['hsc_p']<101),'hsc_p_c']=1





df.loc[df['degree_p']<=60,'degree_p_c']=3

df.loc[(df['degree_p']>60) & (df['degree_p']<81),'degree_p_c']=2

df.loc[(df['degree_p']>80)& (df['degree_p']<101),'degree_p_c']=1



df.loc[df['mba_p']<=60,'mba_p_c']=3

df.loc[(df['mba_p']>60) & (df['mba_p']<81),'mba_p_c']=2

df.loc[(df['mba_p']>80)& (df['mba_p']<101),'mba_p_c']=1



df.loc[df['etest_p']<=60,'etest_p_c']=3

df.loc[(df['etest_p']>60) & (df['etest_p']<81),'etest_p_c']=2

df.loc[(df['etest_p']>80)& (df['etest_p']<101),'etest_p_c']=1







qual_type=['ssc_p','hsc_p','degree_p','mba_p','etest_p']



for qual in qual_type:

    df.drop(qual,axis=1,inplace=True)





df.columns
df=df[['sl_no', 'gender', 'ssc_b', 'hsc_b','ssc_p_c', 'hsc_p_c', 'degree_p_c', 'mba_p_c', 'etest_p_c', 'Arts', 'Commerce', 'Science',

       'Comm&Mgmt', 'Others', 'Sci&Tech', 'workex', 'specialisation', 'status',

       'salary']]
df.head()
sns.catplot('gender',data=df_copy,kind='count',hue='status',palette='rocket')
sns.catplot('gender',data=df_copy,kind='count',palette='winter')

plt.title('M/F ratio={0:.2f}'.format(df_copy['gender'].value_counts()[0]/df_copy['gender'].value_counts()[1]))
df_male=df_copy[df_copy['gender']=='M']
df_male['status'].value_counts()
male_placed_ratio=df_male['status'].value_counts()[0]/df_male['status'].value_counts()[1]
print('Placement ratio of male candidates:{0:.2f}'.format(male_placed_ratio))
df_female=df_copy[df_copy['gender']=='F']

df_female['status'].value_counts()
female_placed_ratio=df_female['status'].value_counts()[0]/df_female['status'].value_counts()[1]

print('Placement ratio of female candidates:{0:.2f}'.format(female_placed_ratio))
sns.catplot('ssc_p_c',data=df,kind='count')

plt.title('10th percentage distribution')

plt.xlabel('10th percentage class')
sns.catplot('ssc_p_c',data=df,kind='count',hue='status',palette='inferno')

plt.title('10th Percentage with placed/unplaced')

plt.xlabel('10th Percentage class')
sns.catplot('hsc_p_c',data=df,kind='count')

plt.title('12th percentage distribution')

plt.xlabel('12th Percentage class')
sns.catplot('hsc_p_c',data=df,kind='count',hue='status',palette='summer')

plt.title('12th Percentage with placement status')

plt.xlabel('12th Percentage class')
fig=plt.figure(figsize=(10,5))



ax1 = fig.add_subplot(121)



g = sns.countplot("degree_p_c" , data=df, ax=ax1,palette='ocean')



ax2=fig.add_subplot(122)



g=sns.countplot('degree_p_c',data=df,ax=ax2,hue='status',palette='winter')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('UG percentage distribution')

ax1.set_xlabel('UG percentage class')

ax2.set_title('UG percentage with placement status')

ax2.set_xlabel('UG percentage class')
fig1=plt.figure(figsize=(10,5))



ax1 = fig1.add_subplot(121)



g = sns.countplot("mba_p_c" , data=df, ax=ax1,palette='rocket')



ax2=fig1.add_subplot(122)



g=sns.countplot('mba_p_c',data=df,ax=ax2,hue='status',palette='viridis')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('MBA percentage distribution')

ax1.set_xlabel('MBA percentage class')

ax2.set_title('MBA percentage with placement status')

ax2.set_xlabel('MBA percentage class')
fig2=plt.figure(figsize=(10,5))



ax1 = fig2.add_subplot(121)



g = sns.countplot("etest_p_c" , data=df, ax=ax1,palette='PuRd')



ax2=fig2.add_subplot(122)



g=sns.countplot('etest_p_c',data=df,ax=ax2,hue='status',palette='BuPu')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('E test percentage distribution')

ax1.set_xlabel('E test percentage class')

ax2.set_title('E test percentage with placement status')

ax2.set_xlabel('E test precentage class')
fig3=plt.figure(figsize=(10,5))



ax1 = fig3.add_subplot(121)



g = sns.countplot("hsc_s" , data=df_copy, ax=ax1,palette='OrRd')



ax2=fig3.add_subplot(122)



g=sns.countplot('hsc_s',data=df_copy,ax=ax2,hue='status',palette='gist_earth')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('High school specialisation distribution')

ax1.set_xlabel('HS specialisation')

ax2.set_title('High school spcialisation distribution')

ax2.set_xlabel('HS specialisation')
fig4=plt.figure(figsize=(10,5))



ax1 = fig4.add_subplot(121)



g = sns.countplot("specialisation" , data=df_copy, ax=ax1,palette='OrRd')



ax2=fig4.add_subplot(122)



g=sns.countplot('specialisation',data=df_copy,ax=ax2,hue='status',palette='gist_earth')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('MBA specialisation distribution')

ax1.set_xlabel('MBA specialisation')

ax2.set_title('MBA spcialisation distribution')

ax2.set_xlabel('MBA specialisation')
fig5=plt.figure(figsize=(10,5))



ax1 = fig5.add_subplot(121)



g = sns.countplot("degree_t" ,data=df_copy, ax=ax1,palette='OrRd')



ax2=fig5.add_subplot(122)



g=sns.countplot('degree_t',data=df_copy,ax=ax2,hue='status',palette='gist_earth')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('UG specialisation distribution')

ax1.set_xlabel('UG specialisation')

ax2.set_title('UG spcialisation distribution')

ax2.set_xlabel('UG specialisation')
fig6=plt.figure(figsize=(10,5))



ax1 = fig6.add_subplot(121)



g = sns.countplot("ssc_b" , data=df_copy, ax=ax1,palette='rocket')



ax2=fig6.add_subplot(122)



g=sns.countplot('ssc_b',data=df_copy,ax=ax2,hue='status',palette='gist_earth')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('10th board students distribution')

ax1.set_xlabel('Students')

ax2.set_title('10th board students distribution')

ax2.set_xlabel('Students')
fig7=plt.figure(figsize=(10,5))



ax1 = fig7.add_subplot(121)



g = sns.countplot("hsc_b" , data=df_copy, ax=ax1,palette='rocket')



ax2=fig7.add_subplot(122)



g=sns.countplot('hsc_b',data=df_copy,ax=ax2,hue='status',palette='gist_earth')

plt.close(2)

plt.close(3)

plt.tight_layout()



ax1.set_title('12th board students distribution')

ax1.set_xlabel('Students')

ax2.set_title('12th board students distribution')

ax2.set_xlabel('Students')
plt.figure(figsize=(10,8))

df_placed=df[df['salary']>0]

sns.distplot(df_placed['salary'])

plt.xlabel('Salary (in Rs)',size=10)

plt.title('Salary distribution for the batch',size=15)
mean=df_copy['salary'].mean()

median=df_copy['salary'].median()

plt.figure(figsize=(10,8))

sns.boxplot(df_copy['salary'],orient='v')
sns.violinplot(df_copy['salary'],orient='v',color='red')
plt.figure(figsize=(10,8))

df_placed=df[df['salary']>0]

sns.distplot(df_placed['salary'])

plt.xlabel('Salary (in Rs)',size=10)

plt.title('Salary distribution for the batch',size=15)

plt.axvline(mean,color='red')

plt.axvline(median,color='green')

plt.title('Mean={0:.2f}   Median={1:.2f}'.format(mean,median))

corr=df.corr()
plt.figure(figsize=(20,10))

sns.heatmap(corr,annot=True,cmap='viridis')
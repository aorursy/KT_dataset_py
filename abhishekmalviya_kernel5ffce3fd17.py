# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl # data visualization

import matplotlib.pyplot as plt # data visualization

import seaborn as sns # data visualization
cbse=pd.read_csv('/kaggle/input/cbse-schools-data/schools_detailed.csv')

cbse.shape
# arrange garph size

plt.subplots(figsize=(10,6))



#use countplot to represent count of school in each state

ax=sns.countplot(x="state", data=cbse, order=cbse['state'].value_counts().index)

#ax=cbse['state'].value_counts().plot.bar(width=0.8,align = 'center',label ='Guido') alternate steps for same results



#rotate name of states to vertical as name were overlapping horizontally

plt.xticks(rotation='vertical') 



#name of horizontal axis

plt.xlabel('State', fontsize=15,color='blue')



#name of vertical axis

plt.ylabel('No. of schools', fontsize=15,color='blue')



# tilte of graph 

plt.title('No. of school per state',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(10,6))

ax=sns.countplot(x="region", data=cbse, order=cbse['region'].value_counts().index)

plt.xticks(rotation='vertical')

plt.xlabel('Region', fontsize=15,color='blue')

plt.ylabel('No. of School', fontsize=15,color='blue')

plt.title('No. of school per region', fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(10,6))

ax=sns.countplot(x="district",data=cbse, order=cbse['district'].value_counts().head(15).index)

plt.xticks(rotation='vertical')

plt.xlabel('District', fontsize=15,color='blue')

plt.ylabel('No. of School', fontsize=15,color='blue')

plt.title('Top 15 District with highest no. of school', fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(10,6))

ax=sns.countplot(x='n_medium', data=cbse, order=cbse['n_medium'].value_counts().index)

plt.xlabel('Medium', fontsize=15,color='blue')

plt.ylabel('No. of School', fontsize=15,color='blue')

plt.title('No. of school per medium',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(10,6))

ax=sns.countplot(x='n_school_type', data=cbse, order=cbse['n_school_type'].value_counts().index)

plt.xticks(rotation='vertical')

plt.xlabel('School type', fontsize=15,color='blue')

plt.ylabel('No. of School', fontsize=15,color='blue')

plt.title('No. of school per school type',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(10,6))

ax=sns.countplot(x='n_school_type',hue='n_medium', data=cbse, order=cbse['n_school_type'].value_counts().index)

plt.legend(title='Medium', loc='upper right')

plt.xticks(rotation='vertical')

plt.xlabel('School type', fontsize=15,color='blue')

plt.ylabel('No. of School', fontsize=15,color='blue')

plt.title('No. of school per school type and Medium',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(10,6))

ax=sns.countplot(x='aff_type',data=cbse, order=cbse['aff_type'].value_counts().index)

plt.xlabel('Affiliation type', fontsize=15,color='blue')

plt.ylabel('No. of School', fontsize=15,color='blue')

plt.title('No. of school as per affiliation type',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(10,6))

ax=sns.countplot(x='n_category',data=cbse, order=cbse['n_category'].value_counts().index)

plt.xlabel('School Category', fontsize=15,color='blue')

plt.ylabel('No. of School', fontsize=15,color='blue')

plt.title('No. of school as per category',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(10,6))

ax=sns.countplot(x='status',data=cbse, order=cbse['status'].value_counts().index)

plt.xlabel('School Status', fontsize=15,color='blue')

plt.ylabel('No. of School', fontsize=15,color='blue')

plt.title('No. of school as per school status',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(10,6))

ax=sns.countplot(x='status',data=cbse,hue='n_category', order=cbse['status'].value_counts().index)

plt.legend(title='Category')

plt.xlabel('School Status', fontsize=15,color='blue')

plt.ylabel('No. of School', fontsize=15,color='blue')

plt.title('No. of school as per school status with category',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(10,6))

ax=sns.countplot(x='n_school_type',hue='status', data=cbse, order=cbse['n_school_type'].value_counts().index)

plt.legend(title='Status')

plt.xticks(rotation='vertical')

plt.xlabel('School type', fontsize=15,color='blue')

plt.ylabel('No. of School', fontsize=15,color='blue')

plt.title('No. of school per school type and Status',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(20,10))

ax=sns.countplot(x="state",hue='status', data=cbse, order=cbse['state'].value_counts().index)

plt.legend(title='School Status', loc='upper right')

plt.xticks(rotation='vertical') 

plt.xlabel('State', fontsize=15,color='blue')

plt.ylabel('No. of schools', fontsize=15,color='blue')

plt.title('No. of school per state with status',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(20,10))

ax=sns.countplot(x="state",hue='n_category', data=cbse, order=cbse['state'].value_counts().index)

plt.legend(title='School Category', loc='upper right')

plt.xticks(rotation='vertical') 

plt.xlabel('State', fontsize=15,color='blue')

plt.ylabel('No. of schools', fontsize=15,color='blue')

plt.title('No. of school per state with category',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(20,10))

ax=sns.countplot(x="state",hue='n_medium', data=cbse, order=cbse['state'].value_counts().index)

plt.legend(title='School Medium', loc='upper right')

plt.xticks(rotation='vertical') 

plt.xlabel('State', fontsize=15,color='blue')

plt.ylabel('No. of schools', fontsize=15,color='blue')

plt.title('No. of school per state with medium',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(20,10))

ax=sns.countplot(x="state",hue='n_school_type', data=cbse, order=cbse['state'].value_counts().index)

plt.legend(title='School Type', loc='upper right')

plt.xticks(rotation='vertical') 

plt.xlabel('State', fontsize=15,color='blue')

plt.ylabel('No. of schools', fontsize=15,color='blue')

plt.title('No. of school per state with school type',fontsize=20,color='purple')

plt.show()
# data transformation for principal gender 1 - Male, 2- Female, 0 - Not Available

s_map={1.0 : 'Male' , 2.0 : 'Female' , 0.0: 'Not Available'}

cbse['sex']=cbse['sex'].map(s_map)

plt.subplots(figsize=(10,6))

ax=sns.countplot(x="sex", data=cbse, order=cbse['sex'].value_counts().index)

plt.xlabel("Principal's Gender", fontsize=15,color='blue')

plt.ylabel('Total Number', fontsize=15,color='blue')

plt.title('School principal gender information',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(20,10))

ax=sns.countplot(x="state",hue='sex', data=cbse, order=cbse['state'].value_counts().index)

plt.legend(title='Gender', loc='upper right')

plt.xticks(rotation='vertical') 

plt.xlabel('State', fontsize=15,color='blue')

plt.ylabel('No. of principal', fontsize=15,color='blue')

plt.title('Principal gender info per state',fontsize=20,color='purple')

plt.show()
plt.subplots(figsize=(10,6))

ax=sns.countplot(x='n_school_type',hue='sex', data=cbse, order=cbse['n_school_type'].value_counts().index)

plt.legend(title='Gender')

plt.xticks(rotation='vertical')

plt.xlabel('School type', fontsize=15,color='blue')

plt.ylabel('No. of principal', fontsize=15,color='blue')

plt.title('No. of school type per school type and gender',fontsize=20,color='purple')

plt.show()
fig1, ax1 = plt.subplots()

ax1.pie(cbse['sex'].value_counts(), autopct='%1.1f%%',labels=cbse['sex'].value_counts().keys(),shadow=True,

       explode= [0.1,0.1,0.1])

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Principal gender info',fontsize=15,color='purple')

plt.show()
fig1, ax1 = plt.subplots()

ax1.pie(cbse['region'].value_counts(), autopct='%1.1f%%',labels=cbse['region'].value_counts().keys(),shadow=True,

       explode= [0.09]*len(cbse['region'].value_counts().keys()))

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Region wise school percentage',fontsize=15,color='purple')

plt.show()
fig1, ax1 = plt.subplots(figsize=(20,10))

ax1.pie(cbse['state'].value_counts().head(19), autopct='%1.1f%%',labels=cbse['state'].value_counts().head(19).keys(),shadow=False,

       pctdistance =.9)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('State wise school percentage(Top 19 States)',fontsize=15,color='purple')

plt.show()
fig1, ax1 = plt.subplots()

ax1.pie(cbse['n_category'].value_counts(), autopct='%1.1f%%',labels=cbse['n_category'].value_counts().keys(),shadow=True,

       explode= [0.2]*len(cbse['n_category'].value_counts().keys()))

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('Category wise school percentage',fontsize=15,color='purple')

plt.show()
fig1, ax1 = plt.subplots()

ax1.pie(cbse['n_school_type'].value_counts(), autopct='%1.1f%%',labels=cbse['n_school_type'].value_counts().keys(),shadow=True,

       explode= [0.3]*len(cbse['n_school_type'].value_counts().keys()))

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('School type wise school percentage',fontsize=15,color='purple')

plt.show()
fig1, ax1 = plt.subplots()

ax1.pie(cbse['status'].value_counts(), autopct='%1.1f%%',labels=cbse['status'].value_counts().keys(),shadow=True,

       explode= [0.06]*len(cbse['status'].value_counts().keys()))

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

plt.title('School status wise school percentage',fontsize=15,color='purple')

plt.show()
trial=cbse[['year_found','date_opened','aff_start','aff_end']]

trial=trial.dropna(subset=['year_found'])
trial['year_found']=pd.to_datetime(trial['year_found'],format='%Y').dt.year



plt.subplots(figsize=(10,6))

ax=sns.countplot(x="year_found", data=trial, order=trial.groupby((trial['year_found']//10)*10).count().index)

plt.xticks(rotation='vertical')

plt.xlabel('year_found', fontsize=15,color='blue')

plt.ylabel('No. of School', fontsize=15,color='blue')

plt.title('No. of school founded per decade', fontsize=20,color='purple')

plt.show()

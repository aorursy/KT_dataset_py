# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
path='../input/immigration-dataset'
ctyeconprofile=pd.read_csv(f'{path}/ctyeconprofile.csv',encoding='latin-1')

soi_migration_clean=pd.read_csv(f'{path}/soi_migration_clean.csv',encoding='latin-1')
ctyeconprofile.tail()
smc=soi_migration_clean

smc.tail()
length=[]

for i in range(1995,2016):

    length.append(len(smc.loc[(smc['year1']==i) & (smc['year2']==i+1)]))



length=pd.DataFrame(length).T

length.columns=np.arange(1995,2016)

length
results=[]

for i in range(1995,2016):

    results.append(smc.loc[(smc['year1']==i) & (smc['year2']==i+1)])
total_pop=ctyeconprofile[['fips','year','ctyname','tot_pop']]

total_pop.tail()
results_with_pop=[]

for i in range(1995,2016):

    results_with_pop.append(pd.merge(results[i-1995],total_pop.loc[total_pop['year']==i],on='fips'))
results_with_pop[0].tail()
for i in range(21):

    temp=results_with_pop[i]

    temp['gross migration']=temp['inmig']+temp['outmig']

    temp['gross migration rate']=temp['gross migration']*100/temp['tot_pop']
results_with_pop[0].tail()
df=results_with_pop[0]

for i in range(1,21):

    df=df.append(results_with_pop[i])

df
len(df['fips'].unique())
label=df['fips'].unique()

for temp_label in label:

    temp_data=df[df['fips'].isin([temp_label])]

    temp_data.index=temp_data['year']

    exec("df%s = temp_data"%temp_label)
df1001.tail()
df1003.tail()
df46102
percentage_difference=[]



for temp_label in label:

    exec("max_percentage=max(df%s['gross migration rate'])"%temp_label)

    exec("min_percentage=min(df%s['gross migration rate'])"%temp_label)

    

    max_temp=float("%s"%max_percentage)

    min_temp=float("%s"%min_percentage)

    temp=max_temp-min_temp

    percentage_difference.append(temp)
percentage_difference_new=pd.DataFrame(percentage_difference)

percentage_difference_new.columns=['percentage difference']

percentage_difference_new['label']=label
max_migration=percentage_difference_new.sort_values(['percentage difference'],ascending=False).head(10)

max_migration
temp_results_percentage=pd.DataFrame()

for temp_label in max_migration['label']:

    temp_results_percentage[str(temp_label)]=eval("(df%s['gross migration rate'])"%temp_label)
temp_results_percentage.index=df20201['year'].astype(int)

temp=temp_results_percentage['20201']

temp_results_percentage.drop(['20201'],axis=1,inplace=True)

temp_results_percentage['20201']=temp

temp_results_percentage.head()
import matplotlib.pyplot as plt

import seaborn as sns

sns.set(rc={"figure.figsize": (16, 6)}); 

sns.set(style="white",palette='deep',color_codes=False)
fig,axes=plt.subplots(1,2)

temp_results_percentage.plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-','X--','|-.'],ax=axes[0],xticks=range(1995,2016,2))

axes[0].set_ylabel('Gross Migration Percentage'),axes[0].set_xlabel('Year')



temp_results_percentage.drop(['20201'],axis=1).plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-','X--','|-.'],ax=axes[1],xticks=range(1995,2016,2))

axes[1].set_ylabel('Gross Migration Percentage'),axes[1].set_xlabel('Year')

sns.despine()
gross_difference=[]



for temp_label in label:

    exec("max_gross=max(df%s['gross migration'])"%temp_label)

    exec("min_gross=min(df%s['gross migration'])"%temp_label)

    

    max_temp=float("%s"%max_gross)

    min_temp=float("%s"%min_gross)

    temp=max_temp-min_temp

    gross_difference.append(temp)
gross_difference_new=pd.DataFrame(gross_difference)

gross_difference_new.columns=['gross difference']

gross_difference_new['label']=label
max_migration=gross_difference_new.sort_values(['gross difference'],ascending=False).head(10)

max_migration
temp_results=pd.DataFrame()

for temp_label in max_migration['label']:

    temp_results[str(temp_label)]=eval("(df%s['gross migration'])"%temp_label)
temp_results.head()
temp_results.index=df20201['year'].drop([2003],axis=0).astype(int)

temp=temp_results['6037']

temp_results.drop(['6037'],axis=1,inplace=True)

temp_results['6037']=temp

temp_results.head()
fig,axes=plt.subplots(1,2)

temp_results.plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-','X--','|-.'],ax=axes[0],xticks=range(1995,2016,2))

axes[0].set_ylabel('Gross Migration'),axes[0].set_xlabel('Year')



temp_results.drop(['6037'],axis=1).plot(style=['+-', 'd--', 'o-.', '.-', 'v:','*-','X--','|-.'],ax=axes[1],xticks=range(1995,2016,2))

axes[1].set_ylabel('Gross Migration'),axes[1].set_xlabel('Year')

sns.despine()
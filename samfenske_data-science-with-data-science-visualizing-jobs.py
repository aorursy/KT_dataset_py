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
jobs=pd.read_csv('/kaggle/input/data-analyst-jobs/DataAnalyst.csv').drop(columns='Unnamed: 0')

jobs
jobs['Salary Estimate'].value_counts()
import re

from statistics import mean 

ranges=[]

averages=[]

lower=[]

upper=[]

for entry in jobs['Salary Estimate']:

    regex =  '[$][0-9]+[K]-[$][0-9]+[K]' 

    if 'K' in entry:

        salary=re.search(regex, entry).group(0)

        ranges.append(salary)

        

        temp = re.findall(r'\d+', entry)

        drop = list(map(int, temp))

        averages.append(mean(drop)*1000)

        lower.append(min(drop)*1000)

        upper.append(max(drop)*1000)

        #upper.append('${:,.2f}'.format(max(drop)*1000))



    else:

        ranges.append(None)

        averages.append(None)

        lower.append(None)

        upper.append(None)

jobs['Range']=ranges

jobs['Average']=averages

jobs['Lower']=lower

jobs['Upper']=upper
jobs.sort_values(by='Lower',ascending=False).head()
jobs.sort_values(by='Upper',ascending=False)
#replace null '-1' entries with None

for col in jobs:

    jobs[col]= jobs[col].replace('-1', np.nan)

    jobs[col]= jobs[col].replace(-1, np.nan)
sectors=jobs['Sector'].value_counts().index

sector_breakdown=pd.DataFrame()

for sector in sectors:

    df=jobs[jobs['Sector'].isin([sector])]

    if len(df)>20:

        sector_breakdown=sector_breakdown.append({'Sector':sector, 'Upper avg':df['Upper'].mean(),'Lower avg':df['Lower'].mean(),

                                             'Average mean':df['Average'].mean(),'Num':len(df)},ignore_index=True)

sector_breakdown
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(30,10))

ax = sns.violinplot(x="Industry", y="Average", data=jobs)

plt.xticks(rotation=90)
sector_breakdown.plot(x='Sector',y=['Lower avg','Average mean','Upper avg'],kind='bar')

plt.gcf().set_size_inches(18.5, 10.5)

sector_breakdown.plot(x='Sector',y='Num',kind='bar')

plt.xticks(rotation=90)

plt.gcf().set_size_inches(18.5, 10.5)
#clean up company name entries

jobs['Company Name']=jobs['Company Name'].astype(str)

jobs['Company Name']=[jobs['Company Name'][index].replace('\n','$').split('$',1)[0] for index in range(len(jobs))]
bio=jobs[jobs['Sector'].isin(['Biotech & Pharmaceuticals'])]

companies=bio['Company Name'].value_counts().index

bio_breakdown=pd.DataFrame()

for com in companies:

    df=bio[bio['Company Name'].isin([com])]

    bio_breakdown=bio_breakdown.append({'Company Name':com, 'Upper avg':df['Upper'].mean(),'Lower avg':df['Lower'].mean(),

                                             'Average mean':df['Average'].mean(),'Num':len(df)},ignore_index=True)

bio_breakdown
len(companies)
bio_breakdown.plot(x='Company Name',y=['Lower avg','Average mean','Upper avg'],kind='bar')

plt.xticks(rotation=90)



plt.gcf().set_size_inches(18.5, 10.5)
it=jobs[jobs['Sector'].isin(['Information Technology'])]

companies=it['Company Name'].value_counts().index

it_breakdown=pd.DataFrame()

for com in companies:

    df=it[it['Company Name'].isin([com])]

    it_breakdown=it_breakdown.append({'Company Name':com, 'Upper avg':df['Upper'].mean(),'Lower avg':df['Lower'].mean(),

                                      'Average mean':df['Average'].mean(),'Num':len(df)},ignore_index=True)



it_breakdown.plot(x='Company Name',y=['Lower avg','Average mean','Upper avg'],kind='bar')



plt.xticks(rotation=90)

plt.gcf().set_size_inches(25, 10.5)
len(it['Company Name'].value_counts())
names=it['Company Name'].value_counts().nlargest(25).index

it_top=pd.DataFrame()

for com in names:

    df=it[it['Company Name'].isin([com])]

    it_top=it_top.append({'Company Name':com, 'Upper avg':df['Upper'].mean(),'Lower avg':df['Lower'].mean(),

                                      'Average mean':df['Average'].mean(),'Num':len(df)},ignore_index=True)



it_top.plot(x='Company Name',y=['Lower avg','Average mean','Upper avg'],kind='bar')



plt.xticks(rotation=90)

plt.gcf().set_size_inches(25, 10.5)
it_top.plot(x='Company Name',y='Num',kind='bar')

plt.xticks(rotation=90)

plt.gcf().set_size_inches(18.5, 10.5)
apple=it[it['Company Name'].isin(['Apple'])]

apple['Job Description']
filtered=apple[-apple['Job Description'].str.contains('rofessional')&-apple['Job Title'].str.contains('enior')&-apple['Job Description'].str.contains('ears')]

filtered
for desc in filtered['Job Description']:

    print(desc)
for index in filtered.reset_index()['index']:

    print(filtered['Job Title'][index]+'\n'+filtered['Job Description'][index]+'\n') 
finance=jobs[jobs['Sector'].isin(['Finance'])]

companies=finance['Company Name'].value_counts().index

finance_breakdown=pd.DataFrame()

for com in companies:

    df=finance[finance['Company Name'].isin([com])]

    finance_breakdown=finance_breakdown.append({'Company Name':com, 'Upper avg':df['Upper'].mean(),'Lower avg':df['Lower'].mean(),

                                      'Average mean':df['Average'].mean(),'Num':len(df)},ignore_index=True)



finance_breakdown.plot(x='Company Name',y=['Lower avg','Average mean','Upper avg'],kind='bar')



plt.xticks(rotation=90)

plt.gcf().set_size_inches(25, 10.5)
len(companies)
names=finance['Company Name'].value_counts().nlargest(25).index

finance_top=pd.DataFrame()

for com in names:

    df=finance[finance['Company Name'].isin([com])]

    finance_top=finance_top.append({'Company Name':com, 'Upper avg':df['Upper'].mean(),'Lower avg':df['Lower'].mean(),

                                      'Average mean':df['Average'].mean(),'Num':len(df)},ignore_index=True)



finance_top.plot(x='Company Name',y=['Lower avg','Average mean','Upper avg'],kind='bar')



plt.xticks(rotation=90)

plt.gcf().set_size_inches(25, 10.5)
finance_top.plot(x='Company Name',y='Num',kind='bar')

plt.xticks(rotation=90)

plt.gcf().set_size_inches(18.5, 10.5)
citi=finance[finance['Company Name'].isin(['Citi'])]

citi
chi=jobs[jobs['Location'].str.contains('Chicago')]

chi['Sector'].value_counts()
chi[chi['Sector'].isin(['Biotech & Pharmaceuticals'])]
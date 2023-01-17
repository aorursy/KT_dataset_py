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
df=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

df
df.head()
df.columns
df.info
#check for null values

df.isnull().sum()
#dropping HDI column as it has half of the data having null values

df.drop('HDI for year',axis=1)
df.columns
df.age.unique()
import matplotlib.pyplot as plt

gp=df.groupby('sex')

df1=gp.get_group('male')

df2=gp.get_group('female')

plt.bar(['male','female'],[df1['suicides_no'].sum(),df2['suicides_no'].sum()])

gpam=df1.groupby('age')

gpaf=df2.groupby('age')

df3=gpam.get_group('5-14 years')

df4=gpam.get_group('15-24 years')

df5=gpam.get_group('25-34 years')

df6=gpam.get_group('35-54 years')

df7=gpam.get_group('55-74 years')

df8=gpam.get_group('75+ years')



dff3=gpaf.get_group('5-14 years')

dff4=gpaf.get_group('15-24 years')

dff5=gpaf.get_group('25-34 years')

dff6=gpaf.get_group('35-54 years')

dff7=gpaf.get_group('55-74 years')

dff8=gpaf.get_group('75+ years')



plt.figure(figsize=(10,6))

plt.bar(['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'],[df3['suicides_no'].sum(),df4['suicides_no'].sum(),df5['suicides_no'].sum()

                                                                                           ,df6['suicides_no'].sum()

                                                                                           ,df7['suicides_no'].sum()

                                                                                            ,df8['suicides_no'].sum()],label='male')



plt.bar(['5-14 years','15-24 years','25-34 years','35-54 years','55-74 years','75+ years'],[dff3['suicides_no'].sum(),dff4['suicides_no'].sum(),dff5['suicides_no'].sum()

                                                                                           ,dff6['suicides_no'].sum()

                                                                                           ,dff7['suicides_no'].sum()

                                                                                            ,dff8['suicides_no'].sum()],label='female')

plt.xlabel('Age-Group')

plt.ylabel(' total No of Sucides')

plt.legend()

import seaborn as sns
df1=df.groupby('country')

df1.groups.keys()

coun=[]

country=[]

for c in df1.groups.keys():

    gp=df1.get_group(c)

    #print(c,gp['suicides_no'].sum())

    coun.append(gp['suicides_no'].sum())

    country.append(c)

country_series=pd.Series(country)

nos=pd.Series(coun)

frame={'country':country_series,'suicide_no':nos}

dfn=pd.DataFrame(frame)

x=dfn.sort_values('suicide_no',ascending=False).head()

sns.barplot(x=x['country'],y=x['suicide_no'])

plt.xticks(rotation=50)

plt.style.use('ggplot')

plt.xlabel('Top 5 Country')

plt.ylabel('no of suicides')





dfn=dfn[dfn['suicide_no']>0]


plt.figure(figsize=(8,20))

sns.barplot(y=dfn['country'],x=dfn['suicide_no'],orient='h')
df
df.drop('HDI for year',axis=1,inplace=True)

df2=df.groupby('country')

df2.groups.keys()

coun=[]

country=[]

for c in df1.groups.keys():

    gp=df2.get_group(c)

    #print(c,gp['suicides_no'].sum())

    coun.append(gp['suicides/100k pop'].sum())

    country.append(c)

country_series=pd.Series(country)

nos=pd.Series(coun)

frame={'country':country_series,'suicide ratio per 100k population':nos}

dfd=pd.DataFrame(frame)

plt.figure(figsize=(8,20))

sns.barplot(y=dfd['country'],x=dfd['suicide ratio per 100k population'],orient='h')



dfd=dfd.sort_values(by='suicide ratio per 100k population',ascending=False).head(5)
plt.figure(figsize=(10,8))

plt.style.use('ggplot')

sns.barplot(x=dfd['country'],y=dfd['suicide ratio per 100k population'],hue=dfd['suicide ratio per 100k population'])
coun=[]

year=[]

plt.figure(figsize=(15,8))

new_gp=df.groupby('year')

for c in new_gp.groups.keys():

    gp=new_gp.get_group(c)

    coun.append(gp['suicides_no'].sum())

    year.append(c)

sns.barplot(x=year,y=coun)

plt.xlabel('Year')

plt.ylabel('no of suicides')

plt.title('No of suicides in whole world every year')
gdp=[]

sno=[]

country=[]

nwgp=df.groupby('country')

for c in nwgp.groups.keys():

    gp=nwgp.get_group(c)

    gdp.append(gp['gdp_per_capita ($)'].iloc[0])

    sno.append(gp['suicides_no'].sum())

    country.append(c)

country_series=pd.Series(country)

nos=pd.Series(sno)

gd=pd.Series(gdp)

frame={'country':country_series,'suicide no':nos,'GDP':gd}

dfd=pd.DataFrame(frame)   

dfd=dfd.sort_values(by='GDP')

dfd

plt.figure(figsize=(18,8))

sns.barplot(x=dfd['GDP'],y=dfd['suicide no'])

plt.xticks(rotation=70)
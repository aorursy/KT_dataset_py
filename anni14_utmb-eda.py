# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from matplotlib import pyplot as plt
sns.set_style("whitegrid")
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
i=2003
df_full=pd.DataFrame()
while i <= 2017:
    df_y = pd.read_csv('../input/utmb_'+str(i)+'.csv')
    df_y['year']=i
    df_y=df_y[['Unnamed: 0','name','nationality','team','time','year','rank','category']]
    df_full=df_full.append(df_y)
    i=i+1
df_full = df_full.rename(columns={'Unnamed: 0': 'Position'})
df_full["nationality"] = df_full.nationality.str.upper()
df_full.head()
cat1=[]
cat2=[]
cat3=[]
for value in df_full.category:
    cat1.append(value[0])
    cat2.append(value[1])
    if len(value) == 4:
        cat3.append(value[3])
    else:
        cat3.append(value[2])
    
df_full['cat1']=cat1
df_full['cat2']=cat2
df_full['cat3']=cat3

df_full.head()
df_full.describe()
df_full.info()
pd.isnull(df_full).sum()
df_finish = df_full[(df_full.time != " ") & (df_full.time.notna())]

df_dnf = df_full[(df_full.time == " ") | (df_full.time.isna())]
df_finish.head()
df_dnf.head()
pd.isnull(df_finish).sum()
df_finish[df_finish.nationality.isna()]
df_full[df_full["year"]==2010]
df_finish[df_finish.cat3=='F'].head()
sns.countplot(x='cat3',  data=df_finish)
plt.title("Female/Male")
plt.show()

g=sns.catplot(x="year", col="cat3", data=df_finish, kind="count")
g.set_xticklabels(rotation=45)
#plt.xticks(rotation=45)
plt.show()
print("Overall man",len(df_finish[df_finish.cat3 == "H"]))
print("Overall women",len(df_finish[df_finish.cat3 == "F"]))
sns.countplot(x='year',  data=df_finish)
plt.title("Year")
plt.xticks(rotation=90)
plt.show()
sns.countplot(x='nationality',  data=df_finish, order=df_finish.nationality.value_counts().iloc[:10].index)
plt.title("Year")
plt.show()
df_finish[df_finish['Position']==0]
df_new=pd.DataFrame()
df_new['time']=pd.to_datetime(df_finish[df_finish['Position']==0].time, format='%H:%M:%S')
df_new['year']=df_finish[df_finish['Position']==0].year
plt.plot_date(x='year', y='time', data=df_new, ydate=True,xdate=False, fmt='r.-')
plt.title("time")
plt.show()
df_new=pd.DataFrame()
df_new['time']=pd.to_datetime(df_finish[df_finish['Position']==1].time, format='%H:%M:%S')
df_new['year']=df_finish[df_finish['Position']==1].year
plt.plot_date(x='year', y='time', data=df_new, ydate=True,xdate=False, fmt='r.-')
plt.title("time")
plt.show()

x=df_finish[df_finish['cat3']=='F'].groupby('year').year.count()
plt.plot(x)
df_new=pd.DataFrame()

df_new['time']=pd.to_timedelta(df_finish[df_finish['cat3']=='F'].time, unit='h')/pd.Timedelta('1 hour')
df_new['year']=df_finish[df_finish['cat3']=='F'].year
df_new=df_new.groupby(['year'], as_index=False)['time'].min()
plt.plot_date(x='year', y='time', data=df_new,xdate=False, fmt='r.-')
plt.title("time")
plt.show()
df_new
df_full.groupby('name').size().sort_values(ascending=False)

df_full[df_full["name"].str.contains("DELEMONTEZ", case=False)]
df_full[df_full["name"].str.contains("drescher", case=False)]
df_full[df_full["nationality"].str.contains("DE", case=False, na=False)].count()
df_de = df_full[df_full["nationality"].str.contains("DE", case=False, na=False)]
lst_de=df_de.groupby('year').year.count()

plt.plot(lst_de)
df_full[df_full['team']!=" "].groupby(['year','team']).team.count().nlargest(20)


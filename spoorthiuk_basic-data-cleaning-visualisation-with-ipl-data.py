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
df=pd.read_csv('/kaggle/input/iplcricket-league-dataset-for-beginner-analaysis/IPL_final.csv')
df.head() #Top 5 rows
df.columns #columns present in the dataset
len(df.columns)
import matplotlib.pyplot as plt

import seaborn as sns
df.info()
print(type(df['Team1'][0]))
l=[]

for i in df['Team1']:

    l.append(i.strip())

df['Team1']=l

l=[]

for i in df['Team2']:

    l.append(i.strip())

df['Team2']=l #removing the extra spaces
l1=df['Team1'].unique()

l2=df['Team2'].unique()
en_l1={}

x=0

for i in l1:

    en_l1[i]=x

    x=x+1      
en_l1
df['Team1']=df['Team1'].map(en_l1)

df['Team2']=df['Team2'].map(en_l1)
df.head(10)
df.drop(df.columns[df.columns.str.contains('unnamed',case = False)],axis = 1, inplace = True)#dropping unnamed columns
l=[]

lis=df['Toss'].unique()

for i in lis:

    l.append(i)

d_l={}

x=0

for i in l:

    s=i[0]

    if(s=='8' or s=='4' or s=='.'):

        d_l[i]=int(10000)

    else:

        d_l[i]=int(x)

        x=x+1
df['Toss']=df['Toss'].map(d_l)

d_l

l=df['Place'].unique()

x=0

en_pl={}

for i in l:

    en_pl[i]=x

    x=x+1

df['Place']=df['Place'].map(en_pl)

df.head()
lis=df['Tied'].unique()

print("Match" in lis[1])
d={}

for i in lis:

    if(type(i)== float):

        d[i]=0

    elif ("Match abandoned" in i):

        d[i]=1

    elif("Match tied" in i):

        d[i]=2

    

df['Tied']=df['Tied'].map(d)
df['won_runs'].unique()
lis=df['won_runs'].unique()

d={}

for i in lis:

    if(type(i)==float):

        d[i]=0

    else:

        d[i]=int(i[0:3])

df['won_runs']=df['won_runs'].map(d)
lis=df['won_wickets'].unique()

d={}

for i in lis:

    if(type(i)==float):

        d[i]=0

    else:

        d[i]=int(i[0:3])

df['won_wickets']=df['won_wickets'].map(d)
lis=df['Result'].unique()

l=[]

for r in lis:

    l.append(r.split(" won")[0])

s=[]

for i in df['Result']:

    s.append(i.split(" won")[0].lower())

df['Result']=s      

df.head()
en_l={}

x=0

for i in l1:

    en_l[i.lower()]=x

    x=x+1

for i in l:

    if ("Match abandoned" in i):

        en_l[i]=20

    elif("Match tied" in i):

        en_l[i]=21

   

df['Result']=df['Result'].map(en_l)  
en_l
en_l1
df.head()
df.info()
dec={'bat':1,'bowl':2}

df['TossDecision']=df['TossDecision'].map(dec)
df.head()
df['Team1'].value_counts().sort_index()
df['Team2'].value_counts().sort_index()
teams=df['Team1'].value_counts().sort_index()+df['Team2'].value_counts().sort_index()
plt.figure(figsize=(10,10))

plt.bar([0,1,2,3,4,5,6,7,8,9,10,11,12],teams,color='g',width=0.75)
en_l1
year=df['Year'].value_counts().sort_index()

plt.figure(figsize=(10,10))

plt.pie(year,labels=[2008,2009,2010,2011,2012,2013,2014,2015,2016,2017,2018,2019],autopct='%1.2f%%')
dec=df['TossDecision'].value_counts().sort_index()

plt.figure(figsize=(10,10))

plt.bar(['Bat','bowl'],dec)
sns.heatmap(df.corr(),vmax=1,annot=True,fmt=".2f")
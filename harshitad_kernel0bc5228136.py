# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/2016 School Explorer.csv')
df.head()
#% of null values column wise

((df.isnull().sum())/1272)*100
#Removing % from the columns

df["Percent ELL"]=df["Percent ELL"].replace('[%$]','',regex=True).astype(float) 

df["Percent Asian"]=df["Percent Asian"].replace('[%$]','',regex=True).astype(float) 

df["Percent Black"]=df["Percent Black"].replace('[%$]','',regex=True).astype(float) 

df["Percent Hispanic"]=df["Percent Hispanic"].replace('[%$]','',regex=True).astype(float) 

df["Percent Black / Hispanic"]=df["Percent Black / Hispanic"].replace('[%$]','',regex=True).astype(float) 

df["Percent White"]=df["Percent White"].replace('[%$]','',regex=True).astype(float) 

df["Student Attendance Rate"]=df["Student Attendance Rate"].replace('[%$]','',regex=True).astype(float) 

df["Percent of Students Chronically Absent"]=df["Percent of Students Chronically Absent"].replace('[%$]','',regex=True).astype(float) 

df["Rigorous Instruction %"]=df["Rigorous Instruction %"].replace('[%$]','',regex=True).astype(float) 

df["Collaborative Teachers %"]=df["Collaborative Teachers %"].replace('[%$]','',regex=True).astype(float) 

df["Supportive Environment %"]=df["Supportive Environment %"].replace('[%$]','',regex=True).astype(float) 

df["Effective School Leadership %"]=df["Effective School Leadership %"].replace('[%$]','',regex=True).astype(float)

df["Strong Family-Community Ties %"]=df["Strong Family-Community Ties %"].replace('[%$]','',regex=True).astype(float)

df["Trust %"]=df["Trust %"].replace('[%$]','',regex=True).astype(float)



#cleaning some more data

df["School Income Estimate"]=df["School Income Estimate"].replace('[\$,]','',regex=True)

df["School Income Estimate"]=df["School Income Estimate"].astype(float)



df['School Income Estimate'].fillna(df['School Income Estimate'].mean(), inplace = True)
from matplotlib import pyplot as plt

import seaborn as sns



plt.figure(figsize=(10,8))



sns.countplot(y='City', data=df,orient = 'h' )

plt.xlabel('Count of schools')

plt.ylabel('Cities')

plt.title('Distribution of schools in US cities')

sns.countplot(x='Community School?',data=df)

siegroupbycomm=df.groupby('Community School?')['School Income Estimate']

fig = plt.figure(figsize=(10,6))

for comm,item in siegroupbycomm:

    sns.kdeplot(item, shade=True,)

    plt.legend(labels=['Non Community School','Community School'])

    plt.title('Community School VS Not Community School Income')


df['Economic Need Index'].fillna(df['Economic Need Index'].mean(), inplace = True) 

 

n, bins, patches = plt.hist(x=df['Economic Need Index'], bins=100, color='#0504aa')



sns.lmplot(x='School Income Estimate', y='Economic Need Index', data=df,

           fit_reg=False)
GBCOMMUNITY=df.groupby('Community School?')['Economic Need Index','School Income Estimate']

for key, item in GBCOMMUNITY:

    print("Community School: ",key)

    print(GBCOMMUNITY.get_group(key), "\n\n")

    print("Mean ENI for Community School(",key,") is: ",GBCOMMUNITY.get_group(key)['Economic Need Index'].mean())

    print("Mean SIE for Community School(",key,") is: ",GBCOMMUNITY.get_group(key)['School Income Estimate'].mean())

    
bhbycommunity=df.groupby('Community School?')['Percent Black / Hispanic'].mean()

bhbycommunity
features = ['Rigorous Instruction %',

'Collaborative Teachers %',

'Supportive Environment %',

'Effective School Leadership %',

'Strong Family-Community Ties %',

'Trust %']



df[['School Name'] + features ]
corrFeatures=['Economic Need Index','School Income Estimate','Percent Asian','Percent Black','Percent Hispanic','Percent Black / Hispanic','Percent White','Student Attendance Rate','Percent of Students Chronically Absent','Rigorous Instruction %','Collaborative Teachers %','Supportive Environment %','Effective School Leadership %','Strong Family-Community Ties %','Trust %']

#sns.heatmap(df[corrFeatures])

#df.dtypes

fig = plt.figure(figsize=(15,6))

dfcorr=df[corrFeatures].corr()

sns.heatmap(dfcorr,cmap="YlGnBu",annot=True)
df1=df[['School Name','District','Percent Asian','Percent Black','Percent Hispanic','Percent White']]

grouped= df1.groupby('District')



abc= grouped[['Percent Asian','Percent Black','Percent Hispanic','Percent White']].agg(np.mean)

result=abc.reset_index()

result.head()
from matplotlib import pyplot as plt



sns.set(style="whitegrid")

plt.figure(figsize=(40,15))

result.set_index(result['District']).plot(kind='bar', stacked=True)

 
NYC=df[(df.City=="NEW YORK")]

nyc_grouped=NYC[['Percent Asian','Percent Black','Percent Hispanic','Percent White']].agg(np.mean)

nyc_grouped=nyc_grouped.reset_index()





sns.countplot(NYC["Community School?"],palette="vlag")

nyc_grouped.columns

plt.pie(nyc_grouped[0], labels=nyc_grouped['index'],startangle=90, autopct='%.1f%%')

plt.show()
sns.countplot(NYC["Grade Low"],palette="rainbow")
sns.countplot(NYC["Grade High"],palette="rainbow")
NYC['Percent of Students Chronically Absent']=NYC['Percent of Students Chronically Absent'].replace('[%$,]','',regex=True)



NYC['Percent of Students Chronically Absent']=NYC['Percent of Students Chronically Absent'].astype(float) 

NYC['Percent of Students Chronically Absent']

NYC['Rigorous Instruction %']=NYC['Rigorous Instruction %'].replace('[%$,]','',regex=True)



NYC['Rigorous Instruction %']=NYC['Rigorous Instruction %'].astype(float) 



absent_30 = NYC[NYC['Percent of Students Chronically Absent']>.30]

absent30_grouped=absent_30[['Percent Asian','Percent Black','Percent Hispanic','Percent White']].agg(np.mean)

absent30_grouped=absent30_grouped.reset_index()

plt.pie(absent30_grouped[0], labels=absent30_grouped['index'],startangle=90, autopct='%.1f%%')

plt.show()





n, bins, patches = plt.hist(x=absent_30['Rigorous Instruction %'], bins=100, color='#0504aa') 
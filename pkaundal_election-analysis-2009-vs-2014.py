# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
%matplotlib inline
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))
df = pd.read_csv('../input/LS2009Candidate.csv')
Gen09 = df['Candidate Sex'].value_counts().tolist()

df = pd.read_csv('../input/LS2014Candidate.csv')
Gen14 = df['Candidate Sex'].value_counts().tolist()[:-1]
Gen = Gen09+Gen14
Data = {'Year':[2009,2009,2014,2014],'Gender':['Male','Female','Male','Female'],'Number':Gen}
DF = pd.DataFrame(data=Data)
sb.set_style('darkgrid')
x=np.arange(2)
y=[0,800]
val = [Gen[0],Gen[2],Gen[1],Gen[3]]
for a,b in zip(x,val[:2]):
    plt.text(a,b,str(b),position=(-0.28+a,4000))
for a,b in zip(x,val[2:]):
    plt.text(a,b,str(b),position=(0.15+a,230))
plt.title('Male and Female Candidates in 2009  and 2014')
sb.barplot(x=DF.Year,y=DF.Number,hue=DF.Gender)
df = pd.read_csv('../input/LS2009Candidate.csv')
df = df[df['Position']==1]
DF = df['Party Abbreviation'].value_counts().head(6).to_dict()
S = sum(df['Party Abbreviation'].value_counts().tolist())
DF['Other Regional Parties'] = S - sum(df['Party Abbreviation'].value_counts().head(6).tolist())

df2 = pd.read_csv('../input/LS2014Candidate.csv')
df2 = df2[df2['Position']==1]
DF2 = df2['Party Abbreviation'].value_counts().head(6).to_dict()
S2 = sum(df2['Party Abbreviation'].value_counts().tolist())
DF2['Other Regional Parties'] = S - sum(df2['Party Abbreviation'].value_counts().head(6).tolist())

fig = plt.figure()

ax1 = fig.add_axes([0, 0, .5, .5], aspect=1)
ax1.pie(DF.values(),labels=DF.keys(),autopct='%1.1f%%',shadow=True,explode=(0.06,0,0,0,0,0,0),pctdistance=0.8,radius = 2)
ax2 = fig.add_axes([0.8, .0, .5, .5], aspect=1)
ax2.pie(DF2.values(),labels=DF2.keys(),autopct='%1.1f%%',shadow=True,explode=(0.06,0,0,0,0,0,0),pctdistance=0.8,radius = 2)

ax1.set_title('2009',loc='center',fontdict={'fontsize':20},position=(0.5,1.55))
ax2.set_title('2014',loc='center',fontdict={'fontsize':20},position=(0.5,1.55))
plt.show()
plt.style.use('seaborn-deep')
df = pd.read_csv('../input/LS2009Candidate.csv')
df=df[df['Position'] == 1]
DF = df['Candidate Age'].tolist()
df = pd.read_csv('../input/LS2014Candidate.csv')
df=df[df['Position'] == 1]
DF2 = df['Candidate Age'].tolist()
bins = np.linspace(20, 90, 10)
plt.hist([DF, DF2], bins, label=['2009', '2014'])
plt.legend(loc='upper right')
plt.xlabel('Age Of winners in years')
plt.ylabel('Total Number of winners')
plt.title('Distribution of Age of the winners')
plt.show()
df = pd.read_csv('../input/LS2009Electors.csv')
df = df.groupby('STATE').mean()
DF = df[['POLL PERCENTAGE']].sort_values('POLL PERCENTAGE',ascending=False).to_dict()
Y09=[2009 for i in range(35)]
S09=list(DF['POLL PERCENTAGE'].keys())
P09=list(DF['POLL PERCENTAGE'].values())

df = pd.read_csv('../input/LS2014Electors.csv')
df = df.groupby('STATE').mean()
DF = df[['POLL PERCENTAGE']].sort_values('POLL PERCENTAGE',ascending=False).to_dict()
Y14=[2014 for i in range(35)]
S14=list(DF['POLL PERCENTAGE'].keys())
P14=list(DF['POLL PERCENTAGE'].values())
Data = {'YEAR':Y09+Y14,'STATE':S09+S14,'Poll_Percentage':P09+P14}
DF = pd.DataFrame(data=Data)
ax = plt.subplots(figsize=(6, 20))
sb.barplot(x=DF.Poll_Percentage,y=DF.STATE,hue=DF.YEAR)
plt.title('Poll Percentage of States 2009 and 2014')
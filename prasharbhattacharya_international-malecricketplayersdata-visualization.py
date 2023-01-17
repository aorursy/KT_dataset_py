import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
df = pd.read_csv('../input/international-cricket-players-data/personal_male.csv')

df1 = pd.read_csv('../input/international-cricket-players-data/personal_male.csv')

df.info()
df.isnull().sum()
df.head()
df['nationalTeam'].value_counts()

df['nationalTeam'].fillna(value='England', inplace=True)

df1['nationalTeam'].value_counts()

df1['nationalTeam'].fillna(value='England', inplace=True)
df['battingStyle'].value_counts()

df['battingStyle'].fillna(value='Right-hand bat', inplace=True)

df1['battingStyle'].value_counts()

df1['battingStyle'].fillna(value='Right-hand bat', inplace=True)
df['bowlingStyle'].value_counts()

df['bowlingStyle'].fillna(value='Right-arm medium', inplace=True)

df1['bowlingStyle'].value_counts()

df1['bowlingStyle'].fillna(value='Right-arm medium', inplace=True)
df.isnull().sum()
df1['total'] = range(1,len(df)+1)
subject =['England','Australia','India','South Africa','Pakistan',

          'New Zealand','Sri Lanka','Zimbabwe','Barbados','Bangladesh',

          'Jamaica','Kenya','Scotland','Afghanistan','Netherlands']

perc = [(687/3856)*100,(501/3856)*100,(468/3856)*100,(369/3856)*100,(340/3856)*100,(309/3856)*100,(196/3856)*100,

        (127/3856)*100,(99/3856)*100,(96/3856)*100,(85/3856)*100,(52/3856)*100,(46/3856)*100,(39/3856)*100,

        (37/3856)*100]

explode = (0,0.1,0,0,0,0,0,0,0,0,0,0,0,0,0)

plt.pie(perc,explode=explode,labels=subject,autopct='%1.2f%%',shadow=True,rotatelabels=False,radius=3)

plt.show()
fig, (axis1) = plt.subplots(1, figsize=(8,20))

sns.barplot(df1['total'],df1['country'], hue=df['battingStyle'], ax=axis1, palette='winter_r').set_title('Batting Style')
India = df[df['country']=='India']

India['total players of India'] = range(1,len(India)+1)

fig, (axis1) = plt.subplots(1, figsize=(8,10))

sns.barplot(India['total players of India'],India['bowlingStyle']).set_title('Bowling style of India')
subject = ['Right-hand bat','Left-hand bat']

perc = [(398/468)*100,(70/468)*100]

explode = (0,0.3)

plt.pie(perc,explode=explode,labels=subject,autopct='%1.2f%%',shadow=True,rotatelabels=False,radius=2)

plt.title('Batting style of India')

plt.show()
Pakistan = df[df['country']=='Pakistan']

Pakistan['total players of Pakistan'] = range(1,len(Pakistan)+1)

fig, (axis1) = plt.subplots(1, figsize=(8,6))

sns.barplot(Pakistan['total players of Pakistan'],Pakistan['bowlingStyle']).set_title('Bowling style of Pakistan')
perc = [(285/340)*100,(55/340)*100]

explode = (0,0.2)

plt.pie(perc,explode=explode,labels=subject,autopct='%1.2f%%',shadow=True,rotatelabels=False,radius=2)

plt.title('Batting style of Pakistan')

plt.show()
Australia = df[df['country']=='Australia']

Australia['total players of Australia'] = range(1,len(Australia)+1)

fig, (axis1) = plt.subplots(1, figsize=(8,12))

sns.barplot(Australia['total players of Australia'],Australia['bowlingStyle']).set_title('Bowling style of Australia')
perc = [(389/501)*100,(112/501)*100]

explode = (0,0.2)

plt.pie(perc,explode=explode,labels=subject,autopct='%1.2f%%',shadow=True,rotatelabels=False,radius=2)

plt.title('Batting style of Australia')

plt.show()
England = df[df['country']=='England']

England['total players of England'] = range(1,len(England)+1)

fig, (axis1) = plt.subplots(1, figsize=(8,18))

sns.barplot(England['total players of England'],England['bowlingStyle']).set_title('Bowling style of England')
perc = [(564/687)*100,(123/687)*100]

explode = (0,0.3)

plt.pie(perc,explode=explode,labels=subject,autopct='%1.2f%%',shadow=True,rotatelabels=False,radius=2)

plt.title('Batting style of England')

plt.show()
SouthAfrica = df[df['country']=='South Africa']

SouthAfrica['total players of SouthAfrica'] = range(1,len(SouthAfrica)+1)

fig, (axis1) = plt.subplots(1, figsize=(8,10))

sns.barplot(SouthAfrica['total players of SouthAfrica'],SouthAfrica['bowlingStyle']).set_title('Bowling style of South Africa')
perc = [(315/369)*100,(54/369)*100]

explode = (0,0.3)

plt.pie(perc,explode=explode,labels=subject,autopct='%1.2f%%',shadow=True,rotatelabels=False,radius=2)

plt.title('Batting style of South Africa')

plt.show()
NewZealand = df[df['country']=='New Zealand']

NewZealand['total players of NewZealand'] = range(1,len(NewZealand)+1)

fig, (axis1) = plt.subplots(1, figsize=(8,6))

sns.barplot(NewZealand['total players of NewZealand'],NewZealand['bowlingStyle']).set_title('Bowling style of New Zealand')
perc = [(239/309)*100,(70/309)*100]

explode = (0,0.3)

plt.pie(perc,explode=explode,labels=subject,autopct='%1.2f%%',shadow=True,rotatelabels=False,radius=2)

plt.title('Batting style of New Zland')

plt.show()
SriLanka = df[df['country']=='Sri Lanka']

SriLanka['total players of SriLanka'] = range(1,len(SriLanka)+1)

fig, (axis1) = plt.subplots(1, figsize=(8,6))

sns.barplot(SriLanka['total players of SriLanka'],SriLanka['bowlingStyle']).set_title('Bowling style of Sri Lanka')
perc = [(136/196)*100,(60/196)*100]

explode = (0,0.3)

plt.pie(perc,explode=explode,labels=subject,autopct='%1.2f%%',shadow=True,rotatelabels=False,radius=2)

plt.title('Batting style of Sri Lanka')

plt.show()
Zimbabwe = df[df['country']=='Zimbabwe']

Zimbabwe['total players of Zimbabwe'] = range(1,len(Zimbabwe)+1)

fig, (axis1) = plt.subplots(1, figsize=(8,5))

sns.barplot(Zimbabwe['total players of Zimbabwe'],Zimbabwe['bowlingStyle']).set_title('Bowling style of Zimbabwe')
perc = [(104/127)*100,(23/127)*100]

explode = (0,0.3)

plt.pie(perc,explode=explode,labels=subject,autopct='%1.2f%%',shadow=True,rotatelabels=False,radius=2)

plt.title('Batting style of South Zimbabwe')

plt.show()
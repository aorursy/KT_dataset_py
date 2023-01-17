import pandas as pd

import numpy as np

import math

import json

import matplotlib.pyplot as plt

%matplotlib inline

import datetime

import seaborn as sns

from sklearn.metrics import fbeta_score, make_scorer
# read in the json files

portfolio = pd.read_json('../input/portfolio.json', orient='records', lines=True)

profile = pd.read_json('../input/profile.json', orient='records', lines=True)

transcript = pd.read_json('../input/transcript.json', orient='records', lines=True)
portfolio
# One hot encode channels data 

available_channels = []



#Differientiate the channels column

for channel in portfolio['channels']:

    for t in channel:

        if t not in available_channels:

            available_channels.append(t)



for channel in available_channels:

    portfolio[channel] = portfolio['channels'].apply(lambda x: 1 if channel in x else 0)  



portfolio.drop('channels', axis = 1, inplace=True) 



# ??? sono già int64

#cols_to_update = ['difficulty', 'duration','reward']

#for col in cols_to_update:

#    portfolio[col] = portfolio[col].apply(lambda x: int(x))



portfolio    
profile.head(100)
profile.shape
profile.groupby(['gender'])['gender'].count()
profile.info()
profile.describe()
profile.age.hist(figsize = (10,5))

plt.ylabel('Count', fontsize = 20)

plt.xlabel('Age', fontsize = 20)

plt.title('Count by age');
profile.head()
profile.income.hist(figsize = (10,5))

plt.ylabel('#', fontsize = 20)

plt.xlabel('Income', fontsize = 20)

plt.title('Income');
print(profile[profile['age']==118].count())

profile[['gender','income','age']][profile['age']==118].head()
# lean and fill data

profile['gender'].fillna('Unknown', inplace = True)

profile['income'] = profile['income'].apply(lambda x: float(x))

profile['age'] = profile['age'].apply(lambda x: int(x))

mean_val = profile.income.mean() 

profile['income'].fillna(mean_val, inplace = True)
# other tuning on data

profile['member_year'] = profile.became_member_on.apply(lambda x: int(str(x)[:4]))

profile['member_month'] = profile.became_member_on.apply(lambda x: int(str(x)[4:6]))

profile['member_day'] = profile.became_member_on.apply(lambda x: int(str(x)[6:]))

profile['member_date'] = profile.became_member_on.apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'))

profile.drop('became_member_on', axis = 1, inplace = True)

profile.head()
transcript.head()
transcript.shape
transcript.info()
# almost useless ;-)

transcript.groupby(['event']).describe()
transcript['value_type'] = transcript['value'].apply(lambda x: list(x.keys())[0]) 

transcript['value'] = transcript['value'].apply(lambda x: list(x.values())[0]) 

transcript['time'] = transcript['time'].apply(lambda x: int(x))

transcript.head()
#transcript[transcript['value_type']=='amount']

transcript.groupby(['value_type'])['value_type'].count()
gender_income = profile.groupby('gender', as_index=False).agg({'income':'mean'})

gender_income.plot(kind = 'bar', x = 'gender', y = 'income', legend=False, figsize=(40,10))

plt.hlines(mean_val, -100 , 1000)

plt.ylabel('Income', fontsize = 20)

plt.xlabel('Gender', fontsize = 20)

plt.title('Gender vs Income')

plt.text(1, mean_val + 1000, 'Average Income');
profile.gender.value_counts(normalize=True).plot('bar', figsize = (10,5))

plt.title('Gender Distribution')

plt.ylabel('User Percentage', fontsize = 15)

plt.xlabel('Gender', fontsize = 15);
# variare con mese-anno

membership_subs = profile[profile['member_year'] >= 2014].groupby(['member_year','member_month'], as_index=False).agg({'id':'count'})

plt.figure(figsize=(10,8))

sns.pointplot(x="member_month", y="id", hue="member_year", data = membership_subs)

plt.ylabel('Customer Subsciptions', fontsize = 12)

plt.xlabel('Month', fontsize = 12)

plt.title('Subsciptions by Month and Year');
portfolio.groupby('offer_type')['id'].count()
transcriptL = transcript.merge(portfolio,how='left',left_on='value', right_on='id')

transcriptL.groupby(['event','offer_type'])['offer_type'].count()
sns.countplot(data=transcriptL[transcriptL['event']!='transaction'],x='offer_type',hue='event');
transcriptL = transcriptL.merge(profile,how='left',left_on='person', right_on='id')
transcriptL.groupby(['gender', 'event'])['offer_type'].count()
sns.countplot(data=transcriptL,x='gender',hue='event');
transcriptL.groupby(['age', 'event'])['offer_type'].count()
transcriptL[transcriptL['value'] == 'fafdcd668e3743c1bb461111dcafc2a4'].groupby(['gender', 'event'])['event'].count()
transcriptL[transcriptL['person'] == '94de646f7b6041228ca7dec82adb97d2'].groupby(['gender', 'event', 'offer_type'])['event'].count()
transcriptR = transcript.merge(portfolio,how='right',left_on='value', right_on='id')

transcriptR = transcriptL.merge(profile,how='left',left_on='person', right_on='id')

transcriptR.groupby(['offer_type','value'])['value'].count()
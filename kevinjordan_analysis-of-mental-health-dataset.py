import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from collections import Counter

%matplotlib inline
data = pd.read_csv("../input/survey.csv")
#print(data.info())

print(data.head())
#data['Age'].value_counts()

#data.dropna(inplace=True)

#print(data.info())
def age_process(age):

    if age>=0 and age<=80:

        return age

    else:

        return np.nan

data['Age'] = data['Age'].apply(age_process)    
data['state'].isnull().sum()
#print(data.head())
data['work_interfere'].isnull().sum()
data['self_employed'].isnull().sum()
data['comments'].isnull().sum()
#data.fillna(0,inplace=True)

data.info()
fig,ax = plt.subplots(figsize=(8,4))

sns.distplot(data['Age'].dropna(),ax=ax,kde=False,color='green')

plt.title('Age Distribution')

plt.ylabel('Freq')
sns.heatmap(data.corr())
data.var()
country_count = Counter(data['Country'].tolist()).most_common(15)

country_idx = [country[0] for country in country_count]

country_val = [country[1] for country in country_count]

plt.subplots(figsize=(15,6))

sns.barplot(x = country_idx,y=country_val )

plt.title('Top fifteen country')

plt.xlabel('Country')

plt.ylabel('Count')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
state_count = Counter(data['state'].tolist()).most_common(20)

print(country_count)

state_idx = [state[0] for state in state_count]

state_val = [state[1] for state in state_count]

plt.subplots(figsize=(8,6))

sns.barplot(x = state_idx,y=state_val )

plt.title('Top twenty state')

plt.xlabel('State')

plt.ylabel('Count')

ticks = plt.setp(ax.get_xticklabels(),rotation=90)
plt.subplots(figsize=(8,4))

sns.countplot(data['work_interfere'].dropna())

plt.title('Work interfere Distribution')

plt.ylabel('Count')
print(data.shape,data.columns)
plt.subplots(figsize=(8,4))

sns.countplot(data['self_employed'].dropna())

plt.title('Self employed Analysis')

plt.ylabel('Count')
plt.subplots(figsize=(8,4))

sns.countplot(data['leave'].dropna())

plt.title('Leave Analysis')

plt.ylabel('Count')
plt.subplots(figsize=(8,4))

sns.countplot(data['tech_company'].dropna())

plt.title('Tech company Analysis')

plt.ylabel('Count')
plt.subplots(figsize=(8,4))

sns.countplot(data['family_history'].dropna())

plt.title('family history Analysis')

plt.ylabel('Count')
from matplotlib import pyplot as plt

import pandas as pd

import numpy as np

import seaborn as sns

import sys, os, csv
pd.set_option("display.max_columns", 500)

dataset = pd.read_csv("../input/crimes-in-boston/crime.csv", encoding='latin-1')

dataset.shape
dataset.info()
dataset.head()
dataset.columns
# Rename columns

rename = {

    'INCIDENT_NUMBER': 'incident_num',

    'OFFENSE_CODE':'code',

    'OFFENSE_CODE_GROUP':'code_group',

    'OFFENSE_DESCRIPTION':'description',

    'DISTRICT':'district',

    'REPORTING_AREA':'area',

    'SHOOTING':'shooting',

    'OCCURRED_ON_DATE':'date',

    'YEAR':'year',

    'MONTH':'month',

    'DAY_OF_WEEK':'day',

    'HOUR':'hour',

    'UCR_PART':'ucr_part',

    'STREET':'street',

    'Lat':'lat',

    'Long':'long',

    'Location':'location',

}



dataset.rename(columns=rename, inplace=True)
dataset['code_group'].value_counts()
dataset['ucr_part'].value_counts()
dataset['year'].value_counts()
dataset['shooting'].isnull().sum()
shooting = dataset['shooting'].copy()

shooting.fillna('N', inplace=True)

dataset['shooting'] = shooting



dataset['shooting'].head()
ucr_part = dataset['ucr_part'].copy()

ucr_part.replace(to_replace='Other', value='Part Four', inplace=True)

dataset['ucr_part'] = ucr_part
dataset['ucr_part'].value_counts()
dataset['ucr_part'].isnull().sum()
dataset[dataset['ucr_part'].isnull()]['code_group'].value_counts()
code_group = dataset['code_group'].copy()

code_group.replace(to_replace="INVESTIGATE PERSON", value="Investigate Person", inplace=True)

dataset['code_group'] = code_group
dataset.loc[(dataset['code_group'] == 'Investigate Person') & (dataset['ucr_part'].isnull()), 'ucr_part']= "Part Three"
dataset['ucr_part'].isnull().sum()
dataset.dropna(subset=['ucr_part'], inplace=True)
dataset['code_group'].value_counts().head()
order = dataset['code_group'].value_counts().head().index

order
plt.figure(figsize=(12,8))

sns.countplot(data=dataset, x='code_group', hue='district', order=order)
data2017 = dataset[dataset['year']==2017].groupby(['month','district']).count()

data2017.head()
plt.figure(figsize=(12,12))

sns.lineplot(data=data2017.reset_index(), x='month', y='code', hue='district')
day_num_name = {'Monday':'1','Tuesday':'2','Wednesday':'3','Thursday':'4','Friday':'5','Saturday':'6','Sunday':'7',}

dataset['day_num'] = dataset['day'].map(day_num_name)
data_day_hour = dataset[dataset['year']==2017].groupby(['day_num','hour']).count()['code'].unstack()

plt.figure(figsize=(8,8))

sns.heatmap(data=data_day_hour, cmap='viridis', yticklabels=['Monday','Tuesday','Wednesday','Thursday','Friday','S'])
df_day_hour_part1 = dataset[(dataset['year'] == 2017) & (dataset['code_group'] == 'Larceny')].groupby(['day_num','hour']).count()['code'].unstack()

plt.figure(figsize=(10,10))

sns.heatmap(data = df_day_hour_part1, cmap='viridis', yticklabels=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
dfpart1 = dataset[(dataset['year'] == 2017) & (dataset['ucr_part'] == 'Part One')].groupby(['code_group','shooting']).count().reset_index().sort_values('code',ascending=False)

dfpart2 = dataset[(dataset['year'] == 2017) & (dataset['ucr_part'] == 'Part One') & (dataset['shooting'] == 'Y')].groupby(['code_group','shooting']).count().reset_index().sort_values('code',ascending=False)
order1 = dataset[dataset['ucr_part'] == 'Part One']['code_group'].value_counts().head()

order1
order1 = dataset[dataset['ucr_part'] == 'Part One']['code_group'].value_counts().head(5).index

plt.figure(figsize=(12,8))

sns.countplot(data = dataset, x='code_group',hue='district', order = order1)
order2 = dataset[dataset['ucr_part'] == 'Part Two']['code_group'].value_counts().head()

order2
order2 = dataset[dataset['ucr_part'] == 'Part Two']['code_group'].value_counts().head(5).index

plt.figure(figsize=(12,8))

sns.countplot(data = dataset, x='code_group',hue='district', order = order2)
order3 = dataset[dataset['ucr_part'] == 'Part Three']['code_group'].value_counts().head().index

plt.figure(figsize=(12,8))

sns.countplot(data = dataset, x='code_group',hue='district', order = order3)

plt.figure(figsize=(16,8))

plt.tight_layout()

sns.set_color_codes("pastel")

ax = sns.barplot(y="code", x="code_group", data=dfpart1, hue='shooting')



ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
import numpy as np
import pandas as pd
import scipy.stats as st
pd.set_option('display.max_columns', None)

import math

import matplotlib.pyplot as plt
%matplotlib inline

import seaborn as sns
sns.set_style('whitegrid')

import missingno as msno

from sklearn.preprocessing import StandardScaler
from scipy import stats



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
data = pd.read_csv('/kaggle/input/bh-customers/customers.csv')
data.head()
data.shape
data.info()
data.describe()
data.describe(include=['object', 'bool'])
data.isnull().sum()
drop_row_index = data[data['turnover'].isnull()].index
data = data.drop(drop_row_index)
data.isnull().sum()
data['regdate'] = pd.to_datetime(data['regdate'])
data['Year'] = data['regdate'].dt.year
data['Month'] = data['regdate'].dt.month
plt.figure(figsize=(15, 10))
sns.countplot(x="Year", data=data, order = data.groupby(by=['Year'])['id'].count().sort_values(ascending=False).index)
plt.xticks(rotation=90)
plt.figure(figsize=(15, 10))
sns.countplot(x="Month", data=data, order = data.groupby(by=['Month'])['id'].count().sort_values(ascending=False).index)
plt.xticks(rotation=90)
data['YearMonth'] = data['Year']*100 +data['Month']
#plt.figure(figsize=(15, 10))
#sns.countplot(x="YearMonth", data=data, order = data.groupby(by=['YearMonth'])['id'].count().sort_values(ascending=False).index)
#plt.xticks(rotation=90)
plt.figure(figsize=(15, 10))
sns.countplot(x="YearMonth", data=data, order = data.groupby(by=['YearMonth'])['YearMonth'].count().index)
plt.xticks(rotation=90)
plt.figure(figsize=(30, 10))
sns.countplot(x="Year", data=data, hue='cshop', order=data.Year.value_counts().iloc[:3].index)
plt.xticks(size=16, rotation=90)
data_year = data.groupby(by=['cshop'])['turnover'].sum()
data_year = data_year.reset_index()
data_year.sort_values(by=['turnover'], ascending=False)
plt.figure(figsize=(15, 10))
sns.barplot(x="cshop", y="turnover", data=data_year)
plt.xticks(rotation=90)
year_max_df = data.groupby(['YearMonth', 'cshop']).size().reset_index(name='count')
year_max_idx = year_max_df.groupby(['YearMonth'])['count'].transform(max) == year_max_df['count']
year_max_genre = year_max_df[year_max_idx].reset_index(drop=True)
#year_max_genre = year_max_genre.drop_duplicates(subset=["YearMonth", "count"], keep='last').reset_index(drop=True)
year_max_genre.head()
genre = year_max_genre['cshop'].values
# genre[0]
plt.figure(figsize=(30, 15))
g = sns.barplot(x='YearMonth', y='count', data=year_max_genre)
index = 0
for value in year_max_genre['count'].values:
#     print(asd)
    g.text(index, value + 5, str(genre[index] + '----' +str(value)), color='#000', size=14, rotation= 90, ha="center")
    index += 1




plt.xticks(rotation=90)
plt.show()
data['agegroup'].value_counts()
plt.figure(figsize=(15, 10))
sns.countplot(x="agegroup", data=data, order = data['agegroup'].value_counts().index)
plt.xticks(rotation=90)
year_sale_dx = data.groupby(by=['agegroup', 'cshop'])['turnover'].sum().reset_index()
year_sale = year_sale_dx.groupby(by=['agegroup'])['turnover'].transform(max) == year_sale_dx['turnover']
year_sale_max = year_sale_dx[year_sale].reset_index(drop=True)
# year_sale_max
genre = year_sale_max['cshop']
plt.figure(figsize=(30, 18))
g = sns.barplot(x='agegroup', y='turnover', data=year_sale_max)
index = 0
for value in year_sale_max['turnover']:
    g.text(index, value + 1, str(genre[index] + '----' +str(round(value, 2))), color='#000', size=14, rotation= 90, ha="center")
    index += 1

plt.xticks(rotation=90)
plt.show()
data['gender'].value_counts()
explode = (0.1,0)  
fig1, ax1 = plt.subplots(figsize=(12,7))
ax1.pie(data['gender'].value_counts(), explode=explode,labels=['Կանայք','Տղամարդիկ'], autopct='%1.1f%%',
        shadow=True, startangle=90)
# Equal aspect ratio ensures that pie is drawn as a circle
ax1.axis('equal')  
plt.tight_layout()
plt.legend()
plt.show()
fig1, ax1 = plt.subplots(figsize=(12,7))
sns.countplot(data['agegroup'],hue=data['gender'])
data_genre = data.groupby(by=['gender'])['turnover'].sum()
data_genre = data_genre.reset_index()
data_genre = data_genre.sort_values(by=['turnover'], ascending=False)
# data_genre
plt.figure(figsize=(15, 10))
sns.barplot(x="gender", y="turnover", data=data_genre)
plt.xticks(rotation=90)
data['usage_ratio'] = data['used']/data['turnover']
data_platform = data.groupby(by=['gender'])['usage_ratio'].mean()
data_platform = data_platform.reset_index()
#data_platform = data_platform.sort_values(by=['Global_Sales'], ascending=False)
data_platform.head(100)
plt.figure(figsize=(15, 10))
sns.barplot(x="gender", y="usage_ratio", data=data_platform)
plt.xticks(rotation=90)
data_platform = data.groupby(by=['gender'])['used'].mean()
data_platform = data_platform.reset_index()
#data_platform = data_platform.sort_values(by=['Global_Sales'], ascending=False)
data_platform.head(100)
plt.figure(figsize=(15, 10))
sns.barplot(x="gender", y="used", data=data_platform)
plt.xticks(rotation=90)
data['agegroup'] = data['agegroup'].str.replace('տարեկան','')
comp_genre = data[['agegroup','collected','used']]
comp_genre
comp_map = comp_genre.groupby(by=['agegroup']).sum()/100000
comp_map
plt.figure(figsize=(15, 10))
sns.set(font_scale=1)
sns.heatmap(comp_map, annot=True, fmt = '.1f')

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()
fig2, ax2 = plt.subplots(figsize=(12,7))
sns.countplot(data['agegroup'],hue=data['sms'])
fig3, ax3 = plt.subplots(figsize=(12,7))
sns.countplot(data['gender'],hue=data['sms'])
plt.figure(figsize=(15, 10))
sns.barplot(x="sms", y="turnover", data=data)
#plt.xticks(rotation=90)
year_sale_dx = data.groupby(by=['agegroup','sms'])['turnover'].sum().reset_index()
year_sale = year_sale_dx.groupby(by=['agegroup','sms'])['turnover']#.transform(max) == year_sale_dx['turnover']

#year_sale_max = year_sale_dx[year_sale].reset_index(drop=True)
year_sale_dx
#fig5, ax5 = plt.subplots(figsize=(72,18))
genre = year_sale_dx['sms']

plt.figure(figsize=(50, 25))
g = sns.barplot(x=year_sale_dx['agegroup']+'_'+year_sale_dx['sms'], y="turnover", data=year_sale_dx)
index = 0
for value in year_sale_dx['turnover']:
    g.text(index, value + 1, str(genre[index] + '-' +str(round(value, 2))), color='#000', size=24, rotation= 0, ha="center")
    index += 1

plt.xticks(rotation=90)
plt.show()
#sns.barplot(x=year_sale_dx['agegroup']+'_'+year_sale_dx['sms'], y="turnover", data=year_sale_dx)
plt.figure(figsize=(13,10))
sns.heatmap(data.corr(), cmap = "Blues", annot=True, linewidth=3)
data_pair = data.loc[:,["agegroup", "cshop","used", "collected", "turnover", 'YearMonth', 'usage_ratio', 'Year']]
data_pair
sns.pairplot(data_pair, hue='cshop')
sns.pairplot(data_pair, hue='agegroup')
data_pair_log = data_pair.copy()
sale_columns = ["used", "collected", "turnover",'usage_ratio']
# for column in sale_columns:
#     if 0 in data[column].unique():
#         pass
#     else:
#         data_pair_log[column] = np.log(data_pair_log[column])
# #         data_pair_log.head()
data_pair_log = data_pair_log[data_pair_log.used != 0]
data_pair_log = data_pair_log[data_pair_log.collected != 0]
data_pair_log = data_pair_log[data_pair_log.turnover != 0]
data_pair_log = data_pair_log[data_pair_log.usage_ratio != 0]
data_pair_log
data_pair_log['used'] = np.log(data_pair_log['used']);
data_pair_log['collected'] = np.log(data_pair_log['collected']);
data_pair_log['turnover'] = np.log(data_pair_log['turnover']);
data_pair_log['turnover'] = np.log(data_pair_log['turnover']);
sns.pairplot(data_pair_log, hue='cshop',  palette="husl")
sns.pairplot(data_pair_log, hue='agegroup',  palette="husl")
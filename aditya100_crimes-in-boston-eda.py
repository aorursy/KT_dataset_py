# Importing Libraries

import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# Importing Dataset

crime_df = pd.read_csv('../input/crime.csv', encoding='latin-1')

oc_df = pd.read_csv('../input/offense_codes.csv', encoding='latin-1')
crime_df.head()
oc_df.head()
# Renaming the columns

oc_df = oc_df.rename(columns={'CODE': 'OFFENSE_CODE', 'NAME': 'OFFENSE_NAME'})
oc_df.head()
# Merging the dataframes on OFFENSE_CODE column

df = pd.merge(crime_df, oc_df, on='OFFENSE_CODE')
df.head()
# Converting to datetime format

df['DATE'] = pd.to_datetime(df['OCCURRED_ON_DATE'])
print('min/max date: %s / %s' % (df['DATE'].min(), df['DATE'].max()))

print('Number of days: %d' % ((df['DATE'].max() - df['DATE'].min()).days + 1))

print('Shape: %d rows' % df.shape[0])
df.isnull().sum()
total_cells = np.product(df.shape)

total_missing = df.isnull().sum().sum()



print('Percent of Missing Data: %d%s' % ((total_missing/total_cells)*100, '%'))
df['SHOOTING'].unique()
df['SHOOTING'] = df['SHOOTING'].apply(lambda x: 1 if x=='Y' else 0)
_ = sns.countplot(df['SHOOTING'])
df.isnull().sum()
df[df['UCR_PART'].isnull()]['OFFENSE_CODE'].unique()
total_cells = np.product(df.shape)

total_missing = df.isnull().sum().sum()



print('Percent of Missing Data: %s%s' % ((total_missing/total_cells)*100, '%'))
df = df.dropna()
df.isnull().sum()
df.head()
plt.figure(figsize=(20, 10))

p = sns.countplot(df['OFFENSE_CODE_GROUP'])

plt.title('Offense Code Group')

_ = plt.setp(p.get_xticklabels(), rotation=90)
df_year = df.groupby(['YEAR']).size().reset_index(name='counts')

df_month = df.groupby(['MONTH']).size().reset_index(name='counts')
fig, axs = plt.subplots(2,2)

fig.set_figheight(15)

fig.set_figwidth(15)



p = sns.countplot(df['DAY_OF_WEEK'], ax=axs[0, 0])

q = sns.lineplot(x=df_month['MONTH'], y=df_month['counts'], ax=axs[1, 0], color='r')

r = sns.lineplot(x=df_year['YEAR'], y=df_year['counts'], ax=axs[0,1], color='g')

s = sns.countplot(df['DISTRICT'], ax=axs[1,1])

df_hour = df.groupby(['HOUR']).size().reset_index(name='counts')
fig, axs = plt.subplots(1,2)

fig.set_figheight(5)

fig.set_figwidth(15)



p = sns.countplot(df['HOUR'], ax=axs[0])

q = sns.lineplot(x=df_hour['HOUR'], y=df_hour['counts'], ax=axs[1], color='y')
df_date = df.groupby(['OCCURRED_ON_DATE']).size().reset_index(name='counts')

df_date['date'] =df_date.apply(lambda x: pd.to_datetime(x['OCCURRED_ON_DATE'].split(' ')[0]), axis=1)
plt.figure(figsize=(20, 10))

p = sns.lineplot(x=df_date['date'], y=df_date['counts'], color='r')
df.Lat.replace(-1, None, inplace=True)

df.Long.replace(-1, None, inplace=True)
plt.figure(figsize=(10, 10))

p = sns.scatterplot(x='Lat', y='Long', hue='DISTRICT',alpha=0.01, data=df)
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
import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

df=pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv')

df.head()
df.columns
for vars in df.columns:

  rows=df.shape[0]

  cols=df.shape[1]

print(f'number of rows : {rows}')

print(f'number of cols : {cols}')
df.info()
df.describe()


for vars in df.columns:

  print(f'no. of unique values in {vars} : {df[vars].nunique()}\n')
df.isna().sum()
df['HDI for year'].isna().sum()/len(df)*100
df.drop(['HDI for year','country-year'], axis=1, inplace=True)

df.head()
df=df.rename(columns={'country':'Country',

                      'year':'Year',

                      'sex':'Sex',

                      'age':'Age',

                      'suicides_no':'No_of_suicides',

                      'population':'Population',

                      'suicides/100k pop':'Suicides_per_100k_pop',

                      'gdp_for_year ($)	':'GDP_in_Dollars',

                      'gdp_per_capita ($)	':'GDP_per_Capita_Dollars',

                      'generation':'Generation'})
unique_cols=['Country','Sex','Age','Generation']

for vars in unique_cols:

  print(f'unique values in {vars} : {df[vars].unique()}\n')
#univariate analysis

plt.figure(figsize=(20,7))

sns.countplot(df['Country'])

plt.xticks(rotation=90)

plt.figure(figsize=(20,7))

sns.countplot(df['Year'])

plt.xticks(rotation=90)
plt.figure(figsize=(10,7))

sns.countplot(df['Sex'])

plt.xticks(rotation=90)
plt.figure(figsize=(10,7))

sns.countplot(df['Age'])

plt.xticks(rotation=90)
plt.figure(figsize=(10,7))

sns.countplot(df['Generation'])

plt.xticks(rotation=90)
df[' gdp_for_year ($) ']=df[' gdp_for_year ($) '].apply(lambda x: x.replace(',',''))

df[' gdp_for_year ($) ']=df[' gdp_for_year ($) '].astype('float64')
# kde plots

kde_plots=['No_of_suicides','Population','Suicides_per_100k_pop',' gdp_for_year ($) ','gdp_per_capita ($)']

fig,axes=plt.subplots(3,2, figsize=(20,10))

for i, j in enumerate(kde_plots):

  ax=axes[int(i/2), i%2]

  sns.kdeplot(df[j], ax=ax)

fig.tight_layout()
kde_plots=['No_of_suicides','Population','Suicides_per_100k_pop',' gdp_for_year ($) ','gdp_per_capita ($)']

fig,axes=plt.subplots(3,2, figsize=(20,10))

for i, j in enumerate(kde_plots):

  ax=axes[int(i/2), i%2]

  sns.boxplot(df[j], ax=ax)

fig.tight_layout()
country_by_suicide_bott15=df.groupby(['Country'])['No_of_suicides'].sum().sort_values(ascending=True)[:15]
country_by_suicide_bott15
country_by_suicide_bott15.plot(kind='bar', figsize=(20,6))

plt.ylabel('No. of suicides')
country_by_suicide_top15=df.groupby(['Country'])['No_of_suicides'].sum().sort_values(ascending=False)[:15]
country_by_suicide_top15
country_by_suicide_top15.plot(kind='bar', figsize=(20,6))

plt.ylabel('No. of suicides')
country_by_suicideper100_bott15=df.groupby(['Country'])['Suicides_per_100k_pop'].mean().sort_values(ascending=True)[:15]
country_by_suicideper100_bott15
country_by_suicideper100_bott15.plot(kind='bar', figsize=(20,6))

plt.ylabel('No. of suicides')
country_by_suicideper100_top15=df.groupby(['Country'])['Suicides_per_100k_pop'].mean().sort_values(ascending=False)[:15]
country_by_suicideper100_top15
country_by_suicideper100_top15.plot(kind='bar', figsize=(20,6))

plt.ylabel('Suicides per 100K pop')
plt.figure(figsize=(20,6))

sns.lineplot(x=df['Year'], y=df['No_of_suicides'])
plt.figure(figsize=(20,6))

sns.lineplot(x=df['Year'], y=df['Suicides_per_100k_pop'], )
plt.figure(figsize=(20,6))

sns.lineplot(x=df['Year'], y=df['No_of_suicides'], hue=df['Sex'])
plt.figure(figsize=(20,6))

sns.lineplot(x=df['Year'], y=df['Suicides_per_100k_pop'], hue=df['Sex'])
generation_suicide=df.groupby('Generation')['No_of_suicides'].sum()
generation_suicideper100k=df.groupby('Generation')['Suicides_per_100k_pop'].mean()
generation_suicideper100k
generation_suicideper100k.plot(kind='bar', figsize=(20,6))

plt.ylabel('No. of suicides')
generation_suicide
generation_suicide.plot(kind='bar', figsize=(20,6))

plt.ylabel('no. of suicides')
generation_suicide_gender=df.groupby(['Generation','Sex'])['No_of_suicides'].sum()
generation_suicide_gender
generation_suicide_gender.plot(kind='bar', figsize=(20,6))

plt.ylabel('no. of suicides')
Age_suicide=df.groupby('Age')['No_of_suicides'].sum()
Age_suicide
Age_suicide.plot(kind='bar', figsize=(20,6))

plt.ylabel('no. of suicides')
Age_suicideper100k=df.groupby(['Age'])['Suicides_per_100k_pop'].mean()
Age_suicideper100k.plot(kind='line',figsize=(20,6))

plt.ylabel('Suicide per 100K population')
Age_suicideper100k.plot(kind='pie',figsize=(20,6))
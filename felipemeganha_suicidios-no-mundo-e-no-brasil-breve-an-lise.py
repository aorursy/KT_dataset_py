import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/master.csv')
df.head() 
df.info()
df.isnull().sum()
df.drop(['HDI for year'], axis=1, inplace=True)

df.isnull().sum()
unique_country = df['country'].unique



alpha = 0.7

plt.figure(figsize=(10,25))

sns.countplot(y='country', data=df, alpha=alpha)

plt.title('DataBase por pais')

plt.show()


plt.figure(figsize=(16,7))

cor = sns.heatmap(df.corr(), annot=True)


plt.figure(figsize=(16,7))

bar_age = sns.barplot(x='sex', y='suicides_no', hue='age', data=df)
#Idade das pessoas que mais se matam

plt.figure(figsize=(16,7))

bar_age = sns.barplot(x='sex', y='suicides_no', hue='generation', data=df)
# Suicidios de acordo com ano e idade

age_5 =  df.loc[df.loc[:, 'age']=='5-14 years',:]

age_15 = df.loc[df.loc[:, 'age']=='15-24 years',:]

age_25 = df.loc[df.loc[:, 'age']=='25-34 years',:]

age_35 = df.loc[df.loc[:, 'age']=='35-54 years',:]

age_55 = df.loc[df.loc[:, 'age']=='55-74 years',:]

age_75 = df.loc[df.loc[:, 'age']=='75+ years',:]



plt.figure(figsize=(16,7))

age_5_lp = sns.lineplot(x='year', y='suicides_no', data=age_5)

age_15_lp = sns.lineplot(x='year', y='suicides_no', data=age_15)

age_25_lp = sns.lineplot(x='year', y='suicides_no', data=age_25)

age_35_lp = sns.lineplot(x='year', y='suicides_no', data=age_35)

age_55_lp = sns.lineplot(x='year', y='suicides_no', data=age_55)

age_75_lp = sns.lineplot(x='year', y='suicides_no', data=age_75)



leg = plt.legend(['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years'])
#Suicidio no Brasil



Brazil =  df.loc[df['country']=='Brazil',]

Brazil.reset_index(drop=True,inplace=True)
Brazil.head()
Brazil_s = sns.barplot(x='age',y='population',data=Brazil, palette='winter',

                order=['5-14 years','15-24 years','25-34 years','35-54 years',

                      '55-74 years','75+ years'])

plt.figure(figsize=(16,7))

Brazil_s.set_xticklabels(Brazil_s.get_xticklabels(), rotation=30)

Brazil_s.set_xlabel('Idade')

Brazil_s.set_ylabel('Num. de Suicidios')

Brazil_s.set_title('Num. de Suicidios por Grupo de Idade no Brazil: 1979-2015')
Br_decades = Brazil.loc[Brazil['year'].isin(['1985', '1995', '2005', '2015'])]
sns.set_style("whitegrid")

g=sns.catplot(x="age",y="suicides/100k pop",  col='sex', hue="year", kind="bar",palette='PRGn', data=Br_decades,order=['5-14 years','15-24 years','25-34 years','35-54 years',

                      '55-74 years','75+ years']).set_xticklabels(rotation=90)



(g.despine(left=True),g.set_axis_labels("", "Number of Suicides per 100k people"))
plt.figure(figsize=(16,7))

bar_age = sns.barplot(x='sex', y='suicides_no', hue='generation', data=Brazil)
#Idade das pessoas que mais se matam

plt.figure(figsize=(16,7))

bar_age = sns.barplot(x='sex', y='suicides_no', hue='age', data=Brazil)
# Suicidios de acordo com ano e idade

age_5 =  Brazil.loc[Brazil.loc[:, 'age']=='5-14 years',:]

age_15 = Brazil.loc[Brazil.loc[:, 'age']=='15-24 years',:]

age_25 = Brazil.loc[Brazil.loc[:, 'age']=='25-34 years',:]

age_35 = Brazil.loc[Brazil.loc[:, 'age']=='35-54 years',:]

age_55 = Brazil.loc[Brazil.loc[:, 'age']=='55-74 years',:]

age_75 = Brazil.loc[Brazil.loc[:, 'age']=='75+ years',:]



plt.figure(figsize=(16,7))

age_5_lp = sns.lineplot(x='year', y='suicides_no', data=age_5)

age_15_lp = sns.lineplot(x='year', y='suicides_no', data=age_15)

age_25_lp = sns.lineplot(x='year', y='suicides_no', data=age_25)

age_35_lp = sns.lineplot(x='year', y='suicides_no', data=age_35)

age_55_lp = sns.lineplot(x='year', y='suicides_no', data=age_55)

age_75_lp = sns.lineplot(x='year', y='suicides_no', data=age_75)



leg = plt.legend(['5-14 years', '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years'])
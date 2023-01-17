import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# load the data file

df = pd.read_csv("../input/master.csv")

df.head()
df.shape
# Check out the brief information of each column

df.info()
# Check out if there is any duplicated data

df.duplicated().sum()
# Check out the unique data of some columns

df.nunique()
df['age'].unique()
df['generation'].unique()
# Returns valid descriptive statistics for each column of data

df.describe()
# Rename columns

df.rename(columns = {'suicides/100k pop': 'suicides_100k_pop', 'country-year': 'country_year', 'HDI for year': 'hdi_for_year'}, inplace=True)
df.rename(columns = {' gdp_for_year ($) ': 'gdp_for_year', 'gdp_per_capita ($)': 'gdp_per_capita'}, inplace=True)
# Change the sex values to simpler forms

df.loc[df['sex'] == 'male', 'sex'] = 'M'

df.loc[df['sex'] == 'female', 'sex'] = 'F'
# Change the age values to simpler forms

df['age'] = df.loc[df['age'].str.contains('years'), 'age'].apply(lambda x: x[:-6])
# Replace the "NaN" to 0

df.fillna(0, inplace=True)
data = df.copy()
subtable = data.pivot_table('suicides_no', index=['year'], columns=['sex'], aggfunc='sum', margins=True)

subtable['F_prop'] = subtable['F'] / subtable['All']*100

subtable['M_prop'] = subtable['M'] / subtable['All']*100
suicide_sex = pd.DataFrame(subtable)

suicide_sex.drop('All', inplace=True)
labels = suicide_sex.index

M = suicide_sex['M_prop']

F = suicide_sex['F_prop']



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize=(16,8))

sns.set(style="white")

rects1 = ax.bar(x - width/2, M, width, label='Male', color = 'royalblue', alpha=.8)

rects2 = ax.bar(x + width/2, F, width, label='Female', color = 'goldenrod', alpha=.8)

ax1 = ax.twinx()

rects3 = ax1.plot(x, M, 'b')

rects4 = ax1.plot(x, F, 'orange')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_xlabel('Years', fontsize=14)

ax.set_ylabel('Suicide Percent', fontsize=14)

ax.set_title('Suicide Percent by Gender', fontsize=16)

ax.set_xticks(x)

ax.set_xticklabels(labels, rotation=30)

ax.legend()

ax1.legend(loc=1)



plt.show()
population = data.groupby('year')['suicides_no', 'population'].sum()

population['suicides_prop'] = population['suicides_no'] / population['population'] * 100
sns.set_style("white")

labels = population.index

y1 = population['suicides_no']

y2 = population['suicides_prop']

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(16,8))

ax.bar(x, y1,alpha=.8,color='royalblue')

ax.set_xlabel('Years', fontsize=14)

ax.set_ylabel('Total Suicide',fontsize=14)

ax.set_title('The Trend of Suicides Number Changes', fontsize=16)

ax.set_xticks(x)

ax.set_xticklabels(labels, rotation=30)

ax1 = ax.twinx()

ax1.plot(x, y2, 'firebrick')

ax1.set_ylabel('Suicide Percent',fontsize=14)



plt.show()
subtable_1 = data.pivot_table('suicides_no', index=['year'], columns=['age'], aggfunc='sum')
suicide_age = subtable_1.div(subtable_1.sum(axis=1), axis=0)

suicide_age.drop(2016, inplace=True)
suicide_age.head()
sns.set_style("whitegrid")

labels = suicide_age.index

y1 = suicide_age['15-24']

y2 = suicide_age['25-34']

y3 = suicide_age['35-54']

y4 = suicide_age['5-14']

y5 = suicide_age['55-74']

y6 = suicide_age['75+']

x = np.arange(len(labels))

fig, ax = plt.subplots(figsize=(16,8))

ax.plot(x, y4, color='c', marker='*', label='5-14')

ax.plot(x, y1, color='r', marker='*', label='15-24')

ax.plot(x, y2, color='b', marker='*', label='25-34')

ax.plot(x, y3, color='g', marker='*', label='35-54')

ax.plot(x, y5, color='k', marker='*', label='55-74')

ax.plot(x, y6, color='m', marker='*', label='75+')

ax.legend()

ax.set_title("The Percentage of Suicide by Age", fontsize=16)

ax.set_xlabel("Years", fontsize=14)

ax.set_ylabel("Percent", fontsize=14)

ax.set_xticks(x)

ax.set_xticklabels(labels, rotation=30)

plt.show()
country_names = list(pd.DataFrame(data.groupby('country')['suicides_no'].sum().sort_values()[-4:]).index)
suicide_country = data[data['country'].isin(country_names)]

suicide_country = suicide_country.pivot_table('suicides_no', index='year', columns='country', aggfunc='sum')
suicide_country.dropna(axis=0, how='any', inplace=True)
country_population = data[data['country'].isin(country_names)].pivot_table('population', index='year', columns='country', aggfunc='sum').dropna(axis=0, how='any')
suicide_country = suicide_country.div(country_population) * 100000
labels = suicide_country.index

F = suicide_country['France']

J = suicide_country['Japan']

R = suicide_country['Russian Federation']

U = suicide_country['United States']



x = np.arange(len(labels))  # the label locations

width = 0.2 # the width of the bars



fig, ax = plt.subplots(figsize=(16,8))

sns.set(style="white")

rects1 = ax.bar(x, F, width, label='France', color = 'tomato', alpha=.8)

rects2 = ax.bar(x + width, J, width, label='Japan', color = 'chocolate', alpha=.8)

rects3 = ax.bar(x + 2*width, R, width, label='Russian Federation', color = 'gold', alpha=.8)

rects4 = ax.bar(x + 3*width, U, width, label='United States', color = 'olive', alpha=.8)

ax1 = ax.twinx()

rects5 = ax1.plot(x, F, 'tomato')

rects6 = ax1.plot(x, J, 'chocolate')

rects7 = ax1.plot(x, R, 'gold')

rects8 = ax1.plot(x, U, 'olive')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_xlabel('Years', fontsize=14)

ax.set_ylabel('Suicide/10k Percent', fontsize=14)

ax.set_title('Suicide Percent by Country', fontsize=16)

ax.set_xticks(x)

ax.set_xticklabels(labels, rotation=30)

ax.legend()

ax1.legend(loc=1)



plt.show()
suicide_country_age = data[data['country'].isin(country_names)].pivot_table('suicides_no', index='age', columns='country', aggfunc='sum')

suicide_country_age
labels = suicide_country_age.index

explode = (0, 0, 0.1, 0, 0, 0)

colors = ('gold', 'orange', 'olive', 'yellowgreen', 'palegreen', 'beige')

fig1, ax= plt.subplots(2, 2, figsize = (12,12))

# France

ax[0,0].pie(suicide_country_age['France'], explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

ax[0,0].set_title('France', fontsize=20)

ax[0,0].axis('equal')

# Japan

ax[0,1].pie(suicide_country_age['Japan'], explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

ax[0,1].set_title('Japan', fontsize=20)

ax[0,1].axis('equal')

# Russian Federation

ax[1,0].pie(suicide_country_age['Russian Federation'], explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

ax[1,0].set_title('Russian Federation', fontsize=20)

ax[1,0].axis('equal')

# United States

ax[1,1].pie(suicide_country_age['United States'], explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)

ax[1,1].set_title('United States', fontsize=20)

ax[1,1].axis('equal')



plt.show()
suicide_generation = data.pivot_table('suicides_no', index='generation', columns='sex', aggfunc='sum')

suicide_generation_population = data.pivot_table('population', index='generation', columns='sex', aggfunc='sum')
suicide_generation = suicide_generation.div(suicide_generation_population) * 100000

suicide_generation
labels = suicide_generation.index

M = suicide_generation['M']

F = suicide_generation['F']



x = np.arange(len(labels))  # the label locations

width = 0.35  # the width of the bars



fig, ax = plt.subplots(figsize=(16,8))

sns.set(style="white")

rects1 = ax.bar(x - width/2, M, width, label='Male', color = 'skyblue', alpha=.8)

rects2 = ax.bar(x + width/2, F, width, label='Female', color = 'darkorange', alpha=.8)

ax.set_xlabel('Generation', fontsize=14)

ax.set_ylabel('Suicide Percent', fontsize=14)

ax.set_title('Suicide Percent by Generation', fontsize=16)

ax.set_xticks(x)

ax.set_xticklabels(labels)

ax.legend()



plt.show();
data.head()
# select numerical data

data_1 = pd.DataFrame(data.groupby('country_year')['suicides_no'].sum())

data_2 = pd.DataFrame(data.groupby('country_year')['population'].sum())

data_3 = pd.DataFrame(data.groupby('country_year')['gdp_per_capita'].mean())

data_4 = pd.DataFrame(data.groupby('country_year')['hdi_for_year'].mean())
# conbine data_1, _2, _3

suicide = pd.concat([data_1, data_2, data_3, data_4], axis=1, join='inner')

suicide['suicide_10k_pop'] = suicide['suicides_no'] / suicide['population'] * 100000

suicide.head()
sns.set(style="darkgrid")

sns.jointplot(y = "suicide_10k_pop", x = "gdp_per_capita", data=suicide, kind="reg", color="slateblue", space=0.5)

plt.title("The Scatter of Suicide/10k Pop and GDP/capita", fontsize=18)

plt.xlabel("GDP/capita", fontsize=15)

plt.ylabel("Suicide/10k Pop", fontsize=15)

plt.show();
sns.set(style="darkgrid")

sns.jointplot(y = "suicide_10k_pop", x = "hdi_for_year", data=suicide[suicide['hdi_for_year'] != 0], kind="reg", color="slateblue", space=0.5)

plt.title("The Scatter of Suicide/10k Pop and HDI/year", fontsize=18)

plt.xlabel("HDI/year", fontsize=15)

plt.ylabel("Suicide/10k Pop", fontsize=15)

plt.show();
import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import numpy as np
df = pd.read_csv('../input/acs2015_county_data.csv', index_col=0)

df.head()
df_ark = df.groupby('State').get_group('Arkansas')

df_ark.head()
print(df_ark.index.unique)
print(df_ark['County'].count())
total = df_ark.isnull().sum().sort_values(ascending=False)

percent = (df_ark.isnull().sum()/df_ark.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total','Percent'])

missing_data
df_ark.describe()
df_ark.columns
pop = df_ark['TotalPop'].sum()

print('Total Population: ', pop)
count_pop = pd.concat([df_ark['County'], df_ark['TotalPop'], df_ark['Men'], 

                       df_ark['Women']], axis=1, keys=['County', 'Population',

                                                       'Men', 'Women'])

count_pop = count_pop.sort_values(by='Population', ascending=True)

count_pop.head()
data = count_pop

fig, ax = plt.subplots(figsize=(16,8))

fig = sns.barplot(x='County', y='Population', data=data)

fig.axis(ymin=0, ymax=400000)

plt.xticks(rotation=90)
sns.distplot(df_ark['TotalPop'])
print("Skewness: %f" % df_ark['TotalPop'].skew())

print("Kurtosis: %f" % df_ark['TotalPop'].kurt())
labels = 'Men', 'Women'

sizes = [df_ark['Men'].sum(), df_ark['Women'].sum()]

explode = (0, 0.1)

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.show
melt = pd.melt(count_pop, id_vars='County', value_vars=('Men', 'Women'), 

               var_name='Sex', value_name='Population')

melt.head()
data = melt

gender = 'Men', 'Women'

fig = sns.factorplot(x='County', y='Population' , hue='Sex', data=data, kind='bar', aspect=.9, size=14)

plt.xticks(rotation=90)
race = pd.concat([df_ark['County'], df_ark['Hispanic'], df_ark['White'],

                  df_ark['Black'], df_ark['Native'], df_ark['Asian'], 

                  df_ark['Pacific']],

                 axis=1, keys=['County', 'Hispanic', 'White', 'Black',

                              'Native', 'Asain'])



race.tail(10)
melt2 = pd.melt(race, id_vars='County', value_vars=('Hispanic', 'White', 'Black', 'Native', 'Asain'), var_name='Race', value_name='Percent')

melt2.head()
data = melt2

race = 'Hispanic', 'White', 'Black', 'Native', 'Asain'

fig = sns.factorplot(x='County', y='Percent' , hue='Race', data=data, kind='bar', aspect=1.5, size=11)

plt.xticks(rotation=90)
sns.distplot(df_ark['Income'])
print("Skewness: %f" % df_ark['Income'].skew())

print("Kurtosis: %f" % df_ark['Income'].kurt())
num_citizens = df_ark['Citizen'].sum()



non_citizens = pop - num_citizens



#total population is saved under "pop"



labels = 'Citizens', 'Non-Citizens'

sizes = [num_citizens, non_citizens]

explode = (.1, 0)

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.show()
Wash = pd.DataFrame(df_ark.loc[5143])

Wash
labels = 'Hispanic', 'White', 'Black', 'Native', 'Asian', 'Pacific'

sizes = [Wash.get_value('Hispanic', 5143), Wash.get_value('White', 5143),

         Wash.get_value('Black', 5143), Wash.get_value('Native', 5143),

         Wash.get_value('Asian', 5143), Wash.get_value('Pacific', 5143)]

f, ax = plt.subplots(figsize=(10,6))

explode = (0, 0.1, 0, 0, 0, 0)

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.show()
labels = 'Professional', 'Service', 'Office', 'Construction', 'Production'

sizes = [Wash.get_value('Professional', 5143), Wash.get_value('Service', 5143),

         Wash.get_value('Office', 5143), Wash.get_value('Construction', 5143),

         Wash.get_value('Production', 5143)]

explode = (0.1, 0, 0, 0, 0)

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.show()
#Washington County labor force is estimated at 147548

#https://www.census.gov/quickfacts/fact/table/washingtoncountyarkansas,AR,US/PST045216



num_unemployed = 0.062 * 147548

num_unemployed = round(num_unemployed)



labels = 'Employed', 'Unemployed'

sizes = [Wash.get_value('Employed', 5143), num_unemployed]

explode = (0.1, 0)

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.show()
labels = 'Public Work', 'Private Work', 'Self-Employed', 'Family Work'

sizes = [Wash.get_value('PublicWork', 5143), Wash.get_value('PrivateWork', 5143),

         Wash.get_value('SelfEmployed', 5143), Wash.get_value('FamilyWork', 5143)]

explode = (0, 0.1, 0, 0)

plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=True)

plt.axis('equal')

plt.show()
#correlation matrix



corrmat = df_ark.corr()

f, ax = plt.subplots(figsize=(12,9))

sns.heatmap(corrmat, vmax=.8, square=True)
k = 15

cols = corrmat.nlargest(k, 'Income')['Income'].index

f, ax = plt.subplots(figsize=(10,6))

cm = np.corrcoef(df_ark[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',

                annot_kws={'size':8}, yticklabels=cols.values,

                xticklabels=cols.values)

plt.show()
#Income linear relationships



sns.set()

cols = ['Income', 'IncomePerCap', 'Employed', 'Men', 'TotalPop', 'Citizen',

       'Women', 'PrivateWork', 'Professional', 'Office', 'Asian', 'Drive']

sns.pairplot(df_ark[cols], size=2.5)

plt.show()
#Poverty



k = 7

cols = corrmat.nlargest(k, 'Poverty')['Poverty'].index

f, ax = plt.subplots(figsize=(10,6))

cm = np.corrcoef(df_ark[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',

                annot_kws={'size':10}, yticklabels=cols.values,

                xticklabels=cols.values)

plt.show()
#poverty liner relationships



sns.set()

cols = ['Poverty', 'ChildPoverty', 'Unemployment', 'Black', 'PublicWork', 'OtherTransp', 'Service']

sns.pairplot(df_ark[cols], size=2.5)

plt.show()
# Employment



k = 15

cols = corrmat.nlargest(k, 'Employed')['Employed'].index

f, ax = plt.subplots(figsize=(10,6))

cm = np.corrcoef(df_ark[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',

                annot_kws={'size':8}, yticklabels=cols.values,

                xticklabels=cols.values)

plt.show()
#employment linear relationships



sns.set()

cols = ['Employed', 'TotalPop', 'Men', 'Women', 'Citizen', 'IncomePerCap',

       'Professional', 'Income', 'Asian', 'Office', 'PrivateWork', 'Pacific']

sns.pairplot(df_ark[cols], size=2.5)

plt.show()
#unemployment



k = 7

cols = corrmat.nlargest(k, 'Unemployment')['Unemployment'].index

cm = np.corrcoef(df_ark[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',

                annot_kws={'size':10}, yticklabels=cols.values,

                xticklabels=cols.values)

plt.show()
#unemployment linear relationships



sns.set()

cols = ['Unemployment', 'Black', 'Poverty', 'PublicWork', 'ChildPoverty', 'OtherTransp',

       'Service']

sns.pairplot(df_ark[cols], size=2.5)

plt.show()
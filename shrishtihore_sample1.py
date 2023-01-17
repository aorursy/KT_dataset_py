import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import matplotlib.pyplot as plt

import seaborn as sns



print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataset = pd.read_csv('../input/master.csv')
dataset.head()
unique_country = dataset['country'].unique()

print(unique_country)
alpha = 0.7

plt.figure(figsize=(10,25))

sns.countplot(y ='country', data=dataset, alpha=alpha)

plt.title('Data by country')

plt.show()
plt.figure(figsize=(16,7))

sex = sns.countplot(x='sex', data=dataset)
plt.figure(figsize=(20,10))

cor = sns.heatmap(dataset.corr(), annot=True)
plt.figure(figsize=(20,10))

bar_age = sns.barplot(x='sex',y='suicides_no',hue='age',data=dataset)
plt.figure(figsize=(20,10))

bar_gen = sns.barplot(x='sex',y='suicides_no',hue='generation',data=dataset)
cat_accord_year = sns.catplot('sex','suicides_no',hue='age',col='year',data=dataset,kind='bar',col_wrap=3)
age_5 = dataset.loc[dataset.loc[:,'age']=='5-14 years',:]

age_15 = dataset.loc[dataset.loc[:,'age']=='15-24 years',:]

age_25 = dataset.loc[dataset.loc[:,'age']=='25-34 years',:]

age_35 = dataset.loc[dataset.loc[:,'age']=='35-44 years',:]

age_45 = dataset.loc[dataset.loc[:,'age']=='45-54 years',:]

age_55 = dataset.loc[dataset.loc[:,'age']=='55-64 years',:]

age_65 = dataset.loc[dataset.loc[:,'age']=='65-74 years',:]

age_75 = dataset.loc[dataset.loc[:,'age']=='75+ years',:]
plt.figure(figsize=(20,7))

age_5_lp = sns.lineplot(x = 'year', y ='suicides_no', data=age_5)

age_15_lp = sns.lineplot(x = 'year', y ='suicides_no', data=age_15)

age_25_lp = sns.lineplot(x = 'year', y ='suicides_no', data=age_25)

age_35_lp = sns.lineplot(x = 'year', y ='suicides_no', data=age_35)

age_45_lp = sns.lineplot(x = 'year', y ='suicides_no', data=age_45)

age_55_lp = sns.lineplot(x = 'year', y ='suicides_no', data=age_55)

age_65_lp = sns.lineplot(x = 'year', y ='suicides_no', data=age_65)

age_75_lp = sns.lineplot(x = 'year', y ='suicides_no', data=age_75)



leg=plt.legend(['5-14 years' , '15-24 years', '25-34 years', '35-54 years', '55-74 years', '75+ years'])
male_population = dataset.loc[dataset.loc[:, 'sex']=='male',:]

female_population = dataset.loc[dataset.loc[:, 'sex']=='female',:]

plt.figure(figsize=(20,10))
lp_male=sns.lineplot(x='year',y='suicides_no',data=male_population)

lp_female=sns.lineplot(x='year',y='suicides_no',data=female_population)

leg1 = plt.legend(['Males','Females'])

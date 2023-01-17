# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



# Any results you write to the current directory are saved as output.
cities = pd.read_csv ("../input/cities_r2.csv")
cities.head()
cities.info()

#checking for null values
cities.isnull().values.any()



#checking for null values
cities.describe()
print (cities.describe(include=['O']))

# from the below output, we can learn that there is two Aurangabad's. One is in Maharashtra 

# and one is in Bihar

# most of the cities are selected from Uttar Pradesh
print(cities['state_name'].unique())



print(len(cities['state_name'].unique()))



print(cities['dist_code'].unique())



print(len(cities['dist_code'].unique()))





#unique states and their cities
population_state = cities[['state_name','population_total']].groupby('state_name').sum().sort_values('population_total',ascending=False)

print(population_state)

# A bar chart to show from which states, how many cities are taken for examination.

fig = plt.figure(figsize=(20,20))

states = cities.groupby('state_name')['name_of_city'].count().sort_values(ascending=True)

states.plot(kind="barh", fontsize = 20)

plt.grid(b=True, which='both', color='Black',linestyle='-')

plt.xlabel('No of cities taken for analysis', fontsize = 20)

plt.show ()

# we can see states like UP and WB are given high priority by taking more than 60 cities.
# States according to literacy rate

lit_by_states  = cities.groupby('state_name').agg({'literates_total': np.sum})

pop_by_states  = cities.groupby('state_name').agg({'population_total': np.sum})

literate_rate = lit_by_states.literates_total * 100 / pop_by_states.population_total

literate_rate = literate_rate.sort_values(ascending=False)



plt.subplots(figsize=(7, 6))

ax = sns.barplot(x=literate_rate, y=literate_rate.index)

ax.set_title('States according to literacy rate', size=20, alpha=0.5, color='blue')

ax.set_xlabel('Literacy Rate(as % of population)', size=15, alpha=0.5, color='red')

ax.set_ylabel('States', size=25, alpha=0.5, color='red')
def proportion(group, col1, col2):

    col = group[col1].sum()

    tot_pop = group[col2].sum()

    return (col * 100 / tot_pop)



prop_female_lit = cities.groupby('state_name').apply(proportion, 'literates_female', 'population_female')

prop_male_lit = cities.groupby('state_name').apply(proportion, 'literates_male', 'population_male')



summary = pd.DataFrame({'literates_female': prop_female_lit, 'literates_male':prop_male_lit})

fem_summary = summary.sort_values([('literates_female')], ascending=False)



plt.subplots(figsize=(7, 6))

ax = sns.barplot(x='literates_female', y=fem_summary.index, data=fem_summary)

ax.set_title('States by female literacy rate', size=20, alpha=0.5, color='green')

ax.set_xlabel('Female literacy Rate', size=20, alpha=0.5, color='red')

ax.set_ylabel('States', size=20, alpha=0.5, color='red')

prop_female_lit = cities.groupby('state_name').apply(proportion, 'literates_female', 'population_female')

prop_female_lit = prop_female_lit * 10

sex_ratio_by_state = cities.groupby('state_name').agg({'sex_ratio':np.mean})



prop_female_lit = pd.DataFrame({'female_lit':prop_female_lit})

df = pd.concat([prop_female_lit, sex_ratio_by_state], axis='columns')



plt.subplots(figsize=(8, 6))

ax = sns.regplot(x='sex_ratio', y='female_lit', data=df, order=3, ci=50, 

                 scatter_kws={'alpha':0.5, 'color':'red'})

ax.set_title('Female literacy vs sex ratio', size=20, alpha=0.5, color='green')

ax.set_xlabel('Sex Ratio', size=20, alpha=0.5, color='red')

ax.set_ylabel('Female literacy(per thousand)', size=20, alpha=0.5, color='red')
# Top 5 most literate state by population

lit_by_states  = cities.groupby('state_name').agg({'literates_total': np.sum}).sort_values(

    [('literates_total')], ascending=False)[:5]

lit_by_states = lit_by_states / 1000000

plt.subplots(figsize=(8, 5))

ax = sns.barplot(data=lit_by_states, x=lit_by_states.index, y='literates_total')



ax.set_title('Top 5 states with most literates', size=25, alpha=0.5, color='green')

ax.set_xlabel('States', size=20, alpha=0.5, color='red')

ax.set_ylabel('Number of literates (millions)', size=20, alpha=0.5, color='red')
graduates_rate = cities.groupby(['state_name']).apply(proportion, 

                                                      'total_graduates', 

                                                      'population_total').sort_values(ascending=False)



plt.subplots(figsize=(7, 6))

ax = sns.barplot(x=graduates_rate, y=graduates_rate.index, palette='cubehelix')

ax.set_title('States by Graduate Rate', size=30, color='red', alpha=0.5)

ax.set_xlabel('Graduate Rate', size=30, color='blue', alpha=0.5)

ax.set_ylabel('States', size=30, color='blue', alpha=0.5)
state_graduate  = cities[["state_name",

                                  "total_graduates",

                                  "male_graduates",

                                  "female_graduates"]].groupby("state_name").agg({"total_graduates":np.sum,

                                                                                "male_graduates":np.sum,

                                                                                "female_graduates":np.sum})

state_graduate.plot(kind="bar",

                      grid=False,

                      figsize=(16,10),

                      #color="r",

                      alpha = 0.5,

                      width=0.6,

                      stacked = False,

                     edgecolor="g",)
plt.figure(figsize=(15,10))

sns.heatmap(cities.corr(),

            annot=True,

            linewidths=.5,

            center=0,

            cbar=False,

            cmap="YlGnBu")

plt.show()

from sklearn.preprocessing import StandardScaler

x=cities.drop(['name_of_city', 'state_name','dist_code','location','state_code'],axis=1)

scaler=StandardScaler().fit(x)

y=pd.DataFrame(scaler.transform(x),columns=x.columns)

y.boxplot(vert=False,figsize=(15,10))
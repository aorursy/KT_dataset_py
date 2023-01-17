# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



plt.rcParams['figure.figsize'] = (15, 5)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
suicide = pd.read_csv('/kaggle/input/suicide-rates-overview-1985-to-2016/master.csv', encoding='latin1')

suicide
suicide = suicide.rename(columns = {' gdp_for_year ($) ' : 'gdp_for_year ($)'})

suicide.columns
suicide['gdp_for_year($)'] = suicide['gdp_for_year ($)'].str.replace(',', '')

suicide['gdp_for_year($)'] = suicide['gdp_for_year($)'].astype('float')

suicide['gdp_for_year($)']
coutries_per_year=[]

for i in suicide.year.unique():

    coutries_per_year.extend(suicide[suicide['year']==i].country.unique())
country_count = {}

for i in coutries_per_year:

  country_count[i] = country_count.get(i, 0) + 1



# print(country_count)
country_count= {k: v for k, v in sorted(country_count.items(), key=lambda item: item[1])}
pays = suicide["country"].isin(list(country_count)[-23:])

suicide[pays]["country"].unique()
# suicide evolution for 1985

data_1985= suicide[(suicide['year']==1985)]

data_country= suicide[pays]

country_1985_population_suicide_rate=[]  

country_unique=data_country["country"].unique()                    

for country in country_unique:

    country_1985_population_suicide_rate.append(sum(data_1985[data_1985['country']==country].population))   

                   

# suicide evolution for 2000

data_2000=suicide[(suicide['year']==2000)]

data_country= suicide[pays]

country_2000_population_suicide_rate=[]  

country_unique=data_country["country"].unique()                    

for country in country_unique:

    country_2000_population_suicide_rate.append(sum(data_2000[data_2000['country']==country].population))                  



# suicide evolution for 2016

data_2016=suicide[(suicide['year']==2016)]

data_country= suicide[pays]

country_2016_population_suicide_rate=[]  

country_unique=data_country["country"].unique()                    

for country in country_unique:

    country_2016_population_suicide_rate.append(sum(data_2016[data_2016['country']==country].population))                  

# Plot suicide evolution for 1985



f,ax=plt.subplots(1,3,figsize=(15,20),sharex=True, sharey=True,squeeze=False)

ax = ax.flatten()



sns.barplot(y=country_unique,x=country_1985_population_suicide_rate,ax=ax[0])

ax[0].set_title('1985 Year Sum  Suicide occurence')

ax[0].set_ylabel('Countries', fontsize=12)

ax[0].set_xlabel("Population Count")



# Plot suicide evolution for 2000

sns.barplot(y=country_unique,x=country_2000_population_suicide_rate,ax=ax[1])

ax[1].set_title('2000 Year Sum  Suicide occurence')

ax[1].set_xlabel("Population Count", fontsize=12)



# Plot suicide evolution for 2016

sns.barplot(y=country_unique,x=country_2016_population_suicide_rate,ax=ax[2])

ax[2].set_title('2016 Year Sum  Suicide occurence')

ax[2].set_xlabel("Population Count", fontsize=12)





plt.show()
female_=[175437,208823,506233,16997,430036,221984]

male_=[633105,915089,1945908,35267,1228407,431134]

plot_id = 0

for i,age in enumerate(['15-24 years','25-34 years','35-54 years','5-14 years','55-74 years','75+ years']):

    plot_id += 1

    plt.subplot(3,2,plot_id)

    plt.title(age)

    fig, ax = plt.gcf(), plt.gca()

    sns.barplot(x=['female','male'],y=[female_[i],male_[i]],color='blue')

    plt.tight_layout()

    fig.set_size_inches(10, 15)

plt.show()  


plt.figure(figsize=(18,8))

suicide['generation'].value_counts().plot.pie(explode=[0.1,0.1,0.1,0.1,0.1,0.1],autopct='%1.1f%%',shadow=True)

plt.title('Generations Count')

plt.ylabel('Count')

 

plt.show()
plt.figure(figsize=(10,7))



sns.set(style="white")

# Plot miles per gallon against horsepower with other semantics

sns.relplot(x="generation",y="suicides_no",hue="sex",

            sizes=(50, 400), alpha=.95, palette="muted",height=6, data=suicide)

plt.show()
# Nous sélectionnons les Data utile pour notre comparaison pour l'Albanie tout d'abord



df_alb = suicide.loc[suicide['country'] == 'Albania']

df_alb
# Pour l'Albanie nous avons les données de 1987 a 2010

# Pour une meilleure comparaison entre les deux pays, nous gardons nos données entre ces valeurs

# Or la croatie démarre à 1995 donc nous changeons aussi pour l'Albanie



df_crot = suicide.loc[suicide['country'] == 'Croatia']

df_crot = df_crot.loc[df_crot["year"] <= 2010]

df_crot
# Stat globale entre les femmes et les hommes en Albanie de 1987 a 2010

df_alb_suicide_per_s = df_alb.groupby(["sex"])['suicides_no'].sum()

# df_alb_suicide_per_s
# Nous avons donc la représentation : plus de suicide chez le shommes que les femmes

df_alb_suicide_per_s.plot(kind='pie',  autopct='%1.1f%%', shadow=True)

plt.title("Taux de suicide entre Hommes et Femmes en Albanie de 1987 a 2010")

plt.legend()
# Stat globale par sex



df_crot_suicide_per_s = df_crot.groupby(["sex"])['suicides_no'].sum()

# df_crot_suicide_per_s
# Représentation de sstats globale par sex en Croatie



df_crot_suicide_per_s.plot(kind='pie', autopct='%1.1f%%', shadow=True)

plt.title("Taux de suicide entre Hommes et Femmes en Croatie de 1987 a 2010")

plt.legend()
# Stat globale par année et sex en regroupant le nb de suicide de chaque tranche d'age



df_alb_suicide_per_y_s = df_alb.groupby(["year","sex"])['suicides_no'].sum()

# df_alb_suicide_per_y_s.head(10)
# link for colors

# https://stackoverflow.com/questions/19852215/how-to-add-a-legend-to-matplotlib-pie-chart



from matplotlib import pyplot as plt

import pandas, numpy as np

from itertools import cycle, islice

plt.rcParams['figure.figsize'] = (25, 5)



x = [{i:np.random.randint(1,2)} for i in range(4)]

df = pandas.DataFrame(x)



my_colors = list(islice(cycle(['b', 'r']), None, len(df)))



df_alb_suicide_per_y_s.plot(kind='bar', color = my_colors)



df_alb_suicide_per_year = df_alb.groupby(["year"])['suicides_no'].sum()

df_alb_suicide_per_year.plot(kind = 'bar')

df_crot_suicide_per_y_s = df_crot.groupby(["year","sex"])['suicides_no'].sum()



# df_crot_suicide_per_y_s.head(10)
# link for colors

# https://stackoverflow.com/questions/19852215/how-to-add-a-legend-to-matplotlib-pie-chart



from matplotlib import pyplot as plt

from itertools import cycle, islice

import pandas, numpy as np

plt.rcParams['figure.figsize'] = (25, 5)



x = [{i:np.random.randint(1,2)} for i in range(4)]

df = pandas.DataFrame(x)



my_colors = list(islice(cycle(['b','r']), None, len(df)))



df_crot_suicide_per_y_s.plot(kind='bar', color = my_colors)
df_crot_suicide_per_year = df_crot.groupby(["year"])['suicides_no'].sum()

df_crot_suicide_per_year.plot(kind='bar')
# On regroupe par année le nb de suicide et la population



df_crot_suicide_per_year = df_crot.groupby(["year"])['suicides_no','population'].sum()

df_crot_suicide_per_year



# Statistique de la prop de suicide pour 100k habitants par an en croatie



nb_mort = df_crot_suicide_per_year['suicides_no']/df_crot_suicide_per_year['population']*100000

nb_mort = round(nb_mort,2)



df_crot_suicide_per_year['Nb de mort pour 100k hab'] = nb_mort

df_crot_suicide_per_year
df_crot_suicide_per_year['Nb de mort pour 100k hab'].plot()
df_alb_suicide_per_year = df_alb.groupby(["year"])['suicides_no','population'].sum()

df_alb_suicide_per_year



# Statistique de la prop de suicide pour 100k habitants par an en albanie



nb_mort = df_alb_suicide_per_year['suicides_no']/df_alb_suicide_per_year['population']*100000

nb_mort = round(nb_mort,2)



df_alb_suicide_per_year['Nb de mort pour 100k hab'] = nb_mort

df_alb_suicide_per_year

# Statistique de la prop de suicide pour 100k habitants par an en albanie et en croatie



df_alb_suicide_per_year['Nb de mort pour 100k hab'].plot()

dataset = suicide

coree = dataset[dataset['country'] == 'Republic of Korea']

# évolution du gdp per capita et du ratio global de suicide dans la population

indice_year_coree = coree_by_year.index

gdp_per_capita_coree = coree_by_year['gdp_per_capita ($)']

ratio_suicides_per_year_coree = coree_by_year['suicide_ratio']



# évolution du gdp per capita et du ratio global de suicide dans la population

indice_year_sa = sa_by_year.index

gdp_per_capita_sa = sa_by_year['gdp_per_capita ($)']

ratio_suicides_per_year_sa = sa_by_year['suicide_ratio']



# group by year

coree_by_year = coree.groupby('year').mean()



# get global suicide ratio

def divide_col(col1, col2):

    return col1 / col2 * 100000



coree_by_year.loc[:,'suicide_ratio'] = coree_by_year.apply(lambda x: divide_col(x['suicides_no'], x['population']), axis=1)

# coree_by_year





south_africa = dataset[dataset['country'] == 'South Africa']



# group by year

sa_by_year = south_africa.groupby('year').mean()



# get global suicide ratio

def divide_col(col1, col2):

    return col1 / col2 * 100000



sa_by_year.loc[:,'suicide_ratio'] = sa_by_year.apply(lambda x: divide_col(x['suicides_no'], x['population']), axis=1)

# sa_by_year





# 2 plots on 1 figure



fig = plt.figure()

fig.suptitle('Evolution du PIB/hab et du taux de suicide', fontsize=20)



ax1 = plt.subplot(1, 2, 1)

# first set of axis

color = 'tab:red'

ax1.bar(indice_year_coree, ratio_suicides_per_year_coree, color=color)

ax1.tick_params(axis='y', labelcolor=color)

ax1.set_ylim([0, 40])

ax1.set(title = "Corée du Sud",

       ylabel = "Suicides / 100 000",

       xlabel = 'year')



# second set of axis

color = 'tab:green'

ax12 = ax1.twinx()

ax12.plot(indice_year_coree, gdp_per_capita_coree, color=color)

ax12.tick_params(axis='y', labelcolor=color)





# ------



ax2 = plt.subplot(1, 2, 2)

# first set of axis

color = 'tab:red'

ax2.bar(indice_year_sa, ratio_suicides_per_year_sa, color=color)

ax2.tick_params(axis='y', labelcolor=color)

ax2.set_ylim([0, 40])

ax2.set(title = "Afrique du Sud",  xlabel = 'year')



# second set of axis

color = 'tab:green'

ax22 = ax2.twinx()

ax22.set_ylim([0, 30000])

ax22.set_ylabel('PIB par tête', color=color)

ax22.plot(indice_year_sa, gdp_per_capita_sa, color=color)

ax22.tick_params(axis='y', labelcolor=color)

# ---



plt.show()

# répartition suicides par genre

coree_suicides_by_gender = coree[['year', 'sex','suicides_no', 'population']].groupby(['year', 'sex']).sum()

coree_suicides_by_gender.loc[:,'suicide_ratio'] = coree_suicides_by_gender.apply(lambda x: divide_col(x['suicides_no'], x['population']), axis=1)

coree_suicides_by_gender



# répartition suicides par genre

sa_suicides_by_gender = south_africa[['year', 'sex','suicides_no', 'population']].groupby(['year', 'sex']).sum()

sa_suicides_by_gender.loc[:,'suicide_ratio'] = sa_suicides_by_gender.apply(lambda x: divide_col(x['suicides_no'], x['population']), axis=1)

#sa_suicides_by_gender





fig = plt.figure()

fig.suptitle('Taux de suicide par sexe', fontsize=20)



ax1 = plt.subplot(1, 2, 1)

ax1.set(title = "Corée du Sud")

coree_suicides_by_gender['suicide_ratio'].loc[[(2015, 'female'), (2015, 'male')]].plot(kind='pie', subplots= True , autopct='%.1f%%')



ax2 = plt.subplot(1, 2, 2)

ax2.set(title = "Afrique du Sud")

sa_suicides_by_gender['suicide_ratio'].loc[[(2015, 'female'), (2015, 'male')]].plot(kind='pie', subplots= True , autopct='%.1f%%')

plt.show()



# par génération, à quel âge les gens se suicident ? 

new_index_gen = ['G.I. Generation', 'Silent', 'Boomers', 'Generation X', 'Millenials', 'Generation Z']

new_index_age = ['75+ years', '55-74 years', '55-74 years', '35-54 years', '25-34 years', '15-24 years', '5-14 years']

sa_juvenil_suicide = south_africa[['year','age', 'generation', 'suicides_no']]

sa_juvenil_suicide = sa_juvenil_suicide.groupby(['year', 'generation', 'age']).sum()

sa_generational_suicide_2015 = sa_juvenil_suicide.iloc[sa_juvenil_suicide.index.get_level_values('year') == 2015]





coree_juvenil_suicide = coree[['year','age', 'generation', 'suicides_no']]

coree_juvenil_suicide = coree_juvenil_suicide.groupby(['year', 'generation', 'age']).sum()

coree_generational_suicide_2015 = coree_juvenil_suicide.iloc[coree_juvenil_suicide.index.get_level_values('year') == 2015]





# suicide juvénil ?
coree_generational_suicide_2015.plot(kind='bar')

sa_generational_suicide_2015.plot(kind='bar')

plt.show()
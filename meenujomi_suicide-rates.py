#Importing required packages

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
#Loading the data set

df = pd.read_csv("../input/master.csv")

df
# The .info() code gives almost the entire information that needs to be inspected, so let's start from there

df.info()
#To get the idea of how the table looks like we can use .head() or .tail() command

df.head()
df.tail()
# The .shape code gives the no. of rows and columns

df.shape
#To get an idea of the numeric values, use .describe()

df.describe()
df = df.drop(columns = 'HDI for year')

df.info()
df.isnull().sum()
df
df['age'] = df['age'].str.rstrip(' years')

df = df.drop(['country-year'], axis= 1)

df = df.rename(columns={' gdp_for_year ($) ':'gdp_for_year','gdp_per_capita ($)':'gdp_per_capital'})

df['gdp_for_year'] = df['gdp_for_year'].apply(lambda x: float(x.split()[0].replace(',', '')))

df
# Compaing the different countries

plt.figure(figsize=(20,10))

country = sns.countplot(df['country'],order = df['country'].value_counts().index)

country.tick_params(axis='x', rotation=90)

plt.show()
# Compaing the different years

plt.figure(figsize=(20,10))

year = sns.countplot(df['year'],order = df['year'].value_counts().index)

year.tick_params(axis='x', rotation=90)

plt.show()
# Compaing the different generations

generation = sns.countplot(df['generation'],order= ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

generation.tick_params(axis='x', rotation=90)

plt.show()
suicides_no_country = pd.pivot_table(df, values = 'suicides_no', aggfunc= 'sum', index ='country')

suicides_no_country
plt.figure(figsize=(20,10))

plt.plot(suicides_no_country)

plt.xticks(rotation=90)

plt.show()
suicides_no_year = pd.pivot_table(df, values = 'suicides_no', aggfunc= 'sum', index ='year')

suicides_no_year
plt.plot(suicides_no_year)

plt.xticks(rotation=90)

plt.show()
suicides_no_sex = pd.pivot_table(df, values = 'suicides_no', aggfunc= 'sum', index ='sex')

suicides_no_sex
plt.plot(suicides_no_sex)

plt.xticks(rotation=90)

plt.show()
suicides_no_age = pd.pivot_table(df, values = 'suicides_no', aggfunc= 'sum', index ='age')

suicides_no_age
plt.plot(['5-14','15-24','25-34','35-54','55-74','75+'],suicides_no_age)

plt.xticks(rotation=90)

plt.show()
suicides_no_generation = pd.pivot_table(df, values = 'suicides_no', aggfunc= 'sum', index ='generation')

suicides_no_generation
plt.plot(['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'],suicides_no_generation)

plt.xticks(rotation=90)

plt.show()
population_country = pd.pivot_table(df, values = 'population', aggfunc= 'sum', index ='country')

population_country
plt.figure(figsize=(20,10))

plt.plot(population_country)

plt.xticks(rotation=90)

plt.show()
population_year = pd.pivot_table(df, values = 'population', aggfunc= 'sum', index ='year')

population_year
plt.plot(population_year)

plt.xticks(rotation=90)

plt.show()
population_sex = pd.pivot_table(df, values = 'population', aggfunc= 'sum', index ='sex')

population_sex
plt.plot(population_sex)

plt.xticks(rotation=90)

plt.show()
population_age = pd.pivot_table(df, values = 'population', aggfunc= 'sum', index ='age')

population_age
plt.plot(['5-14','15-24','25-34','35-54','55-74','75+'],population_age)

plt.xticks(rotation=90)

plt.show()
population_generation = pd.pivot_table(df, values = 'population', aggfunc= 'sum', index ='generation')

population_generation
plt.plot(['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'],population_generation)

plt.xticks(rotation=90)

plt.show()
suicides_100k_pop_country = pd.pivot_table(df, values = 'suicides/100k pop', aggfunc= 'sum', index ='country')

suicides_100k_pop_country
plt.figure(figsize=(15,10))

plt.plot(suicides_100k_pop_country)

plt.xticks(rotation=90)

plt.show()
suicides_100k_pop_year = pd.pivot_table(df, values = 'suicides/100k pop', aggfunc= 'sum', index ='year')

suicides_100k_pop_year
plt.plot(suicides_100k_pop_year)

plt.xticks(rotation=90)

plt.show()
suicides_100k_pop_sex = pd.pivot_table(df, values = 'suicides/100k pop', aggfunc= 'sum', index ='sex')

suicides_100k_pop_sex
plt.plot(suicides_100k_pop_sex)

plt.xticks(rotation=90)

plt.show()
suicides_100k_pop_age = pd.pivot_table(df, values = 'suicides/100k pop', aggfunc= 'sum', index ='age')

suicides_100k_pop_age
plt.plot(['5-14','15-24','25-34','35-54','55-74','75+'],suicides_100k_pop_age)

plt.xticks(rotation=90)

plt.show()
suicides_100k_pop_generation = pd.pivot_table(df, values = 'suicides/100k pop', aggfunc= 'sum', index ='generation')

suicides_100k_pop_generation
plt.plot(['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'],suicides_100k_pop_generation)

plt.xticks(rotation=90)

plt.show()
df_gdp = pd.DataFrame(data = df.gdp_for_year)

df_gdp['gdp_per_capital'] = df['gdp_per_capital']

df_gdp['country'] = df['country']

df_gdp['year'] = df['year']

df_gdp['sex'] = df['sex']

df_gdp['age'] = df['age']

df_gdp['generation'] = df['generation']

df_gdp = df_gdp.drop_duplicates(keep='first')

df_gdp
country_year_suicides_no = pd.pivot_table(df, values= 'suicides_no',index='country', columns = 'year' ,aggfunc='sum')

country_year_suicides_no.fillna(0, inplace=True)

country_year_suicides_no
plt.figure(figsize=(10,50))

sns.heatmap(country_year_suicides_no,cmap= sns.cubehelix_palette(200))

plt.show()
country_year_population = pd.pivot_table(df, values= 'population',index='country', columns = 'year' ,aggfunc='sum')

country_year_population.fillna(0, inplace=True)

country_year_population 
plt.figure(figsize=(10,50))

sns.heatmap(country_year_population ,cmap= sns.cubehelix_palette(200))

plt.show()
country_year_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='country', columns = 'year' ,aggfunc='sum')

country_year_suicides_100k_pop.fillna(0, inplace=True)

country_year_suicides_100k_pop
plt.figure(figsize=(10,50))

sns.heatmap(country_year_suicides_100k_pop,cmap= sns.cubehelix_palette(200))

plt.show()
country_year_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='country', columns = 'year' ,aggfunc='sum')

country_year_gdp_for_year.fillna(0, inplace=True)

country_year_gdp_for_year 
plt.figure(figsize=(10,50))

sns.heatmap(country_year_gdp_for_year ,cmap= sns.cubehelix_palette(200))

plt.show()
country_year_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='country', columns = 'year' ,aggfunc='sum')

country_year_gdp_per_capital.fillna(0, inplace=True)

country_year_gdp_per_capital
plt.figure(figsize=(10,50))

sns.heatmap(country_year_gdp_per_capital ,cmap= sns.cubehelix_palette(200))

plt.show()
country_sex_suicides_no = pd.pivot_table(df, values= 'suicides_no',index='country', columns = 'sex' ,aggfunc='sum')

country_sex_suicides_no.fillna(0, inplace=True)

country_sex_suicides_no
plt.figure(figsize=(2,50))

sns.heatmap(country_sex_suicides_no,cmap= sns.cubehelix_palette(200))

plt.show()
country_sex_population = pd.pivot_table(df, values= 'population',index='country', columns = 'sex' ,aggfunc='sum')

country_sex_population.fillna(0, inplace=True)

country_sex_population 
plt.figure(figsize=(2,50))

sns.heatmap(country_sex_population ,cmap= sns.cubehelix_palette(200))

plt.show()
country_sex_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='country', columns = 'sex' ,aggfunc='sum')

country_sex_suicides_100k_pop.fillna(0, inplace=True)

country_sex_suicides_100k_pop
plt.figure(figsize=(2,50))

sns.heatmap(country_sex_suicides_100k_pop,cmap= sns.cubehelix_palette(200))

plt.show()
country_sex_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='country', columns = 'sex' ,aggfunc='sum')

country_sex_gdp_for_year.fillna(0, inplace=True)

country_sex_gdp_for_year 
plt.figure(figsize=(2,50))

sns.heatmap(country_sex_gdp_for_year ,cmap= sns.cubehelix_palette(200))

plt.show()
country_sex_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='country', columns = 'sex' ,aggfunc='sum')

country_sex_gdp_per_capital.fillna(0, inplace=True)

country_sex_gdp_per_capital
plt.figure(figsize=(2,50))

sns.heatmap(country_sex_gdp_per_capital ,cmap= sns.cubehelix_palette(200))

plt.show()
country_age_suicides_no = pd.pivot_table(df, values= 'suicides_no',index='country', columns = 'age' ,aggfunc='sum')

country_age_suicides_no.fillna(0, inplace=True)

country_age_suicides_no
plt.figure(figsize=(6,50))

sns.heatmap(country_age_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])

plt.show()
country_age_population = pd.pivot_table(df, values= 'population',index='country', columns = 'age' ,aggfunc='sum')

country_age_population.fillna(0, inplace=True)

country_age_population 
plt.figure(figsize=(6,50))

sns.heatmap(country_age_population ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])

plt.show()
country_age_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='country', columns = 'age' ,aggfunc='sum')

country_age_suicides_100k_pop.fillna(0, inplace=True)

country_age_suicides_100k_pop
plt.figure(figsize=(6,50))

sns.heatmap(country_age_suicides_100k_pop,cmap= sns.cubehelix_palette(200))

plt.show()
country_generation_suicides_no = pd.pivot_table(df, values= 'suicides_no',index='country', columns = 'generation' ,aggfunc='sum')

country_generation_suicides_no.fillna(0, inplace=True)

country_generation_suicides_no
plt.figure(figsize=(6,50))

sns.heatmap(country_generation_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
country_generation_population = pd.pivot_table(df, values= 'population',index='country', columns = 'generation' ,aggfunc='sum')

country_generation_population.fillna(0, inplace=True)

country_generation_population 
plt.figure(figsize=(6,50))

sns.heatmap(country_generation_population ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
country_generation_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='country', columns = 'generation' ,aggfunc='sum')

country_generation_suicides_100k_pop.fillna(0, inplace=True)

country_generation_suicides_100k_pop
plt.figure(figsize=(6,50))

sns.heatmap(country_generation_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
country_generation_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='country', columns = 'generation' ,aggfunc='sum')

country_generation_gdp_for_year.fillna(0, inplace=True)

country_generation_gdp_for_year 
plt.figure(figsize=(6,50))

sns.heatmap(country_generation_gdp_for_year ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
country_generation_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='country', columns = 'generation' ,aggfunc='sum')

country_generation_gdp_per_capital.fillna(0, inplace=True)

country_generation_gdp_per_capital
plt.figure(figsize=(6,50))

sns.heatmap(country_generation_gdp_per_capital ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
year_sex_suicides_no = pd.pivot_table(df, values= 'suicides_no',index='year', columns = 'sex' ,aggfunc='sum')

year_sex_suicides_no.fillna(0, inplace=True)

year_sex_suicides_no
plt.figure(figsize=(4,10))

sns.heatmap(year_sex_suicides_no,cmap= sns.cubehelix_palette(200))

plt.show()
year_sex_population = pd.pivot_table(df, values= 'population',index='year', columns = 'sex' ,aggfunc='sum')

year_sex_population.fillna(0, inplace=True)

year_sex_population 
plt.figure(figsize=(4,10))

sns.heatmap(year_sex_population ,cmap= sns.cubehelix_palette(200))

plt.show()
year_sex_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='year', columns = 'sex' ,aggfunc='sum')

year_sex_suicides_100k_pop.fillna(0, inplace=True)

year_sex_suicides_100k_pop
plt.figure(figsize=(4,10))

sns.heatmap(year_sex_suicides_100k_pop,cmap= sns.cubehelix_palette(200))

plt.show()
year_sex_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='year', columns = 'sex' ,aggfunc='sum')

year_sex_gdp_for_year.fillna(0, inplace=True)

year_sex_gdp_for_year 
plt.figure(figsize=(4,10))

sns.heatmap(year_sex_gdp_for_year ,cmap= sns.cubehelix_palette(200))

plt.show()
year_sex_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='year', columns = 'sex' ,aggfunc='sum')

year_sex_gdp_per_capital.fillna(0, inplace=True)

year_sex_gdp_per_capital
plt.figure(figsize=(4,10))

sns.heatmap(year_sex_gdp_per_capital ,cmap= sns.cubehelix_palette(200))

plt.show()
year_age_suicides_no = pd.pivot_table(df, values = 'suicides_no',index = 'year', columns = 'age' ,aggfunc='sum')

year_age_suicides_no.fillna(0, inplace=True)

year_age_suicides_no
plt.figure(figsize=(6,10))

sns.heatmap(year_age_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])

plt.show()
year_age_population = pd.pivot_table(df, values= 'population',index='year', columns = 'age' ,aggfunc='sum')

year_age_population.fillna(0, inplace=True)

year_age_population 
plt.figure(figsize=(6,10))

sns.heatmap(year_age_population ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])

plt.show()
year_age_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='year', columns = 'age' ,aggfunc='sum')

year_age_suicides_100k_pop.fillna(0, inplace=True)

year_age_suicides_100k_pop
plt.figure(figsize=(6,10))

sns.heatmap(year_age_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])

plt.show()
year_age_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='year', columns = 'age' ,aggfunc='sum')

year_age_gdp_for_year.fillna(0, inplace=True)

year_age_gdp_for_year 
plt.figure(figsize=(6,10))

sns.heatmap(year_age_gdp_for_year ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])

plt.show()
year_age_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='year', columns = 'age' ,aggfunc='sum')

year_age_gdp_per_capital.fillna(0, inplace=True)

year_age_gdp_per_capital
plt.figure(figsize=(6,10))

sns.heatmap(year_age_gdp_per_capital ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])

plt.show()
year_generation_suicides_no = pd.pivot_table(df, values = 'suicides_no',index = 'year', columns = 'generation' ,aggfunc='sum')

year_generation_suicides_no.fillna(0, inplace=True)

year_generation_suicides_no
plt.figure(figsize=(6,10))

sns.heatmap(year_generation_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
year_generation_population = pd.pivot_table(df, values= 'population',index='year', columns = 'generation' ,aggfunc='sum')

year_generation_population.fillna(0, inplace=True)

year_generation_population 
plt.figure(figsize=(6,10))

sns.heatmap(year_generation_population ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
year_generation_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='year', columns = 'generation' ,aggfunc='sum')

year_generation_suicides_100k_pop.fillna(0, inplace=True)

year_generation_suicides_100k_pop
plt.figure(figsize=(6,10))

sns.heatmap(year_generation_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
year_generation_gdp_for_year = pd.pivot_table(df_gdp, values= 'gdp_for_year',index='year', columns = 'generation' ,aggfunc='sum')

year_generation_gdp_for_year.fillna(0, inplace=True)

year_generation_gdp_for_year 
plt.figure(figsize=(6,10))

sns.heatmap(year_generation_gdp_for_year ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
year_generation_gdp_per_capital = pd.pivot_table(df_gdp, values= 'gdp_per_capital',index='year', columns = 'generation' ,aggfunc='sum')

year_generation_gdp_per_capital.fillna(0, inplace=True)

year_generation_gdp_per_capital
plt.figure(figsize=(6,10))

sns.heatmap(year_generation_gdp_per_capital ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
sex_age_suicides_no = pd.pivot_table(df, values = 'suicides_no',index = 'sex', columns = 'age' ,aggfunc='sum')

sex_age_suicides_no.fillna(0, inplace=True)

sex_age_suicides_no
plt.figure(figsize=(5,2))

sns.heatmap(sex_age_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])

plt.show()
sex_age_population = pd.pivot_table(df, values= 'population',index='sex', columns = 'age' ,aggfunc='sum')

sex_age_population.fillna(0, inplace=True)

sex_age_population 
plt.figure(figsize=(5,2))

sns.heatmap(sex_age_population ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])

plt.show()
sex_age_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='sex', columns = 'age' ,aggfunc='sum')

sex_age_suicides_100k_pop.fillna(0, inplace=True)

sex_age_suicides_100k_pop
plt.figure(figsize=(5,2))

sns.heatmap(sex_age_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'])

plt.show()
sex_generation_suicides_no = pd.pivot_table(df, values = 'suicides_no',index = 'sex', columns = 'generation' ,aggfunc='sum')

sex_generation_suicides_no.fillna(0, inplace=True)

sex_generation_suicides_no
plt.figure(figsize=(5,2))

sns.heatmap(sex_generation_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
sex_generation_population = pd.pivot_table(df, values= 'population',index='sex', columns = 'generation' ,aggfunc='sum')

sex_generation_population.fillna(0, inplace=True)

sex_generation_population 
plt.figure(figsize=(5,2))

sns.heatmap(sex_generation_population ,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
sex_generation_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='sex', columns = 'generation' ,aggfunc='sum')

sex_generation_suicides_100k_pop.fillna(0, inplace=True)

sex_generation_suicides_100k_pop
plt.figure(figsize=(5,2))

sns.heatmap(sex_generation_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
generation_age_suicides_no = pd.pivot_table(df, values = 'suicides_no',index = 'generation', columns = 'age' ,aggfunc='sum')

generation_age_suicides_no.fillna(0, inplace=True)

generation_age_suicides_no
plt.figure(figsize=(7,3))

sns.heatmap(generation_age_suicides_no,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'],yticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
generation_age_population = pd.pivot_table(df, values= 'population',index='generation', columns = 'age' ,aggfunc='sum')

generation_age_population.fillna(0, inplace=True)

generation_age_population 
plt.figure(figsize=(7,3))

sns.heatmap(generation_age_population ,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'],yticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
generation_age_suicides_100k_pop = pd.pivot_table(df, values= 'suicides/100k pop',index='generation', columns = 'age' ,aggfunc='sum')

generation_age_suicides_100k_pop.fillna(0, inplace=True)

generation_age_suicides_100k_pop
plt.figure(figsize=(7,3))

sns.heatmap(generation_age_suicides_100k_pop,cmap= sns.cubehelix_palette(200),xticklabels=['5-14','15-24','25-34','35-54','55-74','75+'],yticklabels = ['G.I. Generation','Silent','Boomers','Generation X','Millenials','Generation Z'])

plt.show()
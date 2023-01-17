# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import seaborn as sns
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df_cases = pd.read_csv("/kaggle/input/covid19-20200321/ncov-Confirmed_20200829.csv")

df_deaths = pd.read_csv("/kaggle/input/covid19-20200321/ncov-Deaths_20200829.csv")
df_cases
df_deaths
df_cases.drop(df_cases.head(1).index, inplace=True)

df_cases.rename(columns={"Date": "date", "Value": "value_cases", "Country/Region": "country"}, inplace = True)

df_cases["value_cases"] = df_cases["value_cases"].astype(int)



df_deaths.drop(df_deaths.head(1).index, inplace=True)

df_deaths.rename(columns={"Date": "date", "Value": "value_deaths", "Country/Region": "country"}, inplace = True)

df_deaths["value_deaths"] = df_deaths["value_deaths"].astype(int)
df_total_cases = df_cases[["date", "value_cases"]]

df_total_deaths = df_deaths[["date", "value_deaths"]]

df_total_cases["value_cases"]=df_total_cases["value_cases"].astype(int)

df_total_deaths["value_deaths"]=df_total_deaths["value_deaths"].astype(int)
df_merge = pd.concat([df_deaths, df_cases], axis=1, join='inner')

df_merge = df_merge.loc[:,~df_merge.columns.duplicated()]

df_merge.drop(["Province/State","Lat","Long"], axis=1, inplace = True)

df_merge = df_merge.groupby(['country','date'])["value_deaths","value_cases"].sum()

df_merge.reset_index(inplace=True)

df_merge
df_merge.describe()
df_total_cases.head()
df_merge.to_csv("df_merge.csv")
df_total_deaths.head()
df_total_cases = df_total_cases.groupby(['date']).sum()

df_total_deaths = df_total_deaths.groupby(['date']).sum()



df_total_cases.reset_index(inplace=True)

df_total_deaths.reset_index(inplace=True)
df_total_con = pd.concat([df_total_cases, df_total_deaths], axis=1)

df_total_con = df_total_con.loc[:,~df_total_con.columns.duplicated()]

df_total_con.head()
fig, ax = plt.subplots(figsize=(15,5))

df_total_con.plot(x='date', y='value_cases', ax=ax)
fig, ax = plt.subplots(figsize=(15,5))

df_total_con.plot(x='date', y='value_deaths', ax=ax, color='r')
df_merge.country.unique()
df_merge.head()
country_list = ['Azerbaijan','Georgia','Iran','Russia','Germany','Italy','Turkey','Ukraine','France','Spain','US','United Kingdom']
df_selected_cases = df_merge[df_merge['country'].isin(country_list)]

df_selected_cases = df_selected_cases[df_selected_cases['date'] > "2020-02-21"]

df_selected_cases = df_selected_cases



df_selected_cases
plt.figure(figsize=(15,10))

g = sns.lineplot(x="date", y="value_cases", hue="country", data=df_selected_cases)

plt.setp(g.get_xticklabels(), rotation=90)

g
plt.figure(figsize=(15,10))

g = sns.lineplot(x="date", y="value_deaths", hue="country", data=df_selected_cases)

plt.setp(g.get_xticklabels(), rotation=90)

g
df_after_10_d = df_merge[df_merge['country'].isin(['Azerbaijan','Georgia','Iran','Russia','Germany','Italy','Turkey','Ukraine','France','Spain','US','United Kingdom'])]

df_after_10_d = df_after_10_d[df_after_10_d['value_deaths'] >= 10]

df_after_10_d.reset_index(inplace=True)

df_after_10_d['rn'] = df_after_10_d.groupby(['country']).cumcount()

df_after_10_d.head()
plt.figure(figsize=(15,10))

g = sns.lineplot(x="rn", y="value_deaths", hue="country", data=df_after_10_d, style="country", dashes=False)

plt.setp(g.get_xticklabels(), rotation=90)

g
df_az_cases = df_selected_cases[df_selected_cases['country'].isin(['Azerbaijan'])]



plt.figure(figsize=(15,8))

g = sns.lineplot(x="date", y="value_cases", hue="country", data=df_az_cases)

plt.setp(g.get_xticklabels(), rotation=90)

g
df_az_cases = df_selected_cases[df_selected_cases['country'].isin(['Azerbaijan'])]



plt.figure(figsize=(15,8))

g = sns.lineplot(x="date", y="value_deaths", hue="country", data=df_az_cases)

plt.setp(g.get_xticklabels(), rotation=90)

g
df_az_cases = df_selected_cases[df_selected_cases['country'].isin(['Italy'])]

# df_az_cases = df_az_cases.drop(df_az_cases.tail(30).index, axis=0)





plt.figure(figsize=(15,8))

g = sns.lineplot(x="date", y="value_cases", hue="country", data=df_az_cases)

plt.setp(g.get_xticklabels(), rotation=90)

g
df_az_cases = df_selected_cases[df_selected_cases['country'].isin(['France'])]



plt.figure(figsize=(15,8))

g = sns.lineplot(x="date", y="value_cases", hue="country", data=df_az_cases)

plt.setp(g.get_xticklabels(), rotation=90)

g
df_az_cases = df_selected_cases[df_selected_cases['country'].isin(['Spain'])]



plt.figure(figsize=(15,8))

g = sns.lineplot(x="date", y="value_cases", hue="country", data=df_az_cases)

plt.setp(g.get_xticklabels(), rotation=90)

g
df_az_cases = df_selected_cases[df_selected_cases['country'].isin(['US'])]



plt.figure(figsize=(15,8))

g = sns.lineplot(x="date", y="value_cases", hue="country", data=df_az_cases)

plt.setp(g.get_xticklabels(), rotation=90)

g
import requests

url = 'https://worldometers.info/world-population/population-by-country'

header = {

  "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36",

  "X-Requested-With": "XMLHttpRequest"

}



r = requests.get(url, headers=header)

df_world_population = pd.read_html(r.text)[0]

df_world_population
df_world_population.rename(columns={"Country (or dependency)": "country", "Population (2020)": "population", "Med. Age" : "med_age"}, inplace = True)

columns = ['country', 'population', 'med_age']

df_world_population.drop(df_world_population.columns[~df_world_population.columns.isin(columns)], axis=1, inplace=True)

df_world_population.country.replace(['United States'], ['US'], inplace=True)

df_world_population
df_merge
df_with_pop = pd.merge(df_merge, df_world_population, on='country', how='inner')

df_with_pop
df_new = df_with_pop[df_with_pop['date'] == "2020-03-20"]

df_new = df_new[df_new['country'].isin(country_list)]

df_new
df_new.info()
df_new["case_per_population"] = df_new["value_cases"]/df_new["population"]*100

df_new["death_per_population"] = round(df_new["value_deaths"]/df_new["population"]*100,6)

df_new["death_per_case"] = round(df_new["value_deaths"]/df_new["value_cases"]*100,6)
df_new
fig, ax = plt.subplots(figsize=(15,8))

sns.set(style="whitegrid")

sns.barplot(x="country", y="death_per_case", data=df_new)
fig, ax = plt.subplots(figsize=(15,8))

sns.set(style="whitegrid")

sns.barplot(x="country", y="case_per_population", data=df_new)
fig, ax = plt.subplots(figsize=(15,8))

sns.set(style="whitegrid")

sns.barplot(x="country", y="death_per_population", data=df_new)
fig, ax = plt.subplots(figsize=(15,8))

sns.set(style="whitegrid")

sns.set_context('paper')

sns.barplot(x="value_deaths", y="country", data=df_new.sort_values(by=['value_deaths'], ascending=False))

sns.despine(left=True, bottom=True)

plt.show()
df_cases.head()
temp = df_cases.groupby(['date', 'country'])['value_cases'].sum()

temp = temp.reset_index().sort_values(by=['date', 'country'])

plt.style.use('seaborn')

g = sns.FacetGrid(temp, col="country", hue="country",  sharey=False, col_wrap=5)

g = g.map(plt.plot, "date", "value_cases")

g.set_xticklabels(rotation=90)

plt.show()
import nltk

import pandas as pd

import os

import pycountry

import numpy as np

import spacy

import warnings; warnings.simplefilter('ignore')
df = pd.read_csv("/kaggle/input/covidtask8/task8.csv")

df = df[['title', 'text']]

df
all_countries = [i.name.lower() for i in pycountry.countries]

all_countries.append('korea')

all_countries.append('usa')

all_countries.append('uk')

all_countries.append('britain')
geo_df = pd.DataFrame()



for i, row in df.iterrows():

    pars = row['text'].split('\n')

    for par in pars:

        for country in all_countries:

            if country in par.lower():

#                 print(country)

                geo_df = geo_df.append([[row['title'], par, country]])
geo_df.columns = ['title', 'text', 'country']

geo_df
mask = geo_df['text'].str.contains('deaths')

death_df = geo_df[mask].reset_index()

del death_df['index']

death_df
countries_mentioned = []

_ = [countries_mentioned.append(row['country']) for i, row in death_df.iterrows()]

c = []

t = []



for i in countries_mentioned:

    if i not in c:

        c.append(i)

        t.append(1)

    else:

        t[c.index(i)] += 1

        

country_dict = dict(zip(c, t))
death_df = pd.DataFrame(death_df['text'].drop_duplicates())

death_df
import re



d_rate = []

c_rate = []



for i, r in death_df.iterrows():

    sentences = r['text'].lower().replace(',', '').split('.')

    for sentence in sentences:

        if 'cases' in sentence or 'deaths' in sentence:

            rege = re.findall(r'[1-9][0-9]+ cases', sentence) 

            if rege != []: c_rate.append([rege, sentence])

                

            rege = re.findall(r'[1-9][0-9]+ infections', sentence) 

            if rege != []: c_rate.append([rege, sentence])

                

            rege = re.findall(r'[1-9][0-9]+ deaths', sentence) 

            if rege != []: d_rate.append([rege, sentence])
months = ['december', 'january', 'february', 'march', 'april']



deaths = pd.DataFrame(d_rate, columns=['deaths', 'text'])

cases = pd.DataFrame(c_rate, columns=['cases', 'text'])
data_for_d = pd.DataFrame(np.zeros((3, len(months))))

data_for_d.columns = months

data_for_d.index  = ['total', 'times_mentioned', 'mean']



total = data_for_d.iloc[0]

t_mentioned = data_for_d.iloc[1]

mean = data_for_d.iloc[2]



for i, row in deaths.iterrows():

    for month in months:

        if month in row['text']:

            d = 0

            for death in row['deaths']:

                d += int(re.findall(r'[1-9][0-9]+', death)[0])

                t_mentioned[month] += 1

#                 add_pl = data_for_c.iloc[0]

#                 print(add_pl)

            total[month] += d

    



for month in months:

    mean[month] = total[month] / t_mentioned[month]



data_for_d = data_for_d.dropna(axis=1).astype(np.int64)
import matplotlib.pyplot as plt



means = list(data_for_d.iloc[2])

plt.bar(months[:-1], means)

plt.title("Deaths of COVID")

plt.show()
# data_for_cases = pd.DataFrame(0, index=np.arange(1),columns=months)

data_for_c = pd.DataFrame(np.zeros((3, len(months))))

data_for_c.columns = months

data_for_c.index  = ['total', 'times_mentioned', 'mean']



total = data_for_c.iloc[0]

t_mentioned = data_for_c.iloc[1]

mean = data_for_c.iloc[2]



for i, row in cases.iterrows():

    for month in months:

        if month in row['text']:

            d = 0

            for case in row['cases']:

                d += int(re.findall(r'[1-9][0-9]+', case)[0])

                t_mentioned[month] += 1

#                 add_pl = data_for_c.iloc[0]

#                 print(add_pl)

            total[month] += d



for month in months:

    mean[month] = total[month] / t_mentioned[month]



data_for_c = data_for_c.astype(np.int64)
means = list(data_for_c.iloc[2])

plt.bar(months[:-1], means[:-1])

plt.title('Cases of COVID')

plt.show()
# data_for_cases = pd.DataFrame(0, index=np.arange(1),columns=months)

data = pd.DataFrame(np.zeros((3, len(c))))

data.columns = c

data.index  = ['total', 'times_mentioned', 'mean']



total = data.iloc[0]

t_mentioned = data.iloc[1]

mean = data.iloc[2]



for i, row in cases.iterrows():

    for country in c:

        if country in row['text']:

            d = 0

            for case in row['cases']:

                d += int(re.findall(r'[1-9][0-9]+', case)[0])

                t_mentioned[country] += 1

            total[country] += d



mean['usa'] += mean['united states']

t_mentioned['usa'] += t_mentioned['united states']



# mean['uk'] += mean['britain']

# t_mentioned['uk'] += t_mentioned['britain']



# del data['britain']

    

try:

    for country in c:

        mean[country] = total[country] / t_mentioned[country]

except: pass

        

data = data.dropna(axis=1).astype(np.int64)

del data['united states']

data
from matplotlib.pyplot import figure

figure(num=None, figsize=(50, 10), facecolor='w', edgecolor='k')

plt.rcParams.update({'font.size': 20})



means = list(data.iloc[2])

plt.bar(list(data.columns), means)

plt.title('Cases of COVID by country')

plt.show()
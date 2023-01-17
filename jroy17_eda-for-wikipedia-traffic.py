import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
dataset = pd.read_csv('../input/train_1.csv')
dataset.head()
len(dataset)
page = dataset["Page"]
page.head()
# Name of the article

name = []



# Project the article was search in

project = []



# The object used to access the article

access = []



# Agent

agent = []
# Spliting the information in the page

for i in range(len(dataset)):

    split_row = page[i].split('_')

    j = len(split_row)-1

    agent.append(split_row[j])

    j -= 1

    access.append(split_row[j])

    j -= 1

    project.append(split_row[j])

    j -= 1

    while(j>=0):

        name.append(split_row[j])

        j -= 1
unique_agents = set(agent)

print(unique_agents)
unique_access = set(access)

print(unique_access)
unique_projects = set(project)

print(unique_projects)
language = []

source = []
# Extracting the language and source from project

for i in range(len(dataset)):

    split_row = project[i].split('.')

    if split_row[1] == 'wikipedia':

        language.append(split_row[0])

        source.append(split_row[1])

    else :

        language.append("Media")

        source.append(split_row[1])
unique_languages = set(language)

print(unique_languages)
unique_sources = set(source)

print(unique_sources)
print(len(source))   

    
dataset["language"] = language

dataset["source"] = source

dataset["access"] = access

dataset["agent"] = agent
dataset = dataset.fillna(0.0)
dataset.head()
sns.set(style="whitegrid", color_codes=True)

def plotFrequency(key):

    plt.figure(figsize=(12,8))

    ax = sns.countplot(x=key, data=dataset,palette="GnBu_d")

    plt.ylabel('Frequency')

    plt.title('Frequency distribution')

    plt.show()



plotFrequency("language")

plotFrequency("source")

plotFrequency("access")

plotFrequency("agent")
days = [r for r in range(dataset.shape[1]-5)]



def plotDailyViews(table, element):

    fig = plt.figure(1,figsize=[12,12])

    plt.ylabel('Views')

    plt.xlabel('Days')

    plt.title('Comparison of '+element)

    label={#languages

            'na':'Media',

            'de':'German',

            'en':'English',

            'es':'Spanish',

            'fr':'French',

            'ja':'Japanese',

            'ru':'Russian',

            'zh':'Chinese',

           #source

            'mediawiki':'mediawiki',

            'wikimedia':'wikimedia',

            'wikipedia':'wikipedia',

           #access

            'all-access':'all-access',

            'desktop':'desktop',

            'mobile-web':'mobile-web',

           #agents

            'all-agent':'all-agent',

            'spider':'spider'

           }

    done = None

    for key in label:

        if key in table.columns:

            plt.plot(days,table[key],label = label[key] )

    plt.legend()

    plt.show()
# Language daily views

language_daily_sum = dataset.groupby('language').sum().reset_index()

language_daily_sum_transpose = language_daily_sum.T

language_daily_sum_transpose.head()
language_daily_sum_transpose.columns=["na","de","en","es","fr","ja","ru","zh"]

language_daily_sum_transpose = language_daily_sum_transpose[1:]

plotDailyViews(language_daily_sum_transpose,"Languages")
# Source daily views

source_daily_sum = dataset.groupby('source').sum().reset_index()

source_daily_sum_transpose = source_daily_sum.T

source_daily_sum_transpose.head()
source_daily_sum_transpose.columns=["mediawiki","wikimedia","wikipedia"]

source_daily_sum_transpose = source_daily_sum_transpose[1:]

plotDailyViews(source_daily_sum_transpose, "Sources")
# Access daily views

access_daily_sum = dataset.groupby('access').sum().reset_index()

access_daily_sum_transpose = access_daily_sum.T

access_daily_sum_transpose.head()
access_daily_sum_transpose.columns=["all-access","desktop","mobile-web"]

access_daily_sum_transpose = access_daily_sum_transpose[1:]

plotDailyViews(access_daily_sum_transpose, "Access")
# Agent daily views

agent_daily_sum = dataset.groupby('agent').sum().reset_index()

agnet_daily_sum_transpose = agent_daily_sum.T

agnet_daily_sum_transpose.head()
agnet_daily_sum_transpose.columns=["all-agent","spider"]

agnet_daily_sum_transpose = agnet_daily_sum_transpose[1:]

plotDailyViews(agnet_daily_sum_transpose, "Agent")
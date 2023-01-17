import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

plt.rcParams['figure.figsize'] = 10,8

import warnings

warnings.filterwarnings('ignore')
dataset_2015 = pd.read_csv('../input/2015.csv')
dataset_2015.head(2)

dataset_2015.describe()
vis1 = plt.scatter(x = dataset_2015['Happiness Score'], y = dataset_2015['Health (Life Expectancy)']);

plt.xlabel('happiness Score');

plt.ylabel('Health (Life Expectancy)');
vis2 = plt.scatter(x = dataset_2015['Happiness Score'], y = dataset_2015['Freedom']);

plt.xlabel('happiness Score');

plt.ylabel('Freedom');
vis3 = sns.lmplot(data = dataset_2015, x = "Happiness Rank", y = "Trust (Government Corruption)", hue = "Region", \

                  fit_reg = False,  scatter_kws = {"s": 100},size = 10);
vis4 = sns.barplot(data = dataset_2015, x = "Trust (Government Corruption)", y = "Region",)

dataset_2016 = pd.read_csv('../input/2016.csv')
dataset_2016.head(2)
dataset_2016.describe()
vis1 = plt.scatter(x = dataset_2016['Happiness Score'], y = dataset_2016['Health (Life Expectancy)']);

plt.xlabel('happiness Score');

plt.ylabel('Health (Life Expectancy)');
vis2 = plt.scatter(x = dataset_2016['Happiness Score'], y = dataset_2016['Freedom']);

plt.xlabel('happiness Score');

plt.ylabel('Freedom');
vis3 = sns.lmplot(data = dataset_2016, x = "Happiness Rank", y = "Trust (Government Corruption)", hue = "Region", \

                  fit_reg = False,  scatter_kws = {"s": 100},size = 10);

vis4 = sns.barplot(data = dataset_2016, x = "Trust (Government Corruption)", y = "Region")

dataset_2017 = pd.read_csv('../input/2017.csv')
dataset_2017.head(2)
dataset_2017.describe()
vis1 = plt.scatter(x = dataset_2017['Happiness.Score'], y = dataset_2017['Health..Life.Expectancy.']);

plt.xlabel('happiness Score');

plt.ylabel('Health (Life Expectancy)');
vis2 = plt.scatter(x = dataset_2017['Happiness.Score'], y = dataset_2017['Freedom']);

plt.xlabel('happiness Score');

plt.ylabel('Freedom');
vis3 = sns.lmplot(data = dataset_2017, x = "Happiness.Rank", y = "Trust..Government.Corruption.",\

                  fit_reg = False,  scatter_kws = {"s": 100},size = 10)
fig, axs = plt.subplots(ncols=2,sharex=True,sharey = True)

sns.barplot(data = dataset_2015, x = "Trust (Government Corruption)", y = "Region", ax=axs[0]);

sns.barplot(data = dataset_2016, x = "Trust (Government Corruption)", y = "Region", ax=axs[1]);
fig, axs = plt.subplots(ncols=2,sharex=True,sharey = True)

sns.barplot(data = dataset_2015, x = "Freedom", y = "Region", ax=axs[0]);

sns.barplot(data = dataset_2016, x = "Freedom", y = "Region", ax=axs[1]);
fig, axs = plt.subplots(ncols=2,sharex=True,sharey = True)

sns.barplot(data = dataset_2015, x = "Health (Life Expectancy)", y = "Region", ax=axs[0]);

sns.barplot(data = dataset_2016, x = "Health (Life Expectancy)", y = "Region", ax=axs[1]);
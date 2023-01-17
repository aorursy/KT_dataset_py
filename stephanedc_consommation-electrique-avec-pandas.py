# Import des librairies d'analyse et de graphing

import pandas as pd

import seaborn as sns

import numpy as np

from numpy import nan

from numpy import isnan

from scipy.stats import norm

from matplotlib import pyplot
dataset = pd.read_csv('../input/household_power_consumption.txt',

                   sep=';',                            # separateur = ;

                   header=0,                           # ligne des header = 1ère ligne

                   low_memory=False,                   # parsing complet

                   infer_datetime_format=True,         # parsing des dates

                   parse_dates={'datetime':[0,1]},     # les dates se parsent via les 2 premières colonnes

                   index_col=['datetime'])             # l'index sera crée dans une colonne 'datetime'
# Visualisation de quelques éléments du dataset (qui est un DataFrame Panda)

dataset.head()
# Principales informations sur le Dataset:

# Nous trouvons ici principalement le nombre de lignes et les dates des données

# Nous trouvons également le type des colonnes

dataset.info()
# Comptage des valeurs manquantes dans le dataset

dataset.isnull().sum()
# Visualisation de quelques manquants sur la colonne en question

dataset.loc[dataset.Sub_metering_3.isnull()].head()
# Suppression des valeurs non numériques et conversion des colonnes en float

dataset = dataset.dropna()

dataset = dataset.astype('float32')
# Groupes pour le resampling

daily_groups = dataset.resample('D')

weekly_groups = dataset.resample('W')



# Jeux de données resamplés

daily_data = daily_groups.sum()

weekly_data = weekly_groups.sum()
# Visualisation des données / jour

daily_data.head()
# Plot de toutes les features

ax = daily_data.plot(title='Daily Household Power', figsize=(15,5))
# Filtre sur les dates des données de la dernière année

mask = (daily_data.index > '2010-01-01') & (daily_data.index < '2011-01-01')

ax = daily_data.loc[mask].plot(title='Daily Household Power 2010', figsize=(15,5))
# Suppression de la feature voltage pour observer les autres features

ddata = daily_data.drop(columns=['Voltage'])

ax= ddata.plot(title='Household Power / day', figsize=(15,5))
# Suppression de la feature voltage pour observer les autres features weekly

wkdata = weekly_data.drop(columns=['Voltage'])

ax= wkdata.plot(title='Household Power / week', figsize=(15,5))
# Plotting de la puissance totale par jour

ax = daily_data.Global_active_power.plot(title='Global_active_power by days', figsize=(15,5))
# Plotting de la puissance totale par semaine

ax= weekly_data.Global_active_power.plot(title='Global_active_power by weeks', figsize=(15,5))
# Observation de la distribution de la puissance par rapport à une distribution Gaussienne

ax = sns.distplot(daily_data['Global_active_power'], fit=norm, bins=50, kde=True);
# Analyse de la distribution Gaussienne

ax = stats.probplot(daily_data['Global_active_power'], plot=pyplot)
# Plotting de l'ensemble des features du jeu de données deux par deux

ax = sns.pairplot(daily_data, height = 2.5)
# Plotting de deux variables qui semblent corrélées:

ax = daily_data.plot.scatter(x='Global_active_power', y='Global_intensity')
# Analyse plus détaillée entre les deux features précédentes

ax=sns.jointplot(daily_data['Global_active_power'], daily_data['Global_intensity'], kind='kde', 

                   joint_kws={'alpha':0.5}, 

                   xlim=(0, 3200), 

                   ylim=(0, 14000),

                   height=6)
# Analyse entre deux autres features

ax=sns.jointplot(daily_data['Global_active_power'], daily_data['Sub_metering_3'], kind='kde', 

                   joint_kws={'alpha':0.5}, 

                   xlim=(0, 3000), 

                   ylim=(0, 18000),

                   height=6)
# Heatmap de corrélation de type pearson

# Les features corrélées auront des valeurs proche de 1

pearson = daily_data.corr(method='pearson')

mask = np.zeros_like(pearson)

mask[np.triu_indices_from(mask)] = True

sns.heatmap(pearson, vmax=1, vmin=0, square=True, cbar=True, annot=True, cmap="YlGnBu", mask=mask);
# 3D plotting

from mpl_toolkits.mplot3d import Axes3D

fig = pyplot.figure(figsize=(80, 80))

ax = fig.add_subplot(666, projection='3d')

xs = daily_data['Global_active_power']

ys = daily_data['Global_intensity']

zs = daily_data['Sub_metering_3']

ax.scatter(xs, ys, zs, s=100, alpha=0.2, edgecolors='w')

ax.set_xlabel('Active Power')

ax.set_ylabel('Intensity')

ax.set_zlabel('Sub_metering_3')

ax.set(zlim=(0, 20000))

pyplot.show()
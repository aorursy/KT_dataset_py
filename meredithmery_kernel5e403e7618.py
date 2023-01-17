# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import matplotlib.pyplot as plt

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import numpy as np

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print (os.path.join(dirname, filename))

        data = os.path.join(dirname, filename)

        print(pd.read_csv(data))

# Any results you write to the current directory are saved as output.
#Ouverture des fichiers CSV

dt = pd.read_csv("/kaggle/input/world-happiness/2017.csv")

dt2 = pd.read_csv("/kaggle/input/world-happiness/2016.csv")

dt3 = pd.read_csv("/kaggle/input/world-happiness/2015.csv")
#On renomme les colonnes de chaque Datbase car les points ou les espaces peuvent induire

#des erreurs dans les requètes.Le but sera plus tard de faire des comparaisons entre BDD :

print(dt.columns)

print ("\nColonnes du Dataset 2017 avec apres modification : \n")

dt.rename(columns={'Economy..GDP.per.Capita.': 'Economy','Happiness.Score': 'HappiScore', 'Happiness.Rank': 'HappiRank','Whisker.high': 'WhiskerHigh','Whisker.low': 'WhiskerLow','Health..Life.Expectancy.': 'Health','Trust..Government.Corruption.': 'Trust','Dystopia.Residual': 'Dystopia'}, inplace=True)

print(dt.columns)

dt2.rename(columns={'Economy (GDP per Capita)': 'Economy','Happiness Score': 'HappiScore', 'Happiness Rank': 'HappiRank','Upper Confidence Interval': 'WhiskerHigh','Lower Confidence Interval': 'WhiskerLow',

                   'Health (Life Expectancy)': 'Health','Trust (Government Corruption)': 'Trust','Dystopia Residual': 'Dystopia'}, inplace=True)

dt3.rename(columns={'Economy (GDP per Capita)': 'Economy','Happiness Score': 'HappiScore', 'Happiness Rank': 'HappiRank','Standard Error': 'StandardError',

                   'Health (Life Expectancy)': 'Health','Trust (Government Corruption)': 'Trust','Dystopia Residual': 'Dystopia'}, inplace=True)

print(dt2.columns)

print(dt3.columns)

#Les matrices n'ont pas les mêmes tailles :

print("2017 = ",dt.shape)

print("2016 = ",dt2.shape)

print("2015 = ",dt3.shape)
#graphique du taux de bonheur en fonction du critère famillle

sns.barplot(y="HappiScore", x="Family", data=dt)
dt["HappiScore"].min
#On choisit de créer une nouvelle matrice composée des pays avec un score >7 :

MH = dt.loc[dt["HappiScore"] >= 7,:]
# On l'affiche

print(MH)
#Premier tracé d'un graphe, si la courbe est proportionnelle le tracé fonctionne : ok

x= MH.HappiScore

y= MH.HappiScore

plt.plot(x, y)

plt.legend('ABCDEF', ncol=2, loc='upper left');
y= MH.Economy

x= MH.HappiRank

plt.plot(x, y)
y= MH.Family

x= MH.HappiRank

plt.plot(x, y)
y= MH.Family

x= MH.HappiRank

plt.plot(x, y)
y= MH.Health

x= MH.HappiRank

plt.plot(x, y)
y= MH.Freedom

x= MH.HappiRank

plt.plot(x, y)
y= MH.Generosity

x= MH.HappiRank

plt.plot(x, y)
y= MH.Trust

x= MH.HappiRank

plt.plot(x, y)
y= MH.Dystopia

x= MH.HappiRank

plt.plot(x, y)
y= MH.Freedom

x= MH.HappiRank

plt.plot(x, y)
#autre sorte de graphique :

sns.barplot(y="Family", x="HappiRank", data=MH)
sns.barplot(y="Freedom", x="HappiRank", data=MH)
# On enlève les colonnes non utilisées de MH 

MH = MH.loc[:, ['HappiRank','HappiScore','Country','Economy','Freedom','Health','Trust','Generosity']]
print(MH)
# Economy semble un paramètre important :

sns.barplot(y="Economy", x="HappiRank", data=MH)

MH.loc[:,['HappiRank','Country','Economy']]
#Dessin d'un graphique sur les données des caractéristiques en fonction des pays

dfbar = pd.melt(MH, id_vars="Country", var_name="categories", value_name="HappiScore")

graph2 = sns.catplot(x="Country", y="HappiScore", hue="categories", data=dfbar, kind="bar")

graph2.fig.set_size_inches(40,12)
#On peut penser que certaines caractéristiques sont liées entre elles ?

SR = dt.loc[:, ['Country','Economy','Health']]

print(SR)
dfbar = pd.melt(SR, id_vars="Country", var_name="categories", value_name="count")

graph2 = sns.catplot(x="Country", y="count", hue="categories", data=dfbar, kind="bar")

graph2.fig.set_size_inches(60,12)
#Proportionalité entre PIB et accés aux soins, dans tous les pays ?

sns.barplot(y="Economy", x="Health", data=SR)

MH.head()
MHE = MH.loc[:,['HappiRank','HappiScore','Country','Economy','Freedom','Generosity','Family','Trust','Health']]
# graphique de répartition de chaque caractéristiques par pays (heureux (>7))

dfbar = pd.melt(MHE, id_vars="Country", var_name="categories", value_name="HappiScore")

graph2 = sns.catplot(x="Country", y="HappiScore", hue="categories", data=dfbar, kind="bar")

sns.catplot

graph2.fig.set_size_inches(40,12)
# Affichons sans le rang et le score de happiness

dfbar1 = dfbar.loc[dfbar["categories"] != 'HappiRank',:]

#Comparaison de toutes les caractéristiques par pays

dfbar = pd.melt(MHE, id_vars="Country", var_name="categories", value_name="HappiScore")

dfbar2= dfbar1.loc[dfbar1["categories"] != 'HappiScore',:]

dfbar2= dfbar1.loc[dfbar1["categories"] != 'HappiScore',:]

graph2 = sns.catplot(x="Country", y="HappiScore", hue="categories", data=dfbar2, kind="bar")

sns.catplot

graph2.fig.set_size_inches(40,12)
# comparaison de deux caractéristiques sélectionnées  ?

#On peut l'utiliser comme une fonction il suffit de changer les deux paramètres char 1 et char 2 ;

char1 = 'Trust'

char2 = 'Freedom'

# dfbar2= dfbar1.loc[dfbar1["categories"] == 'char1,:] OK

dfbar2= dfbar1.loc[(dfbar1["categories"] == char1) |(dfbar1["categories"] == char2),:]

# print(dfbar2)

graph2 = sns.catplot(x="Country", y="HappiScore", hue="categories", data=dfbar2, kind="bar")

sns.catplot

graph2.fig.set_size_inches(40,12)
# Comparéson de deux autres caractéristiques

char1 = 'Generosity'

char2 = 'Economy'

# dfbar2= dfbar1.loc[dfbar1["categories"] == 'char1,:] OK

dfbar2= dfbar1.loc[(dfbar1["categories"] == char1) |(dfbar1["categories"] == char2),:]

# print(dfbar2)

graph2 = sns.catplot(x="Country", y="HappiScore", hue="categories", data=dfbar2, kind="bar")

sns.catplot

graph2.fig.set_size_inches(40,12)
# Où est classé la Chine dans le critère Liberté ?

print (dt.loc[dt.Country == 'China'])

# Malgrés la surveillance accrue de la république populaire de Chine communiste, les habitants se sentent

#ils libres  ?

print ((dt.loc[dt.Country == 'China']).Freedom)

china_free = (dt.loc[dt.Country == 'China']).Freedom



# Creation d'un classement des pays en fonction de leurs scores de liberté

Lib = dt[['Country','Freedom']]

Lib = Lib.sort_values(by ="Freedom", ascending = False)

print(Lib)

# et la Chine alors ? où est elle dans le classement ?

print ((Lib.loc[Lib.Country == 'China']).Freedom)
#78 ème sur 139 donc en dessous du milieu de tableaux
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.



df = pd.read_csv('/kaggle/input/uci-pro-road-cycling-dataset/UCIRiders0519_2x.csv', delimiter=',')
df_pcteams = df[df["Category"] == "PCT"]

df_wtteams = df[(df["Category"] == "WTT") | (df["Category"] == "PRO")]

df_pro = df_pcteams.append(df_wtteams)

df_pro.head(10)
all_by_year = {}

wtt_by_year = {}

pct_by_year = {}

for y in set(df_pro['Year']):

    all_by_year[y] = df_pro[df_pro['Year'] == y]

    wtt_by_year[y] = df_wtteams[df_pro['Year'] == y]

    pct_by_year[y] = df_pcteams[df_pro['Year'] == y]

for y in set(df_pro['Year']):

    print(y)

    print(str(len(all_by_year[y]))+"  <-  "+str(len(wtt_by_year[y]))+"  &  "+str(len(pct_by_year[y])))

    print('----------------------------------------')

import matplotlib.pyplot as plt

fig = plt.figure()

fig.suptitle("Number of racers in top categories by year")

x = sorted(set(df_pro['Year']))

nb_by_year_all = {}

nb_by_year_wt = {}

nb_by_year_cp = {}

for year in x:

    nb_by_year_all[year] = len(all_by_year[year])

    nb_by_year_wt[year] = len(wtt_by_year[year])

    nb_by_year_cp[year] = len(pct_by_year[year])

fig, ax = plt.subplots()

ax.plot(nb_by_year_all.keys(), nb_by_year_all.values(), label="All riders")

ax.plot(nb_by_year_wt.keys(), nb_by_year_wt.values(), label="WT riders")

ax.plot(nb_by_year_cp.keys(), nb_by_year_cp.values(), label="PC riders")

plt.legend()
nb_all_team_by_year = {}

nb_wt_team_by_year = {}

nb_pc_team_by_year = {}

for year in x:

    nb_all_team_by_year[year] = len(set(all_by_year[year]['Team Code']))

    nb_wt_team_by_year[year] = len(set(wtt_by_year[year]['Team Code']))

    nb_pc_team_by_year[year] = len(set(pct_by_year[year]['Team Code']))

fig2 = plt.figure()

fig2.suptitle("Number of teams in top categories by year")

fig2, ax2 = plt.subplots()

ax2.plot(nb_all_team_by_year.keys(), nb_all_team_by_year.values(), label="All teams")

ax2.plot(nb_wt_team_by_year.keys(), nb_wt_team_by_year.values(), label="WT teams")

ax2.plot(nb_pc_team_by_year.keys(), nb_pc_team_by_year.values(), label="PC teams")

plt.legend()
countries_by_year = {}

for year in x:

    countries_by_year[year] = set(all_by_year[year]['Country'])

riders_by_country_by_year = {}

for year in x:

    riders_by_country_by_year[year] = all_by_year[year]['Country'].value_counts()

    print(year)

    print(riders_by_country_by_year[year].head(5))

    print("**************")
fra = {}

bel = {}

ita = {}

esp = {}

ger = {}

ned = {}

for year in x:

    fra[year] = riders_by_country_by_year[year]['FRA']

    bel[year] = riders_by_country_by_year[year]['BEL']

    ita[year] = riders_by_country_by_year[year]['ITA']

    esp[year] = riders_by_country_by_year[year]['ESP']

    ger[year] = riders_by_country_by_year[year]['GER']

    ned[year] = riders_by_country_by_year[year]['NED']

fig3 = plt.figure()

fig3.suptitle("Number of riders from major countries")

fig3, ax3 = plt.subplots()

ax3.plot(fra.keys(), fra.values(), label="France")

ax3.plot(bel.keys(), bel.values(), label="Belgium")

ax3.plot(ita.keys(), ita.values(), label="Italy")

ax3.plot(esp.keys(), esp.values(), label="Spain")

ax3.plot(ger.keys(), ger.values(), label="Germany")

ax3.plot(ned.keys(), ned.values(), label="Netherlands")

plt.legend()
continents_by_year = {}

for year in x:

    continents_by_year[year] = set(all_by_year[year]['Continent'])

riders_by_continent_by_year = {}

for year in x:

    riders_by_continent_by_year[year] = all_by_year[year]['Continent'].value_counts()

    print(year)

    print(riders_by_continent_by_year[year])

    print("**************")
ame = {}

oce = {}

asi = {}

afr = {}

for year in x:

    ame[year] = riders_by_continent_by_year[year]['AME']

    oce[year] = riders_by_continent_by_year[year]['OCE']

    asi[year] = riders_by_continent_by_year[year]['ASI']

    afr[year] = riders_by_continent_by_year[year]['AFR']

fig4 = plt.figure()

fig4.suptitle("Number of extra-european riders by continent")

fig4, ax4 = plt.subplots()

ax4.plot(ame.keys(), ame.values(), label="Americas")

ax4.plot(oce.keys(), oce.values(), label="Oceania")

ax4.plot(asi.keys(), asi.values(), label="Asia")

ax4.plot(afr.keys(), afr.values(), label="Africa")

plt.legend()
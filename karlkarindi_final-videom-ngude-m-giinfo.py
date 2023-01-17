#Eesmärk oli vaadata ja uurida natuke videomängude müügi kohta olemasolevat infot. Põhiliselt

#vaatasime olemasolevat informatsiooni ja vaatasime, et mis meile huvitav tundus.
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt 

%matplotlib inline

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



vgs = pd.read_csv("../input/vgsales.csv")

vgs
print("Mängude arv: ", len(vgs))

žanrid = vgs["Genre"].unique()

print("Žanrite arv: ", len(žanrid))

väljastajad = vgs["Publisher"].unique()

print("Väljastajate arv: ", len(väljastajad))

na_müüdud = vgs["NA_Sales"].sum()

print("USAs müüdud mänge: ", na_müüdud)

euroopas_müüdud = vgs["EU_Sales"].sum()

print("Euroopas müüdud mänge: ", euroopas_müüdud)

maailmas_müüdud = vgs["Global_Sales"].sum()

print("Maailmas müüdud mänge: ", maailmas_müüdud)

myydud = ["Global_Sales", "NA_Sales", "EU_Sales", "JP_Sales"] #https://stackoverflow.com/questions/27018622/pandas-groupby-sort-descending-order

grupeeritud = vgs.groupby("Publisher")[myydud].sum().reset_index()

järjestatudGRP = grupeeritud.sort_values(myydud, ascending=False)

järjestatudGRP.head(15)
g = sns.FacetGrid(vgs, hue="Global_Sales", size=9)

g.map(plt.scatter, "Year", "Global_Sales", s=75, alpha=0.7)



#Siin on näha scatterploti sellest, kuhu enamus mängud oma müügiarvu poolest paigutuvad. Enamik

#mänge müüb vähem kui miljon, ning ainult mõningad on müünud üle 30 miljoni koopia.
vgs.groupby(['Name'])['Global_Sales'].sum().sort_values(ascending=False)

vgs.head(10)
Platform = pd.crosstab(vgs.Platform,vgs.Publisher)

PlatformTotal = Platform.sum(axis=1).sort_values(ascending = True)

plt.figure(figsize=(20, 10))



indeks = PlatformTotal.index

väärtused = PlatformTotal.values

ax = sns.barplot(indeks, väärtused)

ax.set_xticklabels(labels = indeks, fontsize = 20, rotation = 90)



plt.ylabel = "Platform"

plt.show()
aastad = vgs.groupby(["Year"]).sum()["Global_Sales"]

müügid = aastad.index.astype(int)



plt.figure(figsize = (16,8))

plt.bar(müügid, aastad)

plt.xlabel("Aasta", fontsize = 20)



plt.show();
import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv")

df.head()
cina = df[df["Country/Region"] == "Mainland China"]

cina.head()
cina.ObservationDate = pd.to_datetime(cina.ObservationDate)

cina.index = cina.ObservationDate

cina.head()
cina.tail()
multidfarr = [cina.loc[idx].sum() for idx in cina.index.unique()]
cindf = pd.DataFrame(multidfarr)

cindf.index = cina.index.unique()

cindf["Country/Region"] = "cina"

cindf["Province/State"] = "cina"

cindf["Last Update"] = cindf.index

cindf.head()
fig, ax = plt.subplots()



cindf.Confirmed.plot(ax=ax)

cindf.Deaths.plot(ax=ax)

cindf.Recovered.plot(ax=ax)



ax.legend(["Confermati", "Decessi", "Guariti"]);
df_ita_prov = pd.read_csv("/kaggle/input/covid19-italy-regional-data/dpc-covid19-ita-province.csv")

df_ita_prov.head()
df_ita_reg = pd.read_csv("/kaggle/input/covid19-italy-regional-data/dpc-covid19-ita-regioni.csv")

df_ita_reg.head()
df_ita_reg.index = pd.to_datetime(df_ita_reg.data)

multidfarrita = [df_ita_reg.loc[idx].sum() for idx in df_ita_reg.index.unique()]
itadf = pd.DataFrame(multidfarrita)

itadf.data = df_ita_reg.index.unique()

itadf.index = df_ita_reg.index.unique()

itadf.stato = "italia"

itadf.drop(["codice_regione", "denominazione_regione", "lat", "long"], inplace=True, axis=1)

itadf.head()
fig, ax = plt.subplots()



itadf.totale_attualmente_positivi.plot(ax=ax)

itadf.deceduti.plot(ax=ax)

itadf.dimessi_guariti.plot(ax=ax)



ax.legend(["Confermati", "Decessi", "Guariti"]);
fig, ax = plt.subplots()



itadf.tamponi.plot(ax=ax)

itadf.totale_attualmente_positivi.plot(ax=ax)



ax.legend(["Tamponi", "Confermati"]);
itadf.corr()
sns.heatmap(itadf.corr(), cmap="YlGnBu", square=True)
f"Il {itadf.corr().totale_ospedalizzati.loc['terapia_intensiva'] * 100}% degli ospedalizzati sono in terapia intensiva."
fig, ax = plt.subplots()



cindf.Confirmed.reset_index(drop=True).pct_change().plot(ax=ax)

itadf.totale_attualmente_positivi.reset_index(drop=True).pct_change().plot(ax=ax)



ax.legend(["Cina", "Italia"]);
fig, ax = plt.subplots()



cindf.Deaths.reset_index(drop=True).pct_change().plot(ax=ax)

itadf.deceduti.reset_index(drop=True).pct_change().plot(ax=ax)



ax.legend(["Cina", "Italia"]);
fig, ax = plt.subplots()



cindf.Recovered.reset_index(drop=True).pct_change().plot(ax=ax)

itadf.dimessi_guariti.reset_index(drop=True).pct_change().plot(ax=ax)



ax.legend(["Cina", "Italia"]);
reg = df_ita_reg.copy()

reg.head()
lista_regioni = list(reg.denominazione_regione.unique())
dfreg = pd.concat([reg[reg.denominazione_regione == regidx] for regidx in lista_regioni], keys=lista_regioni, names=["regione", "data"], axis=1)
for parz in reg.denominazione_regione.unique():

    fig, ax = plt.subplots()

    plt.figure()

    dfreg[parz].tamponi.plot(ax=ax)

    dfreg[parz].totale_attualmente_positivi.plot(ax=ax)

    ax.legend([parz, "positivi"]);
plt.plot(np.array([dfreg[regione].totale_attualmente_positivi/dfreg[regione].tamponi for regione in lista_regioni]))
postamp = pd.Series([(dfreg[regione].totale_attualmente_positivi/dfreg[regione].tamponi)[-1] for regione in lista_regioni])

postamp.index = lista_regioni
f"In {postamp.idxmin()} il {postamp.min()*100}% dei tamponi è positivi, in {postamp.idxmax()} il {postamp.max()*100}%"
postamp.sort_values()
fig, ax = plt.subplots()



dfreg.Lombardia.totale_attualmente_positivi.plot(ax=ax)

dfreg.Lombardia.deceduti.plot(ax=ax)

dfreg.Lombardia.dimessi_guariti.plot(ax=ax)



ax.legend(["Confermati", "Decessi", "Guariti"]);
fig, ax = plt.subplots()



dfreg.Veneto.totale_attualmente_positivi.plot(ax=ax)

dfreg.Veneto.deceduti.plot(ax=ax)

dfreg.Veneto.dimessi_guariti.plot(ax=ax)



ax.legend(["Confermati", "Decessi", "Guariti"]);
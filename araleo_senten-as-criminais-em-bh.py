import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv("../input/sentencas.csv")
df.head()
df.set_index("vara", drop=True, inplace=True)

df_varas = df.drop(["6", "11", "t1", "t2", "t3"])
df_totais = pd.DataFrame({

    "Condenações": df_varas["total_con"],

    "Absolvições": df_varas["total_abs"],

    "Neutras": df_varas["total_neu"],

})



fig, axes = plt.subplots(ncols=2, figsize=(18,6))

fig.suptitle("Condenações x Absolvições x Neutras em BH - 2014 a 2019", fontsize=16)

df_totais.sum().sort_values(ascending=False).plot.bar(ax=axes[0], rot=0)

df_totais.sum().plot.pie(ax=axes[1], autopct="%1.0f%%")

axes[1].set_ylabel("");
df_totais_normal = df_totais.div(df_totais.sum(axis=1), axis=0).multiply(100)

ax = df_totais_normal.plot.bar(figsize=(20,6), title="Condenações x Absolvições x Neutras por vara em BH - em % - 2014 a 2019", rot=0)

ax.grid("on", linewidth=.3)

ax.legend(bbox_to_anchor=(1, 1));
df_crimes = df_varas.drop(["total_abs", "total_con", "total_neu"], axis=1)

crimes = set([crime[:-4] for crime in df_crimes.columns])



df_crimes_totais = pd.DataFrame({

    crime: [df_crimes[f"{crime}_con"].sum(), df_crimes[f"{crime}_abs"].sum(), df_crimes[f"{crime}_neu"].sum()]

    for crime in crimes

}, index=["Condenações", "Absolvições", "Neutras"])



df_crimes_totais = df_crimes_totais.transpose()

df_crimes_totais = df_crimes_totais.div(df_crimes_totais.sum(axis=1), axis=0).multiply(100)

ax = df_crimes_totais.plot.bar(figsize=(20,6), rot=0, title="Condenações x Absolvições x Neutras por crime em BH - em % - 2014 a 2019")

ax.grid("on", linewidth=.3)
crimes_count = pd.Series({

    crime: df_crimes.loc[:,df_crimes.columns.str.startswith(crime)].transpose().sum().sum()

    for crime in crimes

})



crimes_count = crimes_count.div(crimes_count.sum()).multiply(100)

ax = crimes_count.sort_values(ascending=False).plot.bar(figsize=(12,6), rot=0, title="Sentenças por crime em BH - em % - 2014 a 2019")

ax.grid("on", linewidth=.3)
def plot_crimes_vara_ax(vara, ax):

    df_crimes = df_varas.drop(["total_abs", "total_con", "total_neu"], axis=1)

    crimes = set([crime[:-4] for crime in df_crimes.columns])

    df_vara = df_crimes.filter(vara, axis=0)

    df_crimes_vara = pd.DataFrame({

        crime: [df_vara[f"{crime}_con"].sum(), df_vara[f"{crime}_abs"].sum(), df_vara[f"{crime}_neu"].sum()]

        for crime in crimes

    }, index=["Condenações", "Absolvições", "Neutras"])                      

    df_crimes_vara = df_crimes_vara.div(df_crimes_vara.sum()).multiply(100)

    _ax = df_crimes_vara.transpose().plot.bar(ax=ax, rot=0, title=vara)

    ax.legend(bbox_to_anchor=(1, 1))

    _ax.grid("on", linewidth=.3)
_varas = ["1", "4", "8", "9"]

fig, axes = plt.subplots(nrows=len(_varas), figsize=(20,13))

for i, vara in enumerate(_varas):

    plot_crimes_vara_ax(vara, axes[i])
def plot_varas_crime_ax(crime, ax):

    df_crime = df_crimes.loc[:, df_crimes.columns.str.startswith(crime)]

    df_crime = df_crime[[f"{crime}_con", f"{crime}_abs", f"{crime}_neu"]]

    df_crime = df_crime.rename({f"{crime}_con": "Condenações", f"{crime}_abs": "Absolvições", f"{crime}_neu": "Neutras"}, axis=1)

    df_crime = df_crime.div(df_crime.sum(axis=1), axis=0).multiply(100)

    ax = df_crime.plot.bar(ax=ax, rot=0, title=crime)

    ax.grid("on", linewidth=.3)

    ax.legend(bbox_to_anchor=(1, 1))

    ax.axes.get_xaxis().get_label().set_visible(False)
_crimes = ["corrupcao", "estelionato", "estupro", "desacato"]

fig, axes = plt.subplots(nrows=len(_crimes), figsize=(20,13))

fig.suptitle("Condenações x Absolvições x Neutras por vara - em % - 2014 a 2019", fontsize=16)

for i, crime in enumerate(_crimes):

    plot_varas_crime_ax(crime, axes[i])
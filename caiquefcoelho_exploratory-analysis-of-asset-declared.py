import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
filename = "/kaggle/input/candidatos/consulta_cand_2020/consulta_cand_2020_BRASIL.csv"

#Import dask dataframe to load large dataset out of memory
import dask.dataframe as dd

candidates = dd.read_csv(filename, sep=";", encoding="ISO-8859-1")
candidates.head()
candidates.columns
filename = "/kaggle/input/candidatos/bem_candidato_2020/bem_candidato_2020_BRASIL.csv"

#Import dask dataframe to load large dataset out of memory
import dask.dataframe as dd

declared_assets = dd.read_csv(filename, sep=";", encoding="ISO-8859-1")
declared_assets.head()
candidates = candidates[["SQ_CANDIDATO", "NM_URNA_CANDIDATO", "SG_PARTIDO"]]
candidates.head()
declared_assets_by_candidate = declared_assets.merge(candidates, on="SQ_CANDIDATO")
declared_assets_by_candidate.head()
# Get only first digit from VR value
declared_assets_by_candidate["VR_BEM_CANDIDATO"] = declared_assets_by_candidate["VR_BEM_CANDIDATO"].apply(lambda x: x.split(",",)[0], meta=('VR_BEM_CANDIDATO', 'object')).astype(float)
declared_assets_by_candidate.head()
declared_assets_by_candidate = declared_assets_by_candidate.groupby("SQ_CANDIDATO").agg({"SG_PARTIDO": "first", "VR_BEM_CANDIDATO": "sum", "SG_UF": "first", "NM_URNA_CANDIDATO": "first"})
declared_assets_by_candidate['SQ_CANDIDATO'] = declared_assets_by_candidate.index
declared_assets_by_candidate.head()
declared_assets_by_candidate.set_index("VR_BEM_CANDIDATO").compute().tail(15)
declared_assets[declared_assets["SQ_CANDIDATO"] == 160001000870].head(15)
agg = declared_assets_by_candidate.groupby("SG_PARTIDO").agg({"VR_BEM_CANDIDATO": "sum"})
VR_BEM_CANDIDATO_SUM = agg.VR_BEM_CANDIDATO.values

VR_BEM_CANDIDATO_MEDIAN = declared_assets_by_candidate.groupby('SG_PARTIDO').VR_BEM_CANDIDATO.apply(pd.Series.median, meta=('x', 'f8'))
VR_BEM_CANDIDATO_MEDIAN = VR_BEM_CANDIDATO_MEDIAN.values

agg['VR_BEM_CANDIDATO_SUM'] = VR_BEM_CANDIDATO_SUM
agg['VR_BEM_CANDIDATO_MEDIAN'] = VR_BEM_CANDIDATO_MEDIAN

agg = agg.drop('VR_BEM_CANDIDATO', axis=1)
agg["SG_PARTIDO"] = agg.index

agg.head()
agg_median = agg
agg_median = agg.drop('VR_BEM_CANDIDATO_SUM', axis=1)
agg_median['VR_BEM_CANDIDATO_MEDIAN_COLUMN'] = agg_median.VR_BEM_CANDIDATO_MEDIAN
agg_median.set_index("VR_BEM_CANDIDATO_MEDIAN").compute().tail(15)
agg_sum = agg
agg_sum = agg.drop('VR_BEM_CANDIDATO_MEDIAN', axis=1)
agg_sum['VR_BEM_CANDIDATO_SUM_COLUMN'] = agg_sum.VR_BEM_CANDIDATO_SUM
agg_sum.set_index("VR_BEM_CANDIDATO_SUM").compute().tail(15)
df_median = agg_median.compute()
df_median.head()
def human_format(num, pos):
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return 'R${}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

fig, ax = plt.subplots(figsize=(25, 14))
sns.barplot(data=df_median, x="SG_PARTIDO", y="VR_BEM_CANDIDATO_MEDIAN_COLUMN", color="blue")
ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))
order = declared_assets_by_candidate.compute().index.array
order
df_declared_assets_by_candidate = declared_assets_by_candidate.compute()
df_declared_assets_by_candidate[df_declared_assets_by_candidate['NM_URNA_CANDIDATO'].str.contains("BOULOS", na=False)].head()
order
# order = df_declared_assets_by_candidate.sort_values(by="VR_BEM_CANDIDATO", ascending=False).index.values
# df_declared_assets_by_candidate.sort_values(by="VR_BEM_CANDIDATO", ascending=False)

fig, ax = plt.subplots(figsize=(25, 14))
sns.barplot(data=df_declared_assets_by_candidate, x="SG_PARTIDO", y="VR_BEM_CANDIDATO", estimator=np.median, color="#c9a8fa", order=order)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))
plt.title("Mediana dos bens declarados por partido para candidatos das eleições 2020", fontsize=32, pad=16)
plt.xlabel("Partido")
plt.ylabel("Valor dos bens declarados")
order = df_declared_assets_by_candidate[df_declared_assets_by_candidate["SG_UF"] == "SP"].groupby("SG_PARTIDO")["VR_BEM_CANDIDATO"].median().sort_values(ascending=False).index.values[:24]

fig, ax = plt.subplots(figsize=(25, 14))
sns.barplot(data=df_declared_assets_by_candidate[df_declared_assets_by_candidate["SG_UF"] == "SP"], x="SG_PARTIDO", y="VR_BEM_CANDIDATO", estimator=np.median, color="#c9a8fa", order=order)
plt.title("Mediana dos bens declarados por partido para candidatos das eleições 2020 em SP", fontsize=32, pad=16)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))
plt.xlabel("Partido")
plt.ylabel("Valor dos bens declarados")
fig, ax = plt.subplots(figsize=(25, 14))
sns.boxplot(data=df_declared_assets_by_candidate, y="VR_BEM_CANDIDATO", x="SG_PARTIDO",
            order=["PSDB", "PT", "PSTU", "NOVO", "PATRIOTA"], color="#c9a8fa",
            whis=1.92)
ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))
plt.ylim([0, 3000000])
plt.title("Distribuição dos bens declarados por partido para candidatos das eleições 2020", fontsize=32, pad=16)
plt.xlabel("Partido")
plt.ylabel("Valor dos bens declarados")
plt.show()
from matplotlib.ticker import PercentFormatter

parties_to_plot = ["NOVO", "PT", "PSDB", "PSTU", "PATRIOTA"]
color_dict = {"NOVO": "orange",
              "PT": "red",
              "PSDB": "blue",
              "PSTU": "green",
              "PATRIOTA": "black"}

max_value = 2000000
n_bins = 70
bins = np.linspace(0, max_value, n_bins)

fig, ax = plt.subplots(len(parties_to_plot), 1, sharex=True,
                       figsize=(20, 14))

for i, party in enumerate(parties_to_plot):
    data = df_declared_assets_by_candidate[df_declared_assets_by_candidate["SG_PARTIDO"] == party]["VR_BEM_CANDIDATO"]
    ax[i].hist(data,
                range = (0, max_value),
                color=color_dict[party],
                #label=party,
                weights=np.ones(len(data)) / len(data),
                alpha=.75,
                bins=bins)
    ax[i].xaxis.set_major_formatter(ticker.FuncFormatter(human_format))    
    ax[i].yaxis.set_major_formatter(PercentFormatter(1))
    ax[i].set_title(party)

fig.suptitle("Distribuição do valor dos bens declarados dos candidatos de {} nas eleições 2020".format(", ".join(parties_to_plot)), 
          fontsize=28)

#plt.xlim([-0, 2000000])
plt.show()

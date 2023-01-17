# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker



plt.style.use('fivethirtyeight')

plt.rcParams['figure.figsize'] = (12, 4)
def human_format(num, pos):

    num = float('{:.3g}'.format(num))

    magnitude = 0

    while abs(num) >= 1000:

        magnitude += 1

        num /= 1000.0

    return 'R${}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])
candidatos = pd.read_csv("/kaggle/input/brazil-elections-2020/candidatos/consulta_cand_2020/consulta_cand_2020_BRASIL.csv", sep=";", engine="python")
candidatos.shape
candidatos.columns
candidatos["SQ_CANDIDATO"].nunique()
candidatos.head()
assets = pd.read_csv("/kaggle/input/brazil-elections-2020/candidatos/bem_candidato_2020/bem_candidato_2020_BRASIL.csv", sep=";", engine="python")
assets.shape
assets.head()
assets["SQ_CANDIDATO"].nunique()
candidatos = candidatos[["SQ_CANDIDATO", "NM_URNA_CANDIDATO", "SG_PARTIDO"]]
assets_by_party = assets.merge(candidatos, on="SQ_CANDIDATO")
assets_by_party["VR_BEM_CANDIDATO"] = assets_by_party["VR_BEM_CANDIDATO"].apply(lambda x: x.split(",")[0]).astype(float)
assets_by_party = assets_by_party.groupby("SQ_CANDIDATO").agg({"SG_PARTIDO": "first", "VR_BEM_CANDIDATO": "sum", "SG_UF": "first", "NM_URNA_CANDIDATO": "first"})
assets_by_party.sort_values(by="VR_BEM_CANDIDATO", ascending=False)[:15]
assets[assets["SQ_CANDIDATO"] == 190001019131]
agg = assets_by_party.groupby("SG_PARTIDO")["VR_BEM_CANDIDATO"].median()
order = agg.sort_values(ascending=False).index.values

agg.sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(25, 14))

sns.barplot(data=assets_by_party, x="SG_PARTIDO", y="VR_BEM_CANDIDATO", color="blue")

ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))
fig, ax = plt.subplots(figsize=(25, 14))

sns.barplot(data=assets_by_party, x="SG_PARTIDO", y="VR_BEM_CANDIDATO", estimator=np.median, color="#c9a8fa", order=order)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))

plt.title("Mediana dos bens declarados por partido para candidatos das eleições 2020", fontsize=32, pad=16)

plt.xlabel("Partido")

plt.ylabel("Valor dos bens declarados")
order = assets_by_party[assets_by_party["SG_UF"] == "SP"].groupby("SG_PARTIDO")["VR_BEM_CANDIDATO"].median().sort_values(ascending=False).index.values[:24]



fig, ax = plt.subplots(figsize=(25, 14))

sns.barplot(data=assets_by_party[assets_by_party["SG_UF"] == "SP"], x="SG_PARTIDO", y="VR_BEM_CANDIDATO", estimator=np.median, color="#c9a8fa", order=order)

plt.title("Mediana dos bens declarados por partido para candidatos das eleições 2020 em SP", fontsize=32, pad=16)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))

plt.xlabel("Partido")

plt.ylabel("Valor dos bens declarados")
average_wo_outliers = assets_by_party.groupby("SG_PARTIDO")["VR_BEM_CANDIDATO"].apply(lambda x: np.mean(x[x < np.quantile(x, 0.98)]))
average_wo_outliers.sort_values(ascending=False)
outliers = assets_by_party.groupby("SG_PARTIDO", as_index=False)["VR_BEM_CANDIDATO"].apply(lambda x: np.quantile(x, 0.995)).rename(columns={"VR_BEM_CANDIDATO": "upper_limit"})
assets_by_party = assets_by_party.merge(outliers, on="SG_PARTIDO")

assets_by_party = assets_by_party[assets_by_party["VR_BEM_CANDIDATO"] < assets_by_party["upper_limit"]]
fig, ax = plt.subplots(figsize=(25, 14))

sns.boxplot(data=assets_by_party, y="VR_BEM_CANDIDATO", x="SG_PARTIDO",

            order=["PSDB", "PT", "PSTU", "NOVO"], color="#c9a8fa",

            whis=1.92)

ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))

plt.ylim([0, 3000000])

plt.title("Distribuição dos bens declarados por partido para candidatos das eleições 2020", fontsize=32, pad=16)

plt.xlabel("Partido")

plt.ylabel("Valor dos bens declarados")

plt.show()
fig, ax = plt.subplots(figsize=(25, 14))

sns.violinplot(data=assets_by_party, y="VR_BEM_CANDIDATO", x="SG_PARTIDO",

               order=["PSDB", "PT", "PSTU", "NOVO"], color="#c9a8fa",

               cut=True)

plt.ylim([-1500000, 5000000])

ax.yaxis.set_major_formatter(ticker.FuncFormatter(human_format))

plt.title("Distribuição dos bens declarados por partido para candidatos das eleições 2020", fontsize=32, pad=16)

plt.xlabel("Partido")

plt.ylabel("Valor dos bens declarados")
from matplotlib.ticker import PercentFormatter



parties_to_plot = ["NOVO", "PT", "PSDB", "PSTU"]

color_dict = {"NOVO": "orange",

              "PT": "red",

              "PSDB": "blue",

              "PSTU": "green"}



max_value = 2000000

n_bins = 70

bins = np.linspace(0, max_value, n_bins)



fig, ax = plt.subplots(len(parties_to_plot), 1, sharex=True,

                       figsize=(20, 14))



for i, party in enumerate(parties_to_plot):

    data = assets_by_party[assets_by_party["SG_PARTIDO"] == party]["VR_BEM_CANDIDATO"]

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
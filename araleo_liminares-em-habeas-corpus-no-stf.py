import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns





df = pd.read_csv("../input/braziliansc-habeascorpusjan1518/BrazilianSC_HC.csv", sep=";")
df.head()
df.drop("Class", axis=1, inplace=True)
print("data mínima: ", df.Date.min(), "data máxima: ", df.Date.max())
df.Decision.unique()
neutras = (

    "Decisão (segredo de justiça)",

    "Prejudicado",

    "Homologada a desistência",

    "Decisão liminar (segredo de justiça)",

    "Determinado arquivamento",

    "Reconsideração",

    "Extinto o processo",

    "À Secretaria, para o regular trâmite",

    "Convertido em diligência",

    "Embargos recebidos",

    "Liminar prejudicada",

    "Declinada a competência",

    "Questão de ordem",

    "Determinada a devolução",

    "Reconsidero e julgo prejudicado o recurso interno"

)

desfavoraveis = (

    "Indeferido",

    "Negado seguimento",

    "Não conhecido(s)",

    "Liminar indeferida",

    "Não provido",

    "Denegada a ordem",

    "Embargos rejeitados",

    "Agravo regimental não conhecido",

    "Improcedente",

    "Conhecido em parte e nessa parte negado provimento",

    "Agravo regimental não provido",

    "Inadmitidos os embargos de divergência",

    "Embargos recebidos como agravo regimental desde logo não provido",

    "Embargos não conhecidos"

)

favoraveis = (

    "Liminar deferida",

    "Deferido",

    "Concedida a ordem",

    "Concedida a ordem de ofício",

    "Concedida em parte a ordem",

    "Liminar deferida em parte",

    "Deferido em parte",

    "Conhecido e provido",

    "Provido",

    "Agravo regimental provido",

    "Declarada a extinção da punibilidade",

    "Conhecido em parte e nessa parte provido"

)
df_finais = df[df["Judgment type"] != "Decisão Interlocutória"]

num_des = df_finais.Decision.isin(desfavoraveis).sum()

num_fav = df_finais.Decision.isin(favoraveis).sum()

num_neu = df_finais.Decision.isin(neutras).sum()

series_dist = pd.Series([num_des, num_fav, num_neu], index=["desfavoráveis", "favoráveis", "neutras"])



fig, axes = plt.subplots(ncols=2, figsize=(18,6))

fig.suptitle("Distribuição de decisões em habeas corpus no STF - jan/15 a jan/18", fontsize=16)

series_dist.plot.bar(ax=axes[0], rot=0)

series_dist.plot.pie(ax=axes[1], autopct="%1.0f%%")

axes[1].set_ylabel("");
df_favoraveis = df_finais[df_finais.Decision.isin(favoraveis)]

serie_favoraveis = df_favoraveis.Decision

serie_favoraveis.value_counts().plot.bar(figsize=(12,6), grid=True, rot=30, title="Decisões favoráveis em habeas corpus no STF por tipo - jan/15 a jan/18");
_df = pd.Series({

    "Mérito": (df_favoraveis["Judgment type"] == "Decisão Final").sum(),

    "Liminar": (df_favoraveis["Judgment type"] == "Decisão Liminar").sum()

})



fig, axes = plt.subplots(ncols=2, figsize=(14,6))

fig.suptitle("Decisões favoráveis em HCs no STF - mérito x liminares - jan/15 a jan/18", fontsize=16)

_df.plot.bar(ax=axes[0], rot=0)

_df.plot.pie(ax=axes[1], autopct="%1.0f%%")

axes[1].set_ylabel("");

alvos = ["Liminar deferida", "Liminar indeferida"]

df_liminares = df[df["Judgment type"] == "Decisão Liminar"]

df_liminares = df_liminares[df_liminares["Decision"].isin(alvos)]

fig, axes = plt.subplots(ncols=2, figsize=(14,6))

fig.suptitle("Liminares deferidas x indeferidas no STF - em % - jan/15 a jan/18", fontsize=16)

df_liminares.Decision.value_counts().plot.bar(ax=axes[0], rot=0)

df_liminares.Decision.value_counts().plot.pie(ax=axes[1], autopct="%1.0f%%")

axes[1].set_ylabel("");
alvos = ["Liminar deferida", "Concedida a ordem de ofício"]

df_lim_ofi = df[df["Decision"].isin(alvos)]

col_count = df_lim_ofi[df_lim_ofi["Decision type"] == "COLEGIADA"].Decision.value_counts()

mon_count = df_lim_ofi[df_lim_ofi["Decision type"] == "MONOCRÁTICA"].Decision.value_counts()

_df = pd.DataFrame({"Monocráticas": mon_count, "Colegiadas": col_count}, index=alvos)

_df.plot.bar(figsize=(8,8), rot=0, title="Decisões monocráticas x colegiadas em concessões de liminares em hc e de hc de ofício no STF - jan/15 a jan/18");
dic_ministros = {}

for ministro in list(df.Justice.unique()):

   dic_ministros[ministro] = df_liminares[df_liminares.Justice == ministro]
cols, rows = 4, 3

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(18,12))



def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:d}\n{:.1f}%".format(absolute, pct)



for i, ministro in enumerate(list(df.Justice.unique())):

    a, b = divmod(i, cols)

    df_ministro = dic_ministros[ministro]

    counts = df_ministro.Decision.value_counts()

    df_ministro.Decision.value_counts().plot.pie(ax=axes[a,b], title=ministro, labels=None, autopct=lambda pct: func(pct, df_ministro.Decision.value_counts()))

    axes[a,b].set_ylabel("");



fig.suptitle("Liminares deferidas x indeferidas por ministro - jan/15 a jan/18", fontsize=16)

axes[0,3].legend(df_ministro.Decision.unique(), bbox_to_anchor=(1, 1));
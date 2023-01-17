import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv("../input/stfhcsconcedidos.csv")
df.head()
df_datas = df.copy(deep=True)

df_datas.loc[:, "Data Andamento"] = pd.to_datetime(df_datas["Data Andamento"], dayfirst=True)
dt_inicio = df_datas["Data Andamento"].min()

dt_fim = df_datas["Data Andamento"].max()



print(f"inicio: {dt_inicio}\nfim: {dt_fim}")
total_dias = (dt_fim - dt_inicio).days

total_hcs = len(df)

media = total_hcs / total_dias

print(f"""Entre 03 de fevereiro de 2009 e 22 de julho de 2020, um período de {total_dias} dias,

o STF concedeu {total_hcs} habeas corpus, para uma média de {media:.2f} habeas corpus concedidos por dia.""") 
def double_plot(title, df1, df2):

    fig, axes = plt.subplots(ncols=2, figsize=(20,8))

    fig.suptitle(title, fontsize=16)

    df1.plot.bar(ax=axes[0], rot=10)

    df2.plot.pie(ax=axes[1], autopct="%1.2f%%")

    axes[1].set_ylabel("")
double_plot(

    "Distribuição de HCs concedidos no STF - 2009 a 2020",

    df.Andamento.value_counts().sort_values(ascending=False),

    df.Andamento.value_counts()

)
df_colegiado = df[df["Tipo Decisão"] == "COLEGIADA"]

df_monocratico = df[df["Tipo Decisão"] == "MONOCRÁTICA"]
double_plot(

    "Distribuição de HCs concedidos no STF - Colegiados - 2009 a 2020",

    df_colegiado.Andamento.value_counts().sort_values(ascending=False),

    df_colegiado.Andamento.value_counts()

)
double_plot(

    "Distribuição de HCs concedidos no STF - Monocráticos - 2009 a 2020",

    df_monocratico.Andamento.value_counts().sort_values(ascending=False),

    df_monocratico.Andamento.value_counts()

)
df_colegiado["Orgão Julgador"].value_counts().sort_values(ascending=False).plot.bar(figsize=(8,6), rot=15, title="HCs concedidos no STF - Colegiados - 2009 a 2020");
df_monocratico["Orgão Julgador"].value_counts().sort_values(ascending=False).plot.bar(figsize=(20,8), rot=30, title="HCs concedidos monocraticamente no STF por Ministro - 2009 a 2020", grid=True);
df_datas.groupby(df_datas["Data Andamento"].dt.year).sum()["Qtd Processos"].plot(figsize=(12,8), title="HCs concedidos por ano no STF - Total", grid=True);
df_dts_col = df_datas[df_datas["Tipo Decisão"] == "COLEGIADA"]

df_dts_mon = df_datas[df_datas["Tipo Decisão"] == "MONOCRÁTICA"]
fig, axes = plt.subplots(ncols=3, figsize=(24,8), sharey=True)

fig.suptitle("HCs concedidos no STF - 2009 a 2020", fontsize=16)

df_dts_col.groupby(df_dts_col["Data Andamento"].dt.year).sum()["Qtd Processos"].plot(ax=axes[0], title="Colegiados");

df_dts_mon.groupby(df_dts_mon["Data Andamento"].dt.year).sum()["Qtd Processos"].plot(ax=axes[1], color="C1", title="Monocráticos");

df_dts_col.groupby(df_dts_col["Data Andamento"].dt.year).sum()["Qtd Processos"].plot(ax=axes[2], title="Colegiados x Monocráticos");

df_dts_mon.groupby(df_dts_mon["Data Andamento"].dt.year).sum()["Qtd Processos"].plot(ax=axes[2]);
ministros_atuais = [

    "RICARDO LEWANDOWSKI",

    "CELSO DE MELLO",

    "CÁRMEN LÚCIA",

    "DIAS TOFFOLI",

    "GILMAR MENDES",

    "ROSA WEBER",

    "LUIZ FUX",

    "ALEXANDRE DE MORAES",

    "ROBERTO BARROSO",

    "EDSON FACHIN",

    "MARCO AURÉLIO",

]

ministros_atuais.sort()

ministros_atuais.append("PRESIDÊNCIA")
def list_all_years(df_ministro):

    anos = tuple(range(2009, 2021))

    vals = [0] * len(anos)

    serie = pd.Series(vals, index=anos)

    for ano in anos:

        if ano in df_ministro.index:

            serie[ano] += df_ministro[ano]

    return serie
def hcs_ministros_ano(title, bar=False):

    rows = 3

    cols = 4

    fig, axes = plt.subplots(nrows=rows, ncols=cols, sharey=True, figsize=(20,15))

    fig.suptitle(title, fontsize=14)



    for i, ministro in enumerate(ministros_atuais):

        a, b = divmod(i, cols)

        df_ministro = df_dts_mon[df_dts_mon["Orgão Julgador"] == ministro]

        df_ministro = df_ministro.groupby(df_ministro["Data Andamento"].dt.year).sum()["Qtd Processos"]

        serie_anos = list_all_years(df_ministro)

        if bar:

            ax = serie_anos.plot.bar(ax=axes[a,b], title=ministro, sharey=True, rot=30)

        else:

            ax = serie_anos.plot(ax=axes[a,b], title=ministro)

        ax.axes.get_xaxis().get_label().set_visible(False)
hcs_ministros_ano("HCs concedidos monocraticamente por ano por ministro")
hcs_ministros_ano("HCs concedidos monocraticamente por ano por ministro", bar=True)
df_2019 = df_datas[df_datas["Data Andamento"].dt.year == 2019]

total_hcs_2019 = len(df_2019)

media_2019 = total_hcs_2019 / 365

print(f"Em 2019 o STF concedeu {total_hcs_2019} HCs, para uma média de {media_2019:.2f} habeas corpus concedidos por dia.")
df_2019.groupby("Orgão Julgador").sum()["Qtd Processos"].sort_values(ascending=False).plot.bar(figsize=(20,6), rot=30, title="Total de Habeas Corpus concedidos no STF em 2019 por Órgão Julgador", grid=True);
def func(pct, allvals):

    absolute = int(pct/100.*np.sum(allvals))

    return "{:d}\n{:.1f}%".format(absolute, pct)



df_2019_colegiado = df_2019[df_2019["Tipo Decisão"] == "COLEGIADA"]

_df = df_2019_colegiado.groupby("Orgão Julgador").sum()["Qtd Processos"]

ax = _df.plot.pie(autopct=lambda pct: func(pct, _df), figsize=(8,8), title="HCs concecidos pelos Colegiados no STF - 2019")

ax.set_ylabel("");
df_dias_mes = df_dts_mon.groupby(df_dts_mon["Data Andamento"].dt.day).sum()["Qtd Processos"]

ax = df_dias_mes.plot.bar(figsize=(16,8), rot=0, title="Quantidade de HCs concedidos em cada dia do mês - 2009 a 2020")

ax.axes.get_xaxis().get_label().set_visible(False);
df_dias_sem = df_dts_mon.groupby(df_dts_mon["Data Andamento"].dt.weekday).sum()["Qtd Processos"]

ax = df_dias_sem.plot.bar(figsize=(14,7), rot=0, title="Quantidade de HCs concedidos em cada dia da semana - 2009 a 2020");

ax.set_xticklabels(["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"])

ax.axes.get_xaxis().get_label().set_visible(False);
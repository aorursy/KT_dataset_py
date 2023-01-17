import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv("../input/military-expenditure-of-countries-19602019/Military Expenditure.csv")
df.head()
df_br = df[df.Name == "Brazil"]

df_br = df_br.drop(["Code", "Type", "Indicator Name"], axis=1)

df_br = df_br.set_index("Name")

df_br
df_br.isnull().sum().sum()
df_br.columns.unique()
len(df_br.columns.unique()) == len(df_br.columns)
df_br = df_br.transpose()
df_br.div(10**9).plot(figsize=(12,12), legend=False, title="Brazilian military spending in billions of US $");
decades = ["196", "197", "198", "199", "200", "201"]

rows = 2

cols = 3

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(20,8), sharey=True)

fig.suptitle("Braziian military budget by year in billions of US$", fontsize=16)



for i, decade in enumerate(decades):

    a, b = divmod(i, cols)

    df_decade = df_br[df_br.transpose().columns.str.startswith(decade)]

    df_decade.div(10**9).plot.bar(ax=axes[a,b], legend=False, rot=30)
_df = df_br.transpose()

df_pre_2k = _df.loc[:, _df.columns.str.startswith("19")].transpose()

df_pos_2k = _df.loc[:, _df.columns.str.startswith("20")].transpose()
df_pre_2k.div(10**9).plot(figsize=(12,8), legend=False, title="Brazilian military spending in billions of US$");
_years = tuple([str(y) for y in list(range(1984, 2000))])

df_80s = df_pre_2k.transpose()

df_80s = df_80s.loc[:, df_80s.columns.str.startswith(_years)].transpose().div(10)

df_80s.div(10**9).plot.bar(figsize=(10,8), legend=False, rot=0, title="Military spending in Brazil in billions of US$");
pct_serie = df_80s.transpose().iloc[0].pct_change()

pct_serie[1:].multiply(100).plot.bar(figsize=(10,8), rot=0, title="Brazil's military spending change by year in percentage", color=(pct_serie[1:] > 0).map({True: 'C0', False: 'C1'}));
budgets_comp = df_80s.transpose()[["1984", "1987"]].iloc[0].pct_change()[1]

budgets_comp
df_pos_2k.div(10**9).plot(figsize=(12,8), legend=False, title="Brazilian military spending in billions of US$");
_years = tuple([str(y) for y in list(range(2000, 2019))])

df_2ks = df_pos_2k.transpose()

df_2ks = df_2ks.loc[:, df_2ks.columns.str.startswith(_years)].transpose().div(10**9)

df_2ks.plot.bar(figsize=(10,8), legend=False, rot=0, title="Military spending in Brazil in billions of US$");
pct_serie_lula = df_2ks.transpose().iloc[0].pct_change()

pct_serie_lula[1:].multiply(100).plot.bar(figsize=(10,8), rot=0, title="Brazil's military spending change by year in percentage", color=(pct_serie_lula[1:] > 0).map({True: 'C0', False: 'C1'}));
lula_comp = df_2ks.transpose()[["2002", "2011"]].iloc[0].pct_change()[1]

lula_comp
df_br.transpose()["2018"].div(10**9).sum()
military_2018 = df_br.transpose()["2018"].div(10**9).multiply(3.3).sum()

military_2018
health_2018 = 119.3 

educat_2018 = 89

series_2018 = pd.Series({

    "health": health_2018,

    "education": educat_2018,

    "military": military_2018

})



fig, axes = plt.subplots(ncols=2, figsize=(18,6))

fig.suptitle("Brazilian military x education x health budget - 2018", fontsize=16)

series_2018.sort_values(ascending=False).plot.bar(ax=axes[0], rot=0)

series_2018.plot.pie(ax=axes[1], autopct="%1.2f%%")

axes[1].set_ylabel("");
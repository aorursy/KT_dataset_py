# importing necessary modules

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# read data

df = pd.read_csv("../input/nfl_draft.csv")



# inspect data

print(df.shape)

df.head()
years = np.arange(1985, 2016, 1)

g = sns.factorplot(x = "Year", data = df, kind = "count",

                   palette = "YlGn", size = 6, aspect = 1.5, order = years)

g.set_xticklabels(step = 5)
n_rounds = np.zeros((len(years), ))

for i, year in enumerate(years) :

    n_rounds[i] = df["Rnd"][df["Year"] == year].unique().shape[0]

rounds_per_year = pd.DataFrame({"years" : years, "n_rounds" : n_rounds})

g = sns.factorplot(x = "years", y = "n_rounds", data = rounds_per_year, kind = "bar",

                   palette = "YlGn", size = 6, aspect = 1.5, order = years)

g.set_xticklabels(step = 5)
df[df["Year"] == 1993][df["Pick"] == 40]
print(df[df["Year"] == 1993]["Tkl"].fillna(-1).value_counts())

print(df[df["Year"] == 1993]["Def_Int"].fillna(-1).value_counts())

print(df[df["Year"] == 1993]["Sk"].fillna(-1).value_counts())
# analyze distributions of Age and First4AV and their correlations

sns.jointplot(x = "Age", y = "First4AV", data = df, size = 5)
# analyze First4AV by Age

sns.boxplot(x = "Age", y = "First4AV", data = df)
# analyze distributions of Rnd and First4AV and their correlations

sns.jointplot(x = "Rnd", y = "First4AV", data = df, size = 5)
# analyze First4AV by Rnd

sns.boxplot(x = "Rnd", y = "First4AV", data = df)
# violin plot of First4Av by Rnd

sns.violinplot(x = "Rnd", y = "First4AV", data = df, size = 6)
# analyze PB by Rnd

sns.boxplot(x = "PB", y = "First4AV", data = df)
df.shape
df.columns.values
df_before_2000 = df[df["Year"] < 2000]

df_before_2000["Years_Played"] = df_before_2000["To"] - df_before_2000["Year"]
# analyze distributions of Rnd and Years_Played and their correlations

sns.jointplot(x = "Rnd", y = "Years_Played", data = df_before_2000, size = 5)
# analyze Years_Played by Rnd

sns.boxplot(x = "Rnd", y = "Years_Played", data = df_before_2000)
# violin plot of Years_Played by Rnd

sns.violinplot(x = "Rnd", y = "Years_Played", data = df_before_2000, size = 6)
df_rb = df[df["Pos"] == "RB"]
df_rb.head()
df_rb["Rush_Att_G"] = df_rb["Rush_Att"] / df_rb["G"]

df_rb["Rush_Yds_G"] = df_rb["Rush_Yds"] / df_rb["G"]

df_rb["Rush_TDs_G"] = df_rb["Rush_TDs"] / df_rb["G"]

df_rb["Rec_G"] = df_rb["Rec"] / df_rb["G"]

df_rb["Rec_Yds_G"] = df_rb["Rec_Yds"] / df_rb["G"]

df_rb["Rec_Tds_G"] = df_rb["Rec_Tds"] / df_rb["G"]
# total purpose yards per game

df_rb["TPY_G"] = df_rb["Rush_Yds_G"] + df_rb["Rec_Yds_G"]
# analyze distributions of Rnd and Rush_Yds_G and their correlations

sns.jointplot(x = "Rnd", y = "TPY_G", data = df_rb, size = 5)
# analyze TPY_G by Rnd

sns.boxplot(x = "Rnd", y = "TPY_G", data = df_rb)
# total purpose TDs per game

df_rb["TPTD_G"] = df_rb["Rush_TDs_G"] + df_rb["Rec_Tds_G"]
# analyze TPTD_G by Rnd

sns.boxplot(x = "Rnd", y = "TPTD_G", data = df_rb)
df_qb = df[df["Pos"] == "QB"]
# analyze G by Rnd

sns.boxplot(x = "Rnd", y = "G", data = df_qb)
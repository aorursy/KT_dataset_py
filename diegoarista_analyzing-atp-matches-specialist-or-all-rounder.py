import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

import seaborn as sns

from scipy import stats

from statsmodels.stats.anova import AnovaRM





atp = pd.read_csv("../input/atp-matches/atp.csv", delimiter = ";", parse_dates= True)
print(atp.head())
def cleaning (variable, groupc, groupr):

    variable2= variable.groupby([groupc, groupr]).count().unstack(level=1)

    variable3= variable2.iloc[:,0:4]

    variable4= variable3.droplevel(level=0, axis=1)

    variable5 = variable4.fillna(0) 

    return variable5
df_winners = cleaning(atp, "winner", "surface")





print(df_winners.head())
df_losers =  cleaning(atp, "loser", "surface")



print(df_losers.head())

    

df_winners.columns = ['carpet_w','clay_w','grass_w',

                     'hard_w']



df_losers.columns = ['carpet_l','clay_l', 'grass_l',

                     'hard_l']

df_full = pd.merge(df_winners, df_losers, left_index=True, right_index=True, how="left")  



print(df_full.head())
df_full["wins"] = df_full.iloc [:, 0:4].sum(axis=1)

df_full["loses"] = df_full.iloc [:, 4:8].sum(axis=1)

df_full["matches"] = df_full.iloc[:, 0:8].sum(axis=1)



print(df_full.head())
df_full.loc["Diez S.":"Dodig I."]
df_plusten = df_full[df_full["wins"] >10]



print(df_plusten.shape)
sns.distplot(df_plusten)
plt.hist(df_plusten["carpet_w"], alpha = 0.4, color="grey")

plt.hist(df_plusten["clay_w"], alpha = 0.8, color="orange")

plt.hist(df_plusten["grass_w"], alpha = 0.7, color ="green")

plt.hist(df_plusten["hard_w"], alpha = 0.4, color="blue")



plt.show()
winsbycourt = df_plusten.loc[:,"carpet_w":"hard_w"]

print(winsbycourt.describe())
print(winsbycourt.median())
wins3surfaces = winsbycourt.drop(columns=["carpet_w"])



print(wins3surfaces.skew(axis = 0, skipna = True))
print(stats.kstest(wins3surfaces["clay_w"], "norm"))

print(stats.kstest(wins3surfaces["grass_w"], "norm"))

print(stats.kstest(wins3surfaces["hard_w"], "norm"))
wins3surfaces = winsbycourt.loc[:,"clay_w":"hard_w"]



sns.heatmap(wins3surfaces.corr("spearman"),

            vmin=-1,cmap='coolwarm',

            annot=True);
 
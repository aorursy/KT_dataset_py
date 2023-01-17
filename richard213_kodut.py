import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 20)

df = pd.read_csv("../input/ign.csv")



df

#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



#Kõikide žanrite skoorid



df2 = df[df["release_year"] > 1990] 

#ühe mängu väljalaskeaastaks on 1970, kuigi ülejäänud andmed on alates 1996. See on ilmselt viga, seega

#eemaldame selle mängu andmestikust

df2.score.plot.hist(bins=10, grid=True, rwidth=0.7)
df2.groupby("genre").aggregate({"score": ["min", "max", "mean", "count"]})
df2.plot.scatter("release_year", "score", alpha = "0.02")
#Shooter ja RPG žanri mängude reitingud läbi aastate



df_fight = df[(df.genre == "Shooter") | ("Shooter" in df.genre)]

df_fight.plot.scatter("release_year", "score", alpha=0.2)



df_genre = df[(df.genre == "RPG") | ("RPG" in df.genre)]

df_genre.plot.scatter("release_year", "score", alpha=0.2)
#PC ja mobiilimängude reitingud läbi aastate



df_pc = df[df.platform == "PC"]

df_mob = df[(df.platform == "Android") | (df.platform == "iPhone") | (df.platform == "iPad")]

df_mob.plot.scatter("release_year", "score", alpha = 0.1)

df_pc.plot.scatter("release_year", "score", alpha=0.1)
#Mis kuus lastakse välja enim mänge

df2.release_month.plot.hist(bins=12, grid=True, rwidth=0.5)
df2.release_day.plot.hist(bins=31, grid=True, rwidth=0.5)
df_ps = df[(df.platform == "PlayStation 1") | (df.platform == "PlayStation 2") | (df.platform == "PlayStation 3")| (df.platform == "PlayStation 3")]

df_ps.release_month.plot.hist(bins=12, grid=True, rwidth=0.5)

df_x = df[(df.platform == "Xbox") | (df.platform == "Xbox 360") | (df.platform == "Xbox One")]

df_x.release_month.plot.hist(bins=24, grid=True, rwidth=0.5)
df_san = df[(df.title == "Grand Theft Auto: San Andreas")]

df_san.groupby("platform").aggregate({"score": ["min"]})
df_ac = df[(df.title == "Assassin's Creed III")]

df_ac.groupby("platform").aggregate({"score": ["min"]})
df_fifa = df[(df.title == "FIFA Soccer 13")]

df_fifa.groupby("platform").aggregate({"score": ["min"]})

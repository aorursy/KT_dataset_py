import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # data visualization

import seaborn as sns

sns.set()



tr = pd.read_csv("../input/TableOfTurkey.csv") ## https://www.kaggle.com/egebozoglu/terrorism-in-turkey-19962017

tr.head()
tr.describe(include="all")
istkill = tr[tr["city"] == "Istanbul"]["Killed"].sum()

istwound = tr[tr["city"] == "Istanbul"]["Wounded"].sum()



ankkill = tr[tr["city"] == "Ankara"]["Killed"].sum()

ankwound = tr[tr["city"] == "Ankara"]["Wounded"].sum()



izmkill = tr[tr["city"] == "Izmir"]["Killed"].sum()

izmwound = tr[tr["city"] == "Izmir"]["Wounded"].sum()



westkill = istkill + izmkill + ankkill

westwound = istwound + izmwound + ankwound

westintwound = int(westwound)
cizkill = (tr[tr["city"] == "Cizre"]["Killed"].sum()) + (tr[tr["city"] == "Cizre district"]["Killed"].sum())

cizwound = (tr[tr["city"] == "Cizre"]["Wounded"].sum()) + (tr[tr["city"] == "Cizre district"]["Wounded"].sum())



diykill = tr[tr["city"] == "Diyarbakir"]["Killed"].sum()

diywound = tr[tr["city"] == "Diyarbakir"]["Wounded"].sum()



sirkill = tr[tr["city"] == "Sirnak"]["Killed"].sum()

sirkwound = tr[tr["city"] == "Sirnak"]["Wounded"].sum()



yukskill = (tr[tr["city"] == "Yuksekova"]["Killed"].sum()) + (tr[tr["city"] == "Yuksekova district"]["Killed"].sum())

yukswound = (tr[tr["city"] == "Yuksekova"]["Wounded"].sum()) + (tr[tr["city"] == "Yuksekova district"]["Wounded"].sum())



cukkill = tr[tr["city"] == "Cukurca"]["Killed"].sum()

cukwound = tr[tr["city"] == "Cukurca"]["Wounded"].sum()



vankill = tr[tr["city"] == "Van"]["Killed"].sum()

vanwound = tr[tr["city"] == "Van"]["Wounded"].sum()



bingkill = tr[tr["city"] == "Bingol"]["Killed"].sum()

bingwound = tr[tr["city"] == "Bingol"]["Wounded"].sum()



semdkill = (tr[tr["city"] == "Semdinli"]["Killed"].sum()) + (tr[tr["city"] == "Semdinli district"]["Killed"].sum())

semdwound = (tr[tr["city"] == "Semdinli"]["Wounded"].sum()) + (tr[tr["city"] == "Semdinli district"]["Wounded"].sum())



eastkill = semdkill + bingkill + vankill + cizkill + cukkill + diykill + sirkill + yukskill

eastwound = semdwound + bingwound + vanwound + cizwound + cukwound + diywound + sirkwound + yukswound

eastintwound = int(eastwound)
n_groups = 2

y = np.array([eastkill,westkill])

z = np.array([eastwound,westwound])
fig, ax = plt.subplots(figsize=(10,6))

index = np.arange(n_groups)

bar_widht = 0.2

opacity = 0.8



killed = plt.bar(index, y, bar_widht, alpha=opacity, color = "r",label="KILLED")

wounded = plt.bar(index+bar_widht, z, bar_widht, alpha=opacity, color = "g", label="WOUNDED")



plt.xlabel("REGION",size=18)

plt.ylabel("NUMBER",size=18)

plt.title("TERRORISM IN TURKEY (East-West)",size=25)

plt.xticks(index + bar_widht, (("EAST, Killed:", eastkill, "Wounded:", eastintwound),("WEST, Killed:", westkill, "Wounded:",westintwound)))

plt.legend()



plt.show()
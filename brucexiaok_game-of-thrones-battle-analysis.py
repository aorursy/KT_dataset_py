import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns

# Input data files are available in the "../input/" directory.
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Set parameters on plt
plt.rcParams["axes.labelsize"] = 16.
plt.rcParams["xtick.labelsize"] = 13.
plt.rcParams["ytick.labelsize"] = 13.
plt.rcParams["legend.fontsize"] = 11.

# Read in files
battles = pd.read_csv("../input/battles.csv")
chardeaths = pd.read_csv("../input/character-deaths.csv")
charpreds = pd.read_csv("../input/character-predictions.csv")
battles.loc[:, "numdefenders"] = (4 - battles[["defender_1", "defender_2", "defender_3", "defender_4"]].isnull().sum(axis = 1))
battles.loc[:, "numattackers"] = (4 - battles[["attacker_1", "attacker_2", "attacker_3", "attacker_4"]].isnull().sum(axis = 1))

housesinvolved = battles.groupby('battle_number').sum()[["numdefenders", "numattackers"]].plot(kind = 'bar', figsize = (15,6), rot = 0)
_ = housesinvolved.set_xlabel("Battle Number"), housesinvolved.set_ylabel("Number of Houses Involved"), housesinvolved.legend(["Defenders", "Attackers"])

deathcapture = battles.groupby('battle_number').sum()[["major_death", "major_capture"]].plot(kind = 'bar', figsize = (15, 6), rot = 0)
_ = deathcapture.set_xlabel("Battle Number"), deathcapture.set_ylabel("Number of Major Deaths of Captures"), deathcapture.legend(["Major Deaths", "Major Captures"])
d1 = battles.dropna(axis = 0, subset = [["attacker_size", "defender_size", "attacker_outcome"]]).copy(deep = True)
col = [sns.color_palette()[1] if x == "win" else "lightgray" for x in d1.attacker_outcome.values]
plot1 = d1.plot(kind = "scatter", x = "attacker_size", y = "defender_size", c = col, figsize = (15, 6), s = 100, lw = 2.)
_ = plot1.set_xlabel("Attacking Size"), plot1.set_ylabel("Defending Size"), plot1.legend("win")
d2 = battles.dropna(axis=0, subset=[["attacker_size", "defender_size", "major_death"]]).copy(deep=True)
col = [sns.color_palette()[0] if x == 1 else sns.color_palette()[4] for x in d2.major_death.values]
plot2 = d2.plot(kind="scatter", x = "attacker_size", y = "defender_size", c = col, figsize=(15, 6), s = 100, lw = 2.)
plot2.set_xlabel("Attacking Size"), plot2.set_ylabel("Defending Size"), plot2.legend("Major Death")


d3 = battles.groupby('region').sum()[["attacker_size", "defender_size"]].plot(kind='barh', figsize=(15,6), rot=0)
_ = d3.set_xlabel("Army Size"), d3.set_ylabel("Region"), d3.legend(["Attackers", "Defenders"])
d4 = battles['region'].value_counts().plot(kind='barh', figsize=(15,6))
_ = d4.set_xlabel("Number of Battles"), d4.set_ylabel("Regions")

#Define the variables
battleregion = battles['region'].value_counts()
attacksize = battles.groupby('region').sum()[["attacker_size"]]
defendsize = battles.groupby('region').sum()[["defender_size"]]

#Find the average per region
avgattack = attacksize / battleregion
avgdefend = defendsize / battleregion

#Plot it
d5 = mpl.plt(avgattack.sort_values('region'), avgdefend.sort_values('region'), figsize=(15,6), rot=0)
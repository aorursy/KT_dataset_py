import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np

from collections import Counter

pd.set_option("display.max_columns", None)

sns.set(style = "white")
plt.rcParams["axes.labelsize"] = 16.

plt.rcParams["xtick.labelsize"] = 14.

plt.rcParams["ytick.labelsize"] = 14.

plt.rcParams["legend.fontsize"] = 12.

plt.rcParams["figure.figsize"] = [15., 7.]
battles = pd.read_csv("../input/game-of-thrones/battles.csv")

battles.head()
p = battles.groupby("year").count()["battle_number"].plot.bar(rot = 0, color = "steelblue")

_ = p.set(xlabel = "Year", ylabel = "Number of Battles"), plt.yticks(range(0, 21, 2))
p = battles["region"].value_counts().sort_values(ascending = True).plot.barh(rot = 0, color = "steelblue")

_ = p.set(xlabel = "Number of Battles")
attacker = pd.DataFrame(battles.attacker_king.value_counts())

defender = pd.DataFrame(battles.defender_king.value_counts())

data = attacker.merge(defender, how = "outer", left_index = True, right_index = True).fillna(0)

data.loc[:, "Sum"] = data.attacker_king + data.defender_king

p = data.sort_values(["Sum", "attacker_king"]).loc[:, ["attacker_king", "defender_king"]].plot.barh(rot = 0, stacked = True, color = ["steelblue", sns.color_palette()[9]])

plt.xticks(np.arange(0, max(data.Sum) + 2, 2))

_ = p.set(xlabel = "Number of Battles"), p.legend(["Attack Battles", "Defense Battles"])
temp = battles.dropna(subset = ["attacker_king", "defender_king"]).copy(deep = True)

c = list(Counter([tuple(set(x)) for x in temp[["attacker_king", "defender_king"]].values if len(set(x)) > 1]).items())

data = pd.DataFrame(c).sort_values(1)

p = data.plot.barh(color = "steelblue", figsize = (12, 7))

_ = plt.xticks(np.arange(0, max(data[1]) + 2, 2))

_ = p.set(yticklabels = ["%s vs. %s" % (x[0], x[1]) for x in data[0]], xlabel = "Number of Battles"), p.legend().remove()
attacker_dict = {}

for column in ["attacker_1", "attacker_2", "attacker_3", "attacker_4"]:

    val_count = battles[column].value_counts()

    for index, value in val_count.items():

        attacker_dict[index] = attacker_dict.get(index, 0) + value

        

defender_dict = {}

for column in ["defender_1", "defender_2", "defender_3", "defender_4"]:

    val_count = battles[column].value_counts()

    for index, value in val_count.items():

        defender_dict[index] = defender_dict.get(index, 0) + value



attacker_df = pd.DataFrame.from_dict(attacker_dict, orient = "index", columns = ["attacker"])

defender_df = pd.DataFrame.from_dict(defender_dict, orient = "index", columns = ["defender"])



data = attacker_df.merge(defender_df, how = "outer", left_index = True, right_index = True).fillna(0)

data.loc[:, "Sum"] = data.iloc[:,0] + data.iloc[:,1]

data = data[data.Sum > 1]

p = data.sort_values(["Sum", "attacker"]).iloc[:, 0:2].plot.barh(rot = 0, stacked = True, color = ["steelblue", sns.color_palette()[9]])

plt.xticks(np.arange(0, max(data.Sum) + 2, 2))

_ = p.set(xlabel = "Number of Battles"), p.legend(["Attack Battles", "Defense Battles"])
data = battles.loc[battles.attacker_outcome.notna()].copy(deep = True)

f, ax = plt.subplots(1, 2, figsize = (15, 7))

f.suptitle("Outcome Distribution", fontsize = 18.)

_ = data.attacker_outcome.value_counts().plot.bar(ax = ax[0], rot = 0, color = ["steelblue", "lightcoral"]).set(xticklabels = ["Win", "Loss"])

_ = data.attacker_outcome.value_counts().plot.pie(labels = ("Win", "Loss"), autopct = "%.2f%%", label = "", fontsize = 14., ax = ax[1],\

colors = ["steelblue", "lightcoral"], wedgeprops = {"linewidth": 1.5, "edgecolor": "#F7F7F7"}), ax[1].texts[1].set_color("#F7F7F7"), ax[1].texts[3].set_color("#F7F7F7")
data = battles.dropna(subset = ["attacker_size", "defender_size", "attacker_outcome"]).copy(deep = True)

p = sns.scatterplot("attacker_size", "defender_size", hue = "attacker_outcome", palette = ["steelblue", "lightcoral"], data = data, s = 200)

_ = p.set(xlabel = "Attacker Size", ylabel = "Defender Size")

legend = p.legend()

legend.texts[0].set_text("Attacker Outcome")

legend.texts[1].set_text("Win")

legend.texts[2].set_text("Loss")
data.loc[data.attacker_size == max(data.attacker_size)]
battles.loc[27, "attacker_king"] = "Mance Rayder"

battles.loc[27, "defender_king"] = "Stannis Baratheon"
attacker = battles.dropna(subset = ["attacker_commander"]).copy(deep = True)

defender = battles.dropna(subset = ["defender_commander"]).copy(deep = True)



d = {}



for names in attacker["attacker_commander"].values:

    name_lst = names.split(", ")

    for name in name_lst:

        d[name] = d.get(name, 0) + 1



for names in defender["defender_commander"].values:

    name_lst = names.split(", ")

    for name in name_lst:

        d[name] = d.get(name, 0) + 1

        

data  = pd.DataFrame.from_dict(d, orient = "index", columns = ["Count"])

p = data.loc[data.Count > 2].sort_values("Count").plot.barh(color = "steelblue")

plt.xticks(np.arange(0, max(data.Count) + 1, 1))

_ = p.set(xlabel = "Number of Battles"), p.legend().remove()
attacker = battles.dropna(subset = ["attacker_commander", "attacker_outcome"]).copy(deep = True)

defender = battles.dropna(subset = ["defender_commander", "attacker_outcome"]).copy(deep = True)



d = {}



for pair in attacker[["attacker_commander", "attacker_outcome"]].values:

    name_lst, outcome = pair[0].split(", "), pair[1]

    for name in name_lst:

        if outcome == "win":

            d[name] = d.get(name, 0) + 1



for pair in defender[["defender_commander", "attacker_outcome"]].values:

    name_lst, outcome = pair[0].split(", "), pair[1]

    for name in name_lst:

        if outcome == "loss":

            d[name] = d.get(name, 0) + 1



data  = pd.DataFrame.from_dict(d, orient = "index", columns = ["Count"])

p = data.loc[data.Count > 1].sort_values("Count").plot.barh(color = "steelblue")

plt.xticks(np.arange(0, max(data.Count) + 1, 1))

_ = p.set(xlabel = "Number of Victories"), p.legend().remove()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/battles.csv")

df.head(10)
kings = pd.Series(df["attacker_king"].append(df["defender_king"].reset_index(drop=True)).unique())

kings.dropna(inplace = True)

kings
outcomes = df[["name", "attacker_king", "defender_king", "attacker_outcome"]]

outcomes.head(3)
attwin = outcomes[outcomes.attacker_outcome == "win"]["attacker_king"]

attwin.dropna(inplace = True)

attwin
defwin = outcomes[outcomes.attacker_outcome == "loss"]["defender_king"]

defwin.dropna(inplace = True)

defwin
winnerkings = defwin.append(attwin).reset_index(drop=True)

winnerkings
winnerkings.value_counts()


kingsscores = pd.DataFrame({"kings_name":[], "number_of_wins": []})

i = 0

for king in kings:

    if king in winnerkings.value_counts():

        kingsscores.loc[df.index[i],["kings_name"]] = king

        kingsscores.loc[df.index[i],["number_of_wins"]] = winnerkings.value_counts()[king]

    else:

        kingsscores.loc[df.index[i],["kings_name"]] = king

        kingsscores.loc[df.index[i],["number_of_wins"]] = 0

    i +=1



kingsscores
plt.figure(dpi = 200)

plt.bar(kingsscores["kings_name"], kingsscores["number_of_wins"], 

        edgecolor = "k",

       )

plt.xticks(kings, rotation = "vertical")

plt.grid()
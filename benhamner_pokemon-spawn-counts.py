import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



spawns = pd.read_csv("../input/pokemon-spawns.csv")

counts = (spawns.groupby("name").size().to_frame().reset_index(level=0)

          .rename(columns={0: "Count", "name": "Pokemon"}).sort_values(by="Count", ascending=False))



sns.set(style="whitegrid")

sns.set_color_codes("muted")

f, ax = plt.subplots(figsize=(11, 35))

sns.barplot(x="Count", y="Pokemon", data=counts, color="b")

ax.set_title("Pokemon Spawn Counts in San Francisco")

ax.set_xlabel("Number of Spawns")
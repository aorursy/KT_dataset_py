import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None 

df = pd.read_csv("../input/pokemon.csv")

print(df.head())

#select types values

types_selection = [

    "water",

    "grass",

    "fire",

    "normal"

]

main_types_df = df.loc[df["type1"].isin(types_selection)]



main_types_df.type1.value_counts().plot(kind='pie', startangle=10, title="Pokemon per type", autopct="%1.1f%%",figsize=(10,10), legend=False)



combat_columns = [

    "type1",

    "attack",

    "defense",

    "hp",

    "speed"

]





combat_main_types_df = main_types_df[combat_columns]





combat_mean_grouped_type = combat_main_types_df.groupby(["type1"]).mean()

combat_mean_grouped_type.plot(kind="bar", figsize=(15, 10), ylim=(40, 85))
import seaborn as sns



f, axs = plt.subplots(2, 2, figsize=(15, 10))

sns.boxplot(x="type1", y="attack", data=combat_main_types_df, ax=axs[0, 0])

sns.boxplot(x="type1", y="defense", data=combat_main_types_df, ax=axs[0, 1])

sns.boxplot(x="type1", y="speed", data=combat_main_types_df, ax=axs[1, 0])

sns.boxplot(x="type1", y="hp", data=combat_main_types_df, ax=axs[1, 1])
other_columns = [

    "type1",

    "weight_kg",

    "height_m",

    "base_happiness",

    "percentage_male"

]



other_main_types_df = main_types_df[other_columns]

cm = other_main_types_df["height_m"].apply(lambda x: x * 100)

other_main_types_df["height_cm"] = cm

other_main_types_df = other_main_types_df.drop(["height_m"], axis=1)

other_mean_grouped_type = other_main_types_df.groupby(["type1"]).mean()

other_mean_grouped_type.plot(kind="bar", figsize=(15, 15))
f, axs = plt.subplots(2, 2, figsize=(15, 10))

sns.boxplot(x="type1", y="weight_kg", data=other_main_types_df, ax=axs[0, 0])

sns.boxplot(x="type1", y="height_cm", data=other_main_types_df, ax=axs[0, 1])

sns.boxplot(x="type1", y="percentage_male", data=other_main_types_df, ax=axs[1, 0])

sns.boxplot(x="type1", y="base_happiness", data=other_main_types_df, ax=axs[1, 1])
p_types = [

    ("water", "blue"),

    ("normal", "gray"),

    ("grass", "green"),

    ("fire", "red"),

]

ax = None

for p_type in p_types:

    sub_df = df.loc[df["type1"] == p_type[0]]

    ax = sub_df.plot(x="defense", y="attack", figsize=(15,15), s=100, kind="scatter", ax=ax, color=p_type[1], alpha=0.50)

    
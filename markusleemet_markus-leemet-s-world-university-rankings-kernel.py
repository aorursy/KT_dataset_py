import numpy as np

import pandas as pd 



df = pd.read_csv("../input/cwurData.csv")

df
df[df["country"] == "Estonia"]
df.groupby(["country"])["quality_of_education"].mean().round(1).sort_values(ascending=False)
tulemused = df[df["year"] == 2015]

tulemused

tulemused["country"].value_counts()
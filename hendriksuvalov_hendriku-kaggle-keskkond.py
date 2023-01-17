import numpy as np

import pandas as pd





pd.set_option("display.max_rows", 20)

df = pd.read_csv("../input/cwurData.csv")

df
df[df["country"] == "Estonia"]
df.groupby("country")["quality_of_education"].mean()
df2= pd.DataFrame(df.groupby("country")["quality_of_education"].mean())

df2.sort_values("quality_of_education", ascending = False)
df3 = df[df["year"] == 2015]

df3.country.value_counts()

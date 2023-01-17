import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 20)



df = pd.read_csv("../input/cwurData.csv")

df
df[df["country"] == "Estonia"]
df.groupby("country")["quality_of_education"].mean()
df_2 = pd.DataFrame(df.groupby("country")["quality_of_education"].mean())

df_2.sort_values("quality_of_education", ascending = False)

df_3 = df[df["year"] == 2015]

df_3.country.value_counts()
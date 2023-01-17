import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 10)



df = pd.read_csv("../input/cwurData.csv")



df
df2 = df[df["country"] == "Estonia"]

df2 
df.sort_values("quality_of_education")
df3 = df.sort_values("quality_of_education", ascending=False)

df3



koolid = set(df["country"])

print("Erinevaid koole: " + str(len(koolid)))

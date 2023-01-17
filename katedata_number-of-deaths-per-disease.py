import pandas as pd
pd.set_option('display.max_rows', 1115000)
df1 = pd.read_csv("../input/Outbreak_240817.csv")
df2 = df1.set_index("disease", drop = False)
df3 =df2.loc[: , "sumDeaths"]
df3 = df3.dropna(axis=0, how='all')
df3 = df3.groupby(df3.index).sum()
df3

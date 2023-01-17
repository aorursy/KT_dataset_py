import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 20)



df = pd.read_csv("../input/guns.csv")



df
df.intent.value_counts(ascending=False)
df["year"].value_counts()
df.education.plot.hist(bins=5, grid=False, rwidth=0.8);
df["age"].describe()
df.plot.scatter("age", "education", alpha=0.002);
df["month"].value_counts().plot.bar(width=0.9);
df2 = pd.DataFrame(df.groupby("race")["age"].mean())

df2.sort_values("age", ascending = False)

df3 = df[df["police"] == 1]

df3.race.value_counts().plot.bar(width=0.9);
df4 = pd.DataFrame(df.groupby("sex")["age"].mean())

df4

df5 = df[df["intent"] == 'Suicide']

df5.sex.value_counts(normalize=True)
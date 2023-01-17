import pandas as pd 
import numpy as np

df = pd.read_csv("../input/04182020-covid19-population-data/04182020_covid19_population_data.csv")
df.head()

df.dtypes
df.describe()
y=df['Deaths Per Millsion']

df_state_dummies = pd.get_dummies(df["State"])
X=df.drop(labels=["Deaths Per Millsion", "State"], axis=1)

X.corr()

X = pd.concat([df_state_dummies, X], axis=1)
X.head()

df_corr = X.corr()
df_corr["Deaths"].sort_values(ascending=False)


import numpy as np

import pandas as pd

import seaborn as sns 

import matplotlib.pyplot as plt

%matplotlib inline



df = pd.read_csv("../input/Pokemon.csv")

df.head()
df.info()
df.describe()
poke = df.groupby('Generation',as_index=False)["#"].count()

poke
sns.stripplot(x="Legendary",y="HP",data=df,jitter=True)
sns.stripplot(x="Generation",y="HP",data=df,jitter=True)
sns.stripplot(x="Generation",y="Attack",data=df,jitter=True)
sns.stripplot(x="Generation",y="Attack",data=df,jitter=True)
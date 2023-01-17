import pandas as pd

df = pd.read_csv('../input/directory.csv')

df.info() 
df.head()
#show the entire world

import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="ticks")

sns.set_context("talk")



x = df['Longitude']

y = df['Latitude']



sns.jointplot(x, y, kind="hex", size=10, color="#4CB391", stat_func=None, 

              xlim=(-180,180), ylim=(-180,180))
countries = (df[["Country", "Store Number"]].groupby(["Country"], as_index=False).count())

countries.sort_values(by='Store Number', axis=0, ascending=False, inplace=True)

countries.head(25)
countries = (df[["City", "Store Number"]].groupby(["City"], as_index=False).count())

countries.sort_values(by='Store Number', axis=0, ascending=False, inplace=True)

countries.head(25)
#show North America

sns.jointplot(x, y, kind="hex", size=10, color="#4CB391", 

              xlim=(-130,-60), ylim=(10,60), stat_func=None)
#show Europe

sns.jointplot(x, y, kind="hex", size=10, color="#4CB391", 

              xlim=(-25,50), ylim=(25,65), stat_func=None)
#show Asia

sns.jointplot(x, y, kind="hex", size=10, color="#4CB391", 

              xlim=(60,165), ylim=(-10,50), stat_func=None)
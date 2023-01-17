# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



%matplotlib inline

pd.set_option('display.max_rows', 20)



df = pd.read_csv('../input/All-seasons.csv')

df
df = df.replace('\n', '', regex=True)

df
df.info()
df.Season = pd.to_numeric(df.Season, errors='coerce').fillna(0).astype(np.int64)

df.Episode = pd.to_numeric(df.Episode, errors='coerce').fillna(0).astype(np.int64)

lause_pikkus = df["Line"].str.len()

#lause_pikkus #näitab iga rea kohta lausete pikkuseid (mitu tähemärki)

df = df.assign(lause_pikkus=lause_pikkus)

df.info()
df['Character'].value_counts()
df["lause_pikkus"].describe()
df1 = (df[["Season", "Character", "Line", "lause_pikkus"]][df["Character"]=="Cartman"]

 .sort_values("lause_pikkus", ascending=False))

df2 = (df[["Season", "Character", "Line", "lause_pikkus"]][df["Character"]=="Stan"]

 .sort_values("lause_pikkus", ascending=False))

df3 = (df[["Season", "Character", "Line", "lause_pikkus"]][df["Character"]=="Kyle"]

 .sort_values("lause_pikkus", ascending=False))

df4 = (df[["Season", "Character", "Line", "lause_pikkus"]][df["Character"]=="Kenny"]

 .sort_values("lause_pikkus", ascending=False))
plt.subplot(2 , 2, 1)

df1.lause_pikkus.plot.hist(range=(0, 100), bins=10, rwidth=0.7, color='orange')

plt.title("Cartman")

plt.ylabel("Lausete arv")

plt.subplot(2 , 2, 2)

df2.lause_pikkus.plot.hist(range=(0, 100), bins=10, rwidth=0.7, color='orange')

plt.title("Stan")

plt.ylabel("Lausete arv")

plt.subplot(2 , 2, 3)

df3.lause_pikkus.plot.hist(range=(0, 100), bins=10, rwidth=0.7, color='orange')

plt.title("Kyle")

plt.ylabel("Lausete arv")

plt.subplot(2 , 2, 4)

df4.lause_pikkus.plot.hist(range=(0, 100), bins=10, rwidth=0.7, color='orange')

plt.title("Kenny")

plt.ylabel("Lausete arv")

plt.tight_layout()
(df[["Character", "Line", "lause_pikkus"]]

 .sort_values("lause_pikkus", ascending=False).head(20))
df5 = (df[["Season", "Character", "Line", "lause_pikkus"]][df["Character"]=="Kanye"]

 .sort_values("lause_pikkus", ascending=False))

df5["lause_pikkus"].describe()

#df5
df5.lause_pikkus.plot.hist(range=(0, 200), bins=10, rwidth=0.7, color='orange')

plt.ylabel("Lausete arv")

plt.tight_layout()
df['lause_pikkus'].value_counts()
df7 = df.groupby(["lause_pikkus"]).aggregate({"Character": ["count"], "lause_pikkus": ["mean"]})

df7
df7.plot.scatter('lause_pikkus', "Character", alpha=0.3)

plt.ylabel("Esinemiste arv")

plt.xlabel("Lause pikkus")
lause_pikkus2 = df["Line"].str.len()

df8 = df.assign(lause_pikkus2=lause_pikkus2)

df8 = (df8[["Season", "Character", "Line", "lause_pikkus", "lause_pikkus2"]][df["Character"]=="Kyle"]

 .sort_values("lause_pikkus", ascending=False))

df9 = df8.groupby(["lause_pikkus"]).aggregate({"lause_pikkus": ["count"], "lause_pikkus2" : ["mean"]})
df9.plot.scatter("lause_pikkus2", "lause_pikkus", alpha=0.3)

plt.ylabel("Esinemiste arv")

plt.xlabel("Lause pikkus")
df.Line.value_counts().head(20)
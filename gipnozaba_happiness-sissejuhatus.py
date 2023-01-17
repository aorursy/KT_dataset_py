import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

ds = pd.read_csv("../input/2016.csv")

df = pd.read_csv("../input/2017.csv")
ds["Happiness Score"].hist(bins=20, grid=False, rwidth=0.44); 

df["Happiness.Score"].hist(bins=20, grid=False, rwidth=0.44);
df.plot.scatter("Happiness.Score", "Generosity", alpha=0.5);

df.plot.scatter("Economy..GDP.per.Capita.", "Generosity", alpha=0.5);

df.plot.scatter("Economy..GDP.per.Capita.", "Happiness.Score", alpha=0.5);

df.plot.scatter("Happiness.Score", "Freedom", alpha=0.5);

# Kirjeldus suuliselt
ds.groupby("Region")["Happiness Score"].mean().sort_values(ascending=False)
ds[ds["Region"]=="Western Europe"].min()
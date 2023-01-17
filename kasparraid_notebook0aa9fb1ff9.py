import numpy as np

import pandas as pd

%matplotlib inline

df = pd.read_csv("../input/vgsales.csv")

df
df.Year.plot.hist(bins=11, grid=False, rwidth=0.95);
df[(df["Name"].str.len() > 7) & (df["Year"] > 2000)].plot.scatter("Global_Sales", "Year", alpha=0.2);
df[(df['Name'].str.len() <= 7)& (df["Year"] > 2000)].plot.scatter("Global_Sales", "Year", alpha=0.2);
df[df["Year"] > 2018]
df.groupby(["Year", df["Name"].str.len() <= 7]).aggregate({"Global_Sales": ["sum", "median"]})
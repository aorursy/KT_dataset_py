import numpy as np

import pandas as pd



%matplotlib inline

pd.set_option('display.max_rows', 20)
df = pd.read_csv("../input/Airplane_Crashes_and_Fatalities_Since_1908.csv")

df
df["Year"] = df["Date"].apply(lambda x: int(str(x)[-4:]))

df.plot.scatter("Year", "Fatalities", alpha=0.3);
df2 = df[df["Fatalities"]>0]

df2.groupby(["Date"])["Fatalities"].max().sort_values(ascending = False)

df["Fatalities"].plot.hist(grid=True, rwidth=0.75);
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

pd.set_option('display.max_rows', 20)

df = pd.read_csv("../input/2017.csv")

df.info()
df["Happiness.Score"]
print("Keskmine skoor:", df["Happiness.Score"].mean())

print("Maksimaalne skoor:", df["Happiness.Score"].max())

print("Minimaalne skoor:", df["Happiness.Score"].min())
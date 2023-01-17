import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



df = pd.read_csv("../input/winemag-data_first150k.csv")

df.groupby("points")["price"].mean().plot(title="Hinna ja veini punktide suhe)");
df["points"].plot.hist(title="Veinide hinde jaotus", grid=True, rwidth=0.2);
df.plot.scatter("points","price", marker="$ - $", alpha=0.1,s=67,color="purple", title="Veinide punktid hinna järgi");
df_lõpp = pd.DataFrame(df.groupby(["points"])["price"].mean())

                      

df_lõpp
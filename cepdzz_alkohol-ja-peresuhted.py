import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv("../input/student-mat.csv")

df["Pstatus"] = df["Pstatus"].map({"A": "Lahku", "T": "Koos"})

df["famsize"] = df["famsize"].map({"GT3":"Rohkem kui 3", "LE3":"Vähem kui 3"})
df1 = pd.DataFrame(df.groupby("age").aggregate({"Dalc":["mean"],"Walc":["mean"]}))

df1.plot.bar()
df2 = pd.DataFrame(df.groupby("famsize").aggregate({"Dalc": ["mean"], "Walc": ["mean"]})

                        .rename(columns={"Dalc":"Nädalapäevas alk", 

                                         "Walc":"Nädalavahetusel alk"}).round(2))

df2.plot.bar()
df3 =  pd.DataFrame(df.groupby("age").aggregate({"Dalc": ["mean"], 

                                                    "Walc": ["mean"], 

                                                    "health":["mean"]})

                        .rename(columns={"Dalc":"Nädalapäevas alk", 

                                         "Walc":"Nädalavahetusel alk", 

                                         "health": "Tervis"}).round(2))

df3.plot()
df_sc1 = df.plot.scatter("famrel","Dalc",s=200,alpha=0.05)

df_sc2 = df.plot.scatter("famrel","Walc",s=200, alpha=0.05)
df_lõpp = pd.DataFrame(df.groupby(["age","famsize","Pstatus"])["Dalc","Walc","health","famrel"].mean()

                       .round(2).rename(columns={"Dalc":"Nädalapäevas alk","Walc":"Nädalavahetusel alk",

                                      "health":"Tervis", "famrel":"Peresuhted"}))

df_lõpp.index.names = "Vanus", "Peresuurus", "Vanemate kooselavus"

df_lõpp
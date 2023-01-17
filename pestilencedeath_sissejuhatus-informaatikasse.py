import numpy as np

import pandas as pd 

import matplotlib.pyplot as plt



%matplotlib inline

pd.set_option('display.max_rows', 20)
df = pd.read_csv('../input/database.csv', low_memory=False)
df = df[["Year", "Month", "Crime Solved", "Victim Age", "Perpetrator Age", "Relationship", "Weapon"]]

df
pd.DataFrame({"Lahendatud või mitte?": df["Crime Solved"].value_counts()})
i=pd.Series(df["Crime Solved"])

d=pd.DataFrame({"Arvukus": i})

d.apply(pd.value_counts).plot(kind="bar", subplots=True)
pd.DataFrame({"Mõrvarelvade arvukus": df["Weapon"].value_counts()})
relvad=pd.Series(df["Weapon"])

d=pd.DataFrame({"Mõrvariistade arv": relvad})

d.apply(pd.value_counts).plot(kind="bar", subplots=True)
df['Perpetrator Age'] = pd.to_numeric(df['Perpetrator Age'], errors='coerce')
df = df[df["Perpetrator Age"] != 0]

df = df[df["Victim Age"] >= 20]

df = df[df["Victim Age"] <= 30]
df.plot.scatter("Perpetrator Age", "Victim Age", alpha=0.2);
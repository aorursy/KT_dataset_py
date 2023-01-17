# To obtain the file path

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import pandas as pd

import numpy as np
df = pd.read_csv("/kaggle/input/ecommerce-purchases/Ecommerce Purchases")
df.head()
df.info()
df["Purchase Price"].mean()
df["Purchase Price"].max()
df["Purchase Price"].min()
(df["Language"] == "en").value_counts()
df[df["Job"] == "Lawyer"].count()
df["AM or PM"].value_counts()
df["Job"].value_counts().head()
df[df["Lot"] == "90 WT"]["Purchase Price"]
df[df["Credit Card"] == 4926535242672853 ]["Email"]
df[(df["CC Provider"] == "American Express")

    & (df["Purchase Price"] > 95)].count()


sum(df["CC Exp Date"].apply(lambda x: x[3:] == "24"))
df["Email"].apply(lambda x: x.split("@")[1]).value_counts().head()
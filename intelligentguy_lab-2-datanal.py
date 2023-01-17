# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/vgsale_1.csv")

df
df[df.isnull().any(axis=1)]
df.dropna(inplace=True)
befor2000 = df[df["Year"] < 2000][["Name", "Genre", "Global_Sales"]]

after2000 = df[df["Year"] >= 2000][["Name", "Genre", "Global_Sales"]]

befor2000 = befor2000.groupby("Genre").agg(global_sales = ('Global_Sales','sum'), games = ("Name", "nunique")).apply(lambda x: x/x.sum(), axis=0)

after2000 = after2000.groupby("Genre").agg(global_sales = ('Global_Sales','sum'), games = ("Name", "nunique")).apply(lambda x: x/x.sum(), axis=0)

befor2000
befor2000.plot.bar(figsize=(18, 10))

after2000.plot.bar(figsize=(18, 10))
df.groupby("Year")["Name"].nunique().plot(figsize=(18, 10))
publishers = df.groupby("Publisher")["Name"].nunique().nlargest(5)

publishers
platforms = df[df["Publisher"].isin(publishers.index)].groupby(["Publisher","Platform"])["Name"].nunique()

platforms = platforms[platforms > 20]

platforms
platforms = platforms.unstack(level=0)

platforms
platforms.plot.bar(figsize=(22, 15), layout=(3, 2), stacked=True)
befor2000 = df[df["Year"] < 2000][["NA_Sales", "EU_Sales", "JP_Sales", "Global_Sales"]].sum(axis=0)

after2000 = df[df["Year"] >= 2000][["NA_Sales", "EU_Sales", "JP_Sales", "Global_Sales"]].sum(axis=0)

befor2000 = befor2000.loc["NA_Sales":"JP_Sales"] / befor2000.loc["Global_Sales"]

after2000 = after2000.loc["NA_Sales":"JP_Sales"] / after2000.loc["Global_Sales"]

sales = pd.DataFrame({"befor 2000": befor2000, "after 2000": after2000})

sales
sales.plot.pie(subplots=True, figsize=(18, 12))
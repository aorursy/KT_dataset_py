# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline
with pd.HDFStore("../input/train.h5", "r") as train:

    # Note that the "train" dataframe is the only dataframe in the file

    df = train.get("train")
print("Number of instruments:", len(df["id"].unique()))

print("Min ID:", df["id"].min())

print("Max ID:", df["id"].max())
stats = df.groupby("id")["y"].agg({"mean":np.mean, "count":len})

sns.jointplot(x="count", y="mean", data=stats)
df.groupby("id")["y"].mean().sort_values().head()
cols_to_use = ['y', 'technical_30', 'technical_20', 'fundamental_11', 'technical_19']

fig = plt.figure(figsize=(8, 20))

plot_count = 0

for col in cols_to_use:

    plot_count += 1

    plt.subplot(5, 2, plot_count)

    plt.plot(df["timestamp"].sample(frac=0.01), df[col].sample(frac=0.01), ".")

    plt.title("Distribution of {}".format(col))

    plot_count += 1

    plt.subplot(5, 2, plot_count)

    plt.plot(df.loc[df["id"]==1431, "timestamp"], df.loc[df["id"]==1431, col], ".-", label="ID 1431")

    plt.plot(df.loc[df["id"]==11, "timestamp"], df.loc[df["id"]==11, col], ".-", label="ID 11", alpha=0.7)

    plt.plot(df.loc[df["id"]==12, "timestamp"], df.loc[df["id"]==12, col], ".-", label="ID 12", alpha=0.7)

    plt.legend()

plt.show()
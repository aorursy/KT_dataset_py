# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv("../input/movie_metadata.csv")
df.head()
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



sns.set_context("poster")

sns.set_style("ticks")
df.columns
df_t = df.pivot_table(index="title_year", values="color", aggfunc=len)

df_t.head()
plt.plot(df_t.index, df_t.values, "-bo")

plt.xlabel("Year")

plt.ylabel("Total movies")

plt.yscale("log")
def plot_vs_year(df, y="gross"):

    df_t = df.pivot_table(index="title_year", values=y, aggfunc=np.mean)

    plt.plot(df_t.index, df_t.values, "-bo")

    plt.ylabel(y)

    plt.xlabel("Year")
plot_vs_year(df, "gross")
plot_vs_year(df, "facenumber_in_poster")
plot_vs_year(df, "aspect_ratio")
plot_vs_year(df, "budget")
ax = sns.countplot(x="language", data=df)

ax.set_yscale("log")

labels = ax.get_xticklabels()

_ = plt.setp(labels, rotation=90, fontsize=10)
ax = sns.countplot(x="country", data=df)

ax.set_yscale("log")

labels = ax.get_xticklabels()

_ = plt.setp(labels, rotation=90, fontsize=10)
df.plot_keywords.head()
df_keywords = df.plot_keywords.apply(lambda x: [] if isinstance(x, float) else x.split("|"))

df_keywords.head()
df_plots = pd.Series(sum(df_keywords.values.tolist(), []))

df_plots.head()
plot_counts = df_plots.value_counts().sort_values(ascending=False)

print("Plot counts: ", plot_counts.shape)

#plot_counts.head()
plot_counts[plot_counts > 50].index
ax = sns.countplot(x=df_plots[df_plots.isin(plot_counts[plot_counts > 50].index.values)])

ax.set_yscale("log")

labels = ax.get_xticklabels()

_ = plt.setp(labels, rotation=90, fontsize=10)
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
import pandas as pd

import numpy as np



import gensim

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv("../input/primary_debates_cleaned.csv", parse_dates=["Date"])
df.head()
ax = sns.countplot(x="Location", data=df, color="red")

labels = ax.get_xticklabels()

plt.setp(labels, rotation=90)

print("Done")
df.Speaker.value_counts().head(20)
ax = sns.countplot(x="Speaker", 

                   data=df[df.Speaker.isin(

                       set(df.Speaker.value_counts().head(20).index.tolist()) 

                       - set(["AUDIENCE"]))],

                   color="red")

labels = ax.get_xticklabels()

plt.setp(labels, rotation=90)

print("Done")
df.dtypes
df_filtered = df[df.Speaker.isin(

                       set(df.Speaker.value_counts().head(10).index.tolist()) 

                       - set(["AUDIENCE"]))].copy()
g = sns.FacetGrid(df_filtered, col="Speaker", col_wrap=3)

g.map(sns.countplot, "Location")

for i in range(-3, 0):

    labels = g.axes[i].get_xticklabels()

    plt.setp(labels, rotation=90)

print("Done")
df_filtered.Location.value_counts()
df_filtered.pivot_table(index="Location", values="Party", columns="Speaker", aggfunc=len)
df.Location.value_counts()
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import re

import math

import datetime as dt

import matplotlib.pyplot as plt

import seaborn as sns

import missingno as mn

from pandas.tools.plotting import scatter_matrix



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



if os.path.exists("../input/PakistanSuicideAttacks Ver 6 (10-October-2017).csv"):

    print("File exists")



def parser(x):

    t = re.findall(r"[\w]+",x)

    t[1] = t[1][:3]

    t.pop(0)

    t1 = " ".join(t)

    return dt.datetime.strptime(t1, "%b %d %Y")

    

df = pd.read_csv("../input/PakistanSuicideAttacks Ver 6 (10-October-2017).csv", encoding="latin1", parse_dates=['Date'], date_parser = parser, dayfirst = True, index_col=[0], converters={'Longitute':np.float64})

df.Longitude = pd.Series(map(float, df.Longitude))



print("First look at features, numbers and their types \n")

print(df.dtypes)

print(df.shape)
mn.bar(df)

mn.matrix(df)

mn.heatmap(df)

mn.dendrogram(df)
df.boxplot(figsize=(16,9))

#df.hist(alpha=0.4, figsize=(16,9))



scatter_matrix(df, alpha=0.6, figsize=(16,9), diagonal='kde');
cols = df.select_dtypes(include=object)

cols.head()
df['Islamic Month'] = df["Islamic Date"].apply(lambda row: re.findall(r"[\D]+", row)[0].lower() if pd.notnull(row) else "")
cols.drop(['Islamic Date', 'Holiday Type', 'Time', 'Location', 'Influencing Event/Event','Hospital Names','Injured Max'], inplace=True, axis=1)
from matplotlib import gridspec



col = 2

row = int(len(cols) / col)



gs = gridspec.GridSpec(row,col)

fig = plt.figure(figsize=(16,450))



for i,v in enumerate(cols):

    ax1 = fig.add_subplot(gs[i])

    sns.countplot(cols[v])

    plt.xticks(rotation=90)

plt.tight_layout()
df['Date'].value_counts()
cols
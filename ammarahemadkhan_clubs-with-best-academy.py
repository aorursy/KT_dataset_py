# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd

df15 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_15.csv")

df16 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_16.csv")

df17 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_17.csv")

df18 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_18.csv")

df19 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_19.csv")

df20 = pd.read_csv("/kaggle/input/fifa-20-complete-player-dataset/players_20.csv")

#find best players in FIFA 18,19,20

df20_top = df20[((df20["age"]<=25) & (df20["age"]>=21))].sort_values(by = "overall",ascending = False).head(200)

df19_top = df19[((df19["age"]<=25) & (df19["age"]>=21))].sort_values(by = "overall",ascending = False).head(200)

df18_top = df18[((df18["age"]<=25) & (df18["age"]>=21))].sort_values(by = "overall",ascending = False).head(200)

df20_top.head()

#finding whereabouts of above top players three years ago

#seeing top players of FIFA 20 in FIFA 17

#seeing top players of FIFA 19 in FIFA 16

#seeing top players of FIFA 18 in FIFA 15





df17_prospects = df17.loc[df17["long_name"].isin(list(df20_top["long_name"]))]

df16_prospects = df16.loc[df16["long_name"].isin(list(df19_top["long_name"]))]

df15_prospects = df15.loc[df15["long_name"].isin(list(df18_top["long_name"]))]

df17_prospects.head()
#grouping players with their former clubs

data1 = df17_prospects.groupby("club").agg({"club":"count"}).rename(columns = {"club":"2017_Batch"}).sort_values(by = "2017_Batch",ascending = False).head(40)

data2 = df16_prospects.groupby("club").agg({"club":"count"}).rename(columns = {"club":"2016_Batch"}).sort_values(by = "2016_Batch",ascending = False).head(40)

data3 = df15_prospects.groupby("club").agg({"club":"count"}).rename(columns = {"club":"2015_Batch"}).sort_values(by = "2015_Batch",ascending = False).head(40)

df = data1.merge(data2,how = "inner",left_index = True,right_index = True).merge(data3,how = "inner",left_index = True,right_index = True)

df["total"] = df["2017_Batch"]+df["2016_Batch"]+df["2015_Batch"]

df.sort_values(by = "total",ascending = False,inplace = True)

df
%matplotlib notebook

import matplotlib.pyplot as plt

plt.figure(figsize = (13,8))

plt.barh(y = df.index.values,width = df["total"],height = 0.5)

plt.title("Top Academies")

plt.xlabel("Number of players in Top 200 after 3 years")

plt.show()
df15_yp= df15[df15["age"]<24].sort_values(by = "potential",ascending = False).head(200)

df_ep = df20.loc[df20["long_name"].isin(list(df15_yp["long_name"]))]

club15 = df_ep.groupby("club").agg({"club":"count"}).rename(columns = {"club":"count"}).sort_values(by = "count",ascending = False).head(15)

club15
plt.figure(figsize = (13,8))

plt.barh(y = club15.index.values,width = club15["count"],height = 0.5)

plt.title("Most Active club in acquiring young prospects of 2015")

plt.xlabel("Number of top prospects acquired")

plt.show()
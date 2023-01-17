# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
base_df = pd.read_csv('../input/aomorilinepoint.csv')

base_df.head()
df = pd.read_csv('../input/h28_fishaomori_utf8_ver3.csv')

df.head()
df['type'].value_counts()
targettype=df[df['type']=="(92)たこ類"]

targetmax=targettype['total (t)'].max()



targettype.sort_values(by="total (t)", ascending=False).head(10)
import matplotlib.pyplot as plt

import seaborn as sns




base_x = base_df["lon"]

base_y = base_df["lat"]



x=targettype["lon"]

y=targettype["lat"]

s=targettype['total (t)'] / targetmax



plt.figure(figsize=(10,10))

plt.title("octopus")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.scatter(base_x, base_y,s=1, alpha=1,color="blue")

plt.scatter(x, y,s=s*150, alpha=1,color="red")

plt.show()
targettype=df[df['type']=="(83)ほたてがい"]

targetmax=targettype['total (t)'].max()



targettype.sort_values(by="total (t)", ascending=False).head(10)
x=targettype["lon"]

y=targettype["lat"]

s=targettype['total (t)'] / targetmax



plt.figure(figsize=(10,10))

plt.title("scallop")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.scatter(base_x, base_y,s=1, alpha=1,color="blue")

plt.scatter(x, y,s=s*150, alpha=1,color="red")

plt.show()
targettype=df[df['type']=="(1)くろまぐろ"]

targetmax=targettype['total (t)'].max()



targettype.sort_values(by="total (t)", ascending=False).head(10)
x=targettype["lon"]

y=targettype["lat"]

s=targettype['total (t)'] / targetmax



plt.figure(figsize=(10,10))

plt.title("bluefin tuna")

plt.xlabel("Longitude")

plt.ylabel("Latitude")

plt.scatter(base_x, base_y,s=1, alpha=1,color="blue")

plt.scatter(x, y,s=s*150, alpha=1,color="red")

plt.show()
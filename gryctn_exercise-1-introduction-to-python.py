# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# to read dataset in specified path

df = pd.read_csv("../input/google-play-store-apps/googleplaystore.csv")
# to check data columns(13)/ entry range(row number-10841)/ data types(float & object)

df.info()
# to see first 10 entries

df.head(10)
# names of column

df.columns
# when column names are checked, unique values are found out on category column.

category = df[["Category"]].values

uc = np.unique(category)

uc
# Filtering Pandas data frame

x = (df['Rating'] == 5.0) & (df["Category"] == "GAME")  # data['Rating'] = 5.0 and category 'Game'

df[x]                    # There are 12 applications which have 5.0 rating in category 'Game'
# Rating histogram 

# bins - number of bar in figure

df.Rating.plot(kind="hist",range=[1, 5], bins = 50, figsize=(15,15),grid=True,alpha=0.8)

plt.xlabel("Rating")

plt.show()
# string that contains numeric values can be converted to numeric 

df['SizeInt'] = df['Size'].str.extract('(\d+)').astype(float)

df['SizeInt']
# kind = line plot (it can be line,histogram,scatter)

# alpha = opacity

# lineplot for size of applications

df.SizeInt.plot(kind="line",color="b",label="Size",linewidth=2,alpha=0.4,grid=True,linestyle=':',figsize=(30,10))

plt.legend(loc="upper left")

plt.xlabel("application numbers") 

plt.ylabel("size") 

plt.title("Line Plot for size")

plt.show()
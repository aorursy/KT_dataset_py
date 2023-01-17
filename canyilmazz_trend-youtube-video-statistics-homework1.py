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
#First ı will import some visualization libaries

import matplotlib.pyplot as plt

import seaborn as sns
#I will read csv file that includes data about US trend videos

data=pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

data.head()
data.info()
data.describe().T
data.columns
data.likes.plot(kind="line",color="red",label="likes",linewidth=1,alpha=0.5,linestyle=":")

data.dislikes.plot(color="green",label="dislikes",linewidth=1,alpha=0.5,linestyle="-")

plt.legend(loc="upper right")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Line Plot")

plt.show()

data.plot(kind="scatter",x="views",y="likes",color="purple")

plt.xlabel("views")

plt.ylabel("likes")

plt.title("Views Likes Scatter Plot")

plt.show()
data.views.plot(kind="hist",bins=50,figsize = (10,10), grid = True)

plt.show()
# To create a dictionary

dict={'name':'CRonaldo','team':'Juventus'}

print(dict)
# update existing entry

dict["name"]='Dybala'

print(dict)
# Add new entry

dict["nation"]="Arjantina"

print(dict)
# remove entry with key 'nation'

del dict['nation']

print(dict)
# check include or no

print('name' in dict)
# remove all entries in dict

dict.clear()

print(dict)
# 1 - Filtering Pandas data frame

data[data['likes']>500000]
# Filtering data with 'and' logical

data[(data['views']>500000) & (data['likes']>900000)]
data[data['likes']>5000000][['title','tags']].sort_values("tags",ascending=False)
data['likes'].mean()
data['dislikes'].mean()
data[(data["views"].max())==data["views"]]["title"]
data[data["views"].min()==data["views"]]["title"]
data.sort_values("views",ascending=False).head(10)
data.groupby("category_id").mean().sort_values("likes",ascending=False)["likes"].head(5)
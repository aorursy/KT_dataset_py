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
import matplotlib.pyplot as plt

import seaborn as sns
trend_US=pd.read_csv("/kaggle/input/youtube-new/USvideos.csv")

trend_US.info()

trend_US.head()

trend_US.head(10)

trend_US.columns

trend_US.corr()
f,ax=plt.subplots(figsize=(18,18))

sns.heatmap(trend_US.corr(),annot=True,linecolor="pink",vmin=-1.0,vmax=1.0,linewidths=.5,ax=ax,cmap="coolwarm",fmt=".1f")
plt.scatter(data=trend_US,x="views",y="likes",color="purple",alpha=0.5)

plt.xlabel("Views")

plt.ylabel("Likes")

plt.show()
trend_US["likes"].plot(kind="line",color="green",label="likes",grid=True,linewidth=1,alpha=0.5,linestyle="-")

trend_US["dislikes"].plot(kind="line",color="blue",label="dislikes",grid=True,linewidth=1,alpha=0.5,linestyle=":")

plt.legend(loc="upper left")

plt.xlabel("x axis")

plt.ylabel("y axis")

plt.title("Views and Comments")

plt.show()
trend_US["comments_disabled"].astype("int").plot(kind="hist",bins=30) #I used astype to conver bool to integer

plt.show()



trend_US["ratings_disabled"].astype("int").plot(kind="hist",bins=30)

plt.show()
trend_US[trend_US["comments_disabled"]==True]

trend_US[(trend_US["comments_disabled"]==True)&(trend_US["ratings_disabled"]==True)]
trend_US[trend_US["views"]<10000]

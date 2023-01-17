# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # visualizition tool



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
playerdata = pd.read_csv("../input/fifa19/data.csv")
playerdata.info()
### total = 89 columns about football players 
display(playerdata.head(10))

display(playerdata.columns)

display(playerdata.corr())
f,ax = plt.subplots(figsize=(30,30)) # determine figure size



sns.heatmap(playerdata.corr(), annot = True ,linewidths= .1,fmt = ".1f",ax=ax)

plt.title("Players")

plt.show()

playerdata["Age"].plot(kind= "line",color="blue",label="Age",linewidth = 1, alpha = 1, grid = True,figsize=(12,6), linestyle = ":" )

playerdata["Reactions"].plot(kind= "line",color="red",label="Reactions",linewidth = 1, alpha = 1, grid = True,figsize=(12,6), linestyle = "-." )

plt.show()



playerdata.Age.plot(kind="hist",label="Ages",color = "green",bins= 50,figsize = (15,25))

plt.show()
#filterr # 

morethanthirty = playerdata["Age"]>30

playerdata[morethanthirty]


lessthantwenty = playerdata["Age"]<20

playerdata[lessthantwenty]

wonderkid = playerdata[(playerdata['Age']<25) & (playerdata["Potential"]<85)]

display(wonderkid)



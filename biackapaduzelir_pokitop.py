# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns #visualization tool
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/pokemon-challenge/pokemon.csv")
data.info()
data.head()
data.columns
#correlation map
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt=".1f", ax=ax)
plt.show()
#lineplot
#linewidth = width of line, alpha = opacity, linestyle = style
data.Speed.plot(kind="line",color="g",label="Speed",linewidth=1,alpha=0.5,grid= True,linestyle=":")
data.Defense.plot(color="r",label="Defense",linewidth=1,alpha=0.5,grid=True,linestyle="-.")
plt.legend(loc="upper-right") #legend = puts label into plot
plt.xlabel("x-axis") #label = name of label
plt.ylabel("y-axis") 
plt.title("Line Plot")
plt.show()
#scatter plot
# x= attack y= defense
data.plot(kind="scatter",x="Attack",y="Defense",alpha=0.5,color="red") #plt.scatter(data.Attack,data.Defense,color="red",alpha=0.5) ile aynı islemi yapıyor
plt.xlabel=("Attack")
plt.ylabel=("Defense")
plt.title("Attack Defense Scatter Plot")
plt.show()
#Histogram
#bins= number of bar in figure
data.Speed.plot(kind="hist",bins=50,figsize=(15,15))
plt.show()
#clf() = cleans it up again you can start a fresh
data.Speed.plot(kind="hist",bins=50)
plt.clf()
# We cannot see plot due to clf()
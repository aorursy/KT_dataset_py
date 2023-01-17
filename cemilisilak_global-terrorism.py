# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import codecs
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#First of all we should import our data as I did below this sentence.
data = pd.read_csv("../input/globalterrorismdb_0718dist.csv", encoding="ISO-8859-1", low_memory=False)
# In this section we are checking our work informations: types, memory usage,rangeindex etc. 
data.info()
# This section gives us to numerical values. Should you know;
#std: standard deviation,
#min: the minimum value that we have.
#max: the maximum value that we have.
#mean: the average of the numbers
data.describe()
# this section demonstrates us the first five regions' features. Normally we have more than 5 region but I just would like to show you first five.
data.head(10)
data.info("columns")
dataframe=data[["iyear","imonth","iday","city","attacktype1","targtype1","targsubtype1",
                "weaptype1","nkill","nwound","suicide","success"]]
#correlation map 
#correlation map helps us to understanding beetwen the features relations. It means that we can understand the columns relations with each other thanks to correlation map

f,ax = plt.subplots(figsize=(8,8)) #figure size command
sns.heatmap(dataframe.corr(), annot=False, linewidths=.4, fmt =".1f=", ax=ax) 
# line plot 
# In this figure we can figure out number of kills and number of wounds for every each attacks via line plot diagram.

f,ax = plt.subplots(figsize=(18,5))
data.nkill.plot(kind="line",color = "g", label="number of kills",linewidth=5, alpha=0.5, grid=True, linestyle=":")
data.nwound.plot(color="r", label="number of wounds",linewidth=5, alpha=0.5, grid=True,linestyle=":")
plt.legend(loc="upper right")
plt.xlabel("Number of attacks", size=15)
plt.ylabel("Person", size=15)
plt.title("Line Plot")


#Scatter Year-Kill Scatter Plot
#In this section we will find a relation between two features(columns) (The relation between in which years and kills)

data.plot(kind="scatter", x="iyear", y="nkill",grid=True, alpha=0.5, color="b",figsize=(18,5))
plt.xlabel("year", size= 25 )
plt.ylabel("kills", size= 25) # size command helps you to change for labels size.
plt.title("Year-Kill Scatter Plot")
# In this section we are analyzing frequency of target types
data.targsubtype1.plot(kind= "hist",bins=300, figsize= (18,5))
plt.xlabel("Types of Target (kind : number) ", size= 20 )
plt.ylabel("Frequency", size= 20)

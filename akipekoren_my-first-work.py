# In this code, we will examine the detailed information of Fifa 19 players


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
myDataFrame=pd.read_csv("../input/data.csv")   # import file
myDataFrame.info()   # to see how many string-float-int-etc
myDataFrame.columns  # to see columns
myDataFrame.head(10)  # to show first 10 row
myDataFrame.corr()  # correlation
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(myDataFrame.corr(), annot=True, linewidths=.8, fmt= '.01f',ax=ax)
plt.show()
myDataFrame.Finishing.plot(kind="line" , color ="blue" , label = "Finishing" , alpha=0.5 , grid=True)
myDataFrame.Crossing.plot(kind="line" , color ="green" , label = "Crossing" , alpha=0.5, grid=True)
plt.xlabel("x")     # two different feature is shown in the graph
plt.ylabel("y")
plt.title("line graph")
plt.show()
myDataFrame.plot(kind="scatter" , x="Age", y ="Overall", color="red", alpha=0.8)
plt.xlabel("Age")      #Age-Overall graph
plt.ylabel("Overall")
plt.title("Scatter plot")
plt.show()
myDataFrame.Age.plot(kind="hist" , bins=50 , figsize=(15,15)) #histogram graph

filter1=myDataFrame["Age"]<20   # first filter
filter2=myDataFrame["Overall"]>80  # second filter
youngPlayerData= myDataFrame[filter1 & filter2]  # create a new dataframe
youngPlayerData.head()

youngPlayerData.Age.plot(kind="hist",bins=10 , figsize=(8,8) ) #newdataframe's histogram graph

myDataFrame[myDataFrame["Club"]=="FC Barcelona"]  #another example of filtering
myDataFrame[myDataFrame["Club"]=="FC Barcelona"].Vision .plot(kind="line" , color ="blue" , label = "Vision " , alpha=0.5 , grid=True)
myDataFrame[myDataFrame["Club"]=="FC Barcelona"].Marking.plot(kind="line" , color ="green" , label = "Marking" , alpha=0.5, grid=True)
plt.xlabel("x")     
plt.ylabel("y")
plt.title("line graph")
plt.show()

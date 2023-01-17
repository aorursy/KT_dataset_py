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
data = pd.read_csv("../input/cwurData.csv")
data.info()
data.corr() #for correlation 
data.describe()
#Correlation
f,ax = plt.subplots(figsize=(15,15))
sns.heatmap(data.corr(),annot=True,linewidths=5,fmt=".2f",ax=ax)
plt.show()
#Line Plot
data.quality_of_education.plot(kind="line",label="Quality of Education",color="blue",linewidth=1,alpha=0.5,grid=True,linestyle=":")
data.quality_of_faculty.plot(color="red",label="Quality of Faculty",linewidth=1,alpha=0.5,grid=True,linestyle="-.")
plt.legend("upper right")
plt.xlabel("Quality of Education")
plt.ylabel("Quality of Faculty")
plt.title("Quality of Education and Faculty (Line)")
plt.show()
data.plot(kind="scatter",x="world_rank",y="national_rank",alpha=0.5,color="red")
plt.xlabel("World Rank")
plt.ylabel("National Rank")
plt.title("World Rank and National Rank (Scatter)")
plt.legend()
plt.show()
data.citations.plot(kind="hist",bins=50,figsize=(15,15),color="red")
x = data["citations"] > 800 #more than 800 students
data[x].institution
data["country"].value_counts()
data[data["score"].max()==data["score"]]["institution"].iloc[0]
data[data["alumni_employment"].max()==data["alumni_employment"]]["institution"]
x=data[(data["world_rank"]>500) & (data["quality_of_education"]>300) & (data["country"]=="Turkey")]["institution"].unique()
for i in x:
    print(i)
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
import matplotlib.pyplot as plt
import seaborn as sns
lebrondata=pd.read_csv("../input/lebron_career.csv")
jordandata=pd.read_csv("../input/jordan_career.csv")
lebrondata.info()
jordandata.columns
lebrondata.describe()
jordandata.describe()
jordandata.pts.describe()

lebrondata.pts.describe()
jordandata.ast.mean()
lebrondata.ast.mean()

lebrondata.head(100).pts.mean()  
#jordan points mean  for first 100 match 
jordandata.head(100).pts.mean()  
lebrondata.corr()
buyutme=plt.subplots(figsize=(20,20))
sns.heatmap(lebrondata.corr(),annot=True,linewidth=1,fmt='.1f')

lebrondata.three.plot(kind="line",color="yellow",linewidth=1,alpha=1,grid=True,label="LeBron three points",figsize=(10,10))
jordandata.three.plot(kind="line",color="red",linewidth=1,alpha=1,grid=True,label="Jordan three points",figsize=(10,10))
plt.legend(loc='upper right') 
plt.xlabel('match')
plt.ylabel('three points')
plt.title('LeBRON23 VS JORDAN23 for three points') 
plt.show()

lebrondata.plot(kind="scatter",x="fg",y="pts",color="black",figsize=(10,10))
plt.xlabel("fg")
plt.ylabel("pts")
plt.title("fg-pts scatter plot")
plt.show()
jordandata.plot(kind="scatter",x="fg",y="pts",color="black",figsize=(10,10))
plt.xlabel("fg")
plt.ylabel("pts")
plt.title("fg-pts scatter plot")
plt.show()
lebrondata.trb.plot(kind="hist",bins=30,color="black",figsize=(15,15))
jordandata.trb.plot(kind="hist",bins=30,color="red",figsize=(15,15))
plt.xlabel("rebound")
plt.legend(loc='upper right')

lebrondata.drb.plot(kind="hist",bins=30,color="blue",figsize=(15,15))
lebrondata.orb.plot(kind="hist",bins=30,color="red",figsize=(15,15))
plt.title("James offensive rebound and James defensive rebound histogram  defensive rebound blue -offensive rebound red ")
plt.legend(loc='upper right')
plt.show
x1=lebrondata["pts"]>=50
lebrondata[x1]

x2=jordandata["pts"]>=50
jordandata[x2]
jordandata[(jordandata["pts"]>35) & (jordandata["ast"]>10) & (jordandata["trb"]>10)]

lebrondata[(lebrondata["pts"]>35) & (lebrondata["ast"]>10) & (lebrondata["trb"]>10)]
lebrondata[(lebrondata["team"]=="CLE") & (lebrondata["fg"]>15)]

lebrondata[(lebrondata["team"]=="MIA") & (lebrondata["fg"]>15)]
jordandata[jordandata["fg"]>15]
lebrondata.fg.plot(kind="line",color="black",alpha=0.5,linewidth=2,figsize=(18,18))
jordandata.fg.plot(kind="line",color="red",alpha=0.5 ,linewidth=2,figsize=(18,18))
plt.xlabel("match")
plt.ylabel("fg")
plt.title("LeBron vs Jordan for fg ")
plt.legend()
lebrondata.ast.plot(kind="line",color="black",alpha=0.5,linewidth=2,figsize=(18,18))
jordandata.ast.plot(kind="line",color="red",alpha=0.5 ,linewidth=2,figsize=(18,18))
plt.xlabel("match")
plt.ylabel("asist")
plt.title("LeBron vs Jordan for asist plot Lebron is black Jordan is red")
plt.legend()
x3=lebrondata["team"]=="CLE"
lebrondata[x3].pts.mean()

# x4=lebrondata["team"]=="MIA"
# lebrondata[x4].pts
print(lebrondata.team.value_counts(dropna=False))
print(jordandata.pts.value_counts(dropna=False))
print(lebrondata.pts.value_counts(dropna=False))
lebrondata.boxplot(column="pts",by="ast")
jordandata.boxplot(column="pts",by="ast")
lebrondata.boxplot(column="pts",by="fg")
jordandata.boxplot(column="pts",by="fg")
lebrondata_new=lebrondata.head()


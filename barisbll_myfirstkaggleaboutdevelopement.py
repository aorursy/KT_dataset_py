# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/Country.csv")
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(),annot=True,linewidths=.5,fmt=".1f",ax=ax)
plt.show()
data.head(10)
data.columns
#Line Plot
# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line
data.LatestIndustrialData.plot(kind ="line",color="g",label="Latest Industrial Data",linewidth=1,alpha=0.5,grid=True,lineStyle=":")
data.LatestTradeData.plot(color="r",label="Latest Trade Data",linewidth=1,alpha=0.5,grid=True,lineStyle="-.")

plt.legend(loc="upper right") #legend = put Label into Plot
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Line Plot")
plt.show()
# Scatter Plot
# x= Industrial Data y= Trade Data

data.plot(kind="scatter",x="LatestIndustrialData",y="LatestTradeData",alpha=0.5,color="red")
plt.xlabel("Industrial Data")
plt.ylabel("Trade Data")
plt.title("Industrial Data Trade Date Scatter Plot")
#Histogram
#bins = number of bars in the figure

data.Speed.plot(kind="hist",bins=50,figsize=(12,12))
plt.show()
# clf() = cleans it up again you can start a fresh project

data.Speed.plot(kind="hist",bins=50)
data.clf()
#We cannot see plot due to clf()
#Create dictionary and look its keys and values
dictionary={"spain":"madrid","usa":"vegas"}
print(dictionary.keys())
print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles
# List is not immutable
# Keys are unique

dictionary["spain"] = "barselona"
print(dictionary)
dictionary["france"] = "paris"
print(dictionary)
del dictionary["spain"]
print(dictionary)
print("france" in dictionary)
dictionary.clear()
print(dictionary)
# In order to run all code you need to take comment this line
#del dictionary         # delete entire dictionary     
print(dictionary)       # it gives error because dictionary is deleted
data = pd.read_csv("../input/Country.csv")
data.head(20)
series= data["ShortName"]
print(type(series))
data_frame=data["TableName"]
print(type(data_frame))
# Comparison operator
print(3 > 2)
print(3!=2)
# Boolean operators
print(True and False)
print(True or False)
# 1 - Filtering Pandas Data Frame
x= data["LatestWaterWithdrawalData"]==2000
data[x]
# 2 - Filtering pandas with logical_and

data[np.logical_and(data["NationalAccountsBaseYear"]>2005,data["PppSurveyYear"]<2012)]









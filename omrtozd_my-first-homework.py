### This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/globalterrorismdb_0617dist.csv',encoding='ISO-8859-1')
data.info()
data.columns
data.head(10)
data.corr()
f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(data.corr(),cmap="YlGnBu", linewidths=.5,ax=ax)
plt.show()
terror = data[["eventid","iyear","imonth","iday","extended","country","country_txt","region","region_txt","city","specificity","nkill","weaptype1","weaptype1_txt","nwound","nhours","ndays","suicide","attacktype1","attacktype1_txt"]]
terror.rename(columns={"iyear":"year","imonth":"month","iday":"day","nkill":"kill","nwound":"wound","attacktype1":"attacktype","attacktype1_txt":"attacktype_txt","weaptype1":"weaptype","weaptype1_txt":"weaptype_txt","nhours":"hours","ndays":"days"},inplace=True)
terror['kill'].astype('int')
#Line Plot
terror.year.plot(kind="line",x="year",color="red")
plt.grid()
plt.show()


country_filter = data[data["country"] == 209]
data.iyear.plot(kind="hist", bins = 40,figsize = (10,10))
plt.xlabel("Years")
plt.ylabel("Teror Events")
plt.title("Terrorist Events by Years Between 1970-2016")
plt.grid()
plt.show()
plt.figure(figsize=(30,60))
sns.countplot(data=data,y = data["country_txt"])
plt.show()
plt.figure(figsize=(30,10))
sns.countplot(data=data,y = data["attacktype1_txt"])
plt.grid()
plt.show()

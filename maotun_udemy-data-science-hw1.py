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
import pandas as pd
import numpy as np
import seaborn as sns
data = pd.read_csv("../input/2017.csv")
data
data.info()
data.describe()
hrank = data["Happiness.Rank"]
hscore = data["Happiness.Score"]
data.plot()
plt.show()
data.plot(x="Happiness.Rank",y="Happiness.Score",label = "Rank - Score",title = "World Happiness Report",grid = True,figsize=[13,13])
plt.legend()
plt.xlabel("Happiness Rank")
plt.ylabel("Happiness Score")
plt.show()

plt.plot(data["Happiness.Rank"],data["Happiness.Score"],label = "Rank - Score")
plt.legend()
plt.xlabel("Happiness Rank")
plt.ylabel("Happiness Score")
plt.title("World Happiness Report")
plt.show()

plt.plot(data["Happiness.Rank"],data["Happiness.Score"],label = "Happines Rank")
plt.plot(data["Happiness.Rank"],data["Freedom"],label = "Happines Score")
plt.title("Graph")
plt.legend()
plt.show()
plt.subplots(figsize = [13,13])
plt.scatter(data["Happiness.Rank"],data["Happiness.Score"],color = "green",label="Happines Score")
plt.scatter(data["Happiness.Rank"],data["Freedom"],color = "red",label ="Freedom")
plt.title("Happines Rank-Fredom-Happines Score")
plt.legend()
plt.show()
filtered_data =data.Country[:10]
filtered_data2 = data['Trust..Government.Corruption.'][:10]
plt.bar(filtered_data,filtered_data2,width = 0.80,label="Countries - Trust")
plt.xlabel("Countries")
plt.ylabel("Trusr..Goverment")
plt.title("Trusting to Goverment on the World")
plt.legend()
plt.show()
filtered_data2 = data['Trust..Government.Corruption.'][:10]
filtered_data2.plot(kind = "bar")
plt.xlabel("Countries")
plt.ylabel("Trust..Goverment")
plt.title("Graph")
plt.show()
plt.hist(data.Family,rwidth=0.9,label = "Family")
plt.xlabel("Family")
plt.ylabel("# of Family")
plt.legend()
plt.show()
plt.hist([data["Whisker.low"],data["Whisker.high"]],color = ["green","orange"],rwidth = 0.9,label =["Whisker high","Whisker low"] )
plt.legend()
plt.xlabel("whisker low and high")
plt.show()
plt.hist([data["Whisker.low"],data["Whisker.high"]],color = ["green","orange"],rwidth = 0.9,label =["Whisker high","Whisker low"],orientation="horizontal")
plt.legend()
plt.xlabel("whisker low and high")
plt.show()
plt.subplot(3,1,1)
plt.plot(data["Happiness.Rank"],data["Happiness.Score"],label = "Happines Rank")
plt.title("Graph")
plt.grid()
plt.legend()
plt.subplot(3,1,2)
plt.plot(data["Happiness.Rank"],data["Freedom"],label = "Happines Score",color = "blue")
plt.grid()
plt.legend()
plt.subplot(3,1,3)
plt.plot(data["Happiness.Rank"],data["Trust..Government.Corruption."],label = "Trust..Government.Corruption.",color = "red")
plt.grid()
plt.legend()
plt.show()
plt.pie(data.Freedom[:9],labels= data.Country[:9])
plt.show()
plt.axis("equal")
plt.pie(data.Generosity[:7],labels= data.Country[:7],radius = 2,autopct="%0.0f%%",shadow = True)
plt.show()
plt.axis("equal")
plt.pie(data.Generosity[:7],labels= data.Country[:7],radius = 2,autopct="%0.0f%%",shadow = True,explode=[0.2,0,0,0,0,0,0])
plt.show()

plt.axis("equal")
plt.pie(data.Generosity[:7],labels= data.Country[:7],radius = 2,autopct="%0.0f%%",shadow = True,explode=[0,0.4,0,0,0,0,0])
plt.show()
plt.axis("equal")
plt.pie(data.Generosity[:7],labels= data.Country[:7],radius = 2,autopct="%0.0f%%",shadow = True,explode=[0,0.4,0,0,0,0,0],startangle = 180)
plt.show()
data.corr()
f,ax = plt.subplots(figsize=(13,13))
sns.heatmap(data.corr(), annot=True, linewidths=.6, fmt= '.1f',ax=ax)
plt.show()
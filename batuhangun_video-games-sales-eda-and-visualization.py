import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict,OrderedDict
from operator import itemgetter

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings("ignore")
data = pd.read_csv("../input/vgsales.csv")
data.info()
data.head(10)
df = pd.DataFrame(index = data.Publisher.unique(),data = data.Publisher.value_counts())
df.columns=["Total Games"]
df.sort_values("Total Games",ascending =False).head(10)
dic = defaultdict(list)

for i in data.Genre.unique():
    gs = data[data.Genre == i].Global_Sales
    genre = data[data.Genre == i].Genre
    for k,v in zip(list(genre),list(gs)):
        dic[k].append(v)
        

plt.figure(figsize=(20,7))        
sns.barplot(x = [k for k,v in dic.items()],y = [sum(v)/len(v) for k,v in dic.items()])
plt.xlabel("Genre")
plt.ylabel("Avg Global Sales")
plt.show()
dic = defaultdict(list)

for i in data.Publisher.unique():
    name = data[data.Publisher == i].Publisher
    gs = data[data.Publisher == i].Global_Sales
    for k,v in zip(list(name),list(gs)):
        dic[k].append(v)

orderList = {}
for k,v in dic.items():
    orderList[k]=(sum(v))

orderList = OrderedDict(sorted(orderList.items() ,key=itemgetter(1),reverse=True))

plt.figure(figsize=(20,7))        
sns.barplot(x = [k for k,v in list(orderList.items())[:10]],y = [v for k,v in list(orderList.items())[:10]])
plt.xlabel("Publisher")
plt.ylabel("Total Global Sales")
plt.show()

df = data[["Publisher","Global_Sales"]]
d={}

for i in df.Publisher.unique():
    x = df[df.Publisher ==i].Global_Sales    
    if sum(x) != 0:
        d[i] = sum(x) / len(x)

df2 = pd.DataFrame(index = [k for k,v in d.items()], data = [v for k,v in d.items()])
df2.columns = ["average sales for each game"]
df2.sort_values("average sales for each game", ascending = False).head(10)
dfJP = data.sort_values("JP_Sales",ascending=False)[["Name","Year"]].head(10)
dfEU = data.sort_values("EU_Sales",ascending=False)[["Name","Year"]].head(10)
dfNA = data.sort_values("NA_Sales",ascending=False)[["Name","Year"]].head(10)

dfJP.index = range(1,11)
dfEU.index = range(1,11)
dfNA.index = range(1,11)

df = pd.DataFrame(data=dfJP)
df.columns=["Japan Top 10 ","Japan Year"]
df["European Top 10"] = dfEU.Name
df["European Year"] = dfEU.Year
df["North American Top 10"] = dfNA.Name
df["North American Year"] = dfNA.Year
df
plt.figure(figsize=(20,10))
sns.swarmplot(x = data[data.Year == 2009.0].Genre , y = data[data.Year == 2009.0].Global_Sales, hue = data.Platform)
plt.title("Sales Genres by Platforms in 2009")
plt.show()
na = data[data.Publisher == "Bethesda Softworks"].NA_Sales
eu = data[data.Publisher == "Bethesda Softworks"].EU_Sales
jp = data[data.Publisher == "Bethesda Softworks"].JP_Sales
ot = data[data.Publisher == "Bethesda Softworks"].Other_Sales

labels = ["North American","European","Japan","Other"]
explode = [0,0,0,0]
sizes = [sum(na),sum(eu),sum(jp),sum(ot)]

plt.figure(figsize = (8,8))
plt.pie(sizes,labels = labels,explode=explode,autopct='%1.1f%%')
plt.title("Bethesda Market Share")
plt.show()
group = data[["Year","Platform"]].groupby("Platform")

plt.figure(figsize=(20,7))        
sns.barplot(x=[k for k,v in group], y = [len(v) for k,v in group])
plt.xlabel("Platform")
plt.ylabel("Total Games")
plt.title("Total game by platforms")
plt.show()
plt.figure(figsize=(20,7))
sns.lineplot(x = data[data.Publisher == "SquareSoft"].Name, y = data.EU_Sales,label="EU Sales",color="blue",linewidth = 2.5)
sns.lineplot(x = data[data.Publisher == "SquareSoft"].Name, y = data.JP_Sales,label="JP Sales",color ="red",linewidth = 2.5)
sns.lineplot(x = data[data.Publisher == "SquareSoft"].Name, y = data.NA_Sales,label="NA Sales",color="black",linewidth = 2.5)
plt.xticks(rotation="90")
plt.legend()
plt.ylabel("Sales")
plt.title("LucasArts Games European/Japan/North American Sales")
plt.show()
plt.figure(figsize=(15,7))
sns.lineplot(x=data.Year,y=data.Global_Sales)
plt.title("sales by years")
plt.show()
palette = sns.color_palette("Paired")
plt.figure(figsize=(15,7))
sns.lineplot(x=data.Year,y=data[data.Platform=="DS"].Global_Sales,hue=data[data.Publisher=="Activision"].Genre,err_style=None,palette=palette)
plt.show()
df = data[(data.Year > 2000.0) & ((data.Publisher =="Nintendo")|(data.Publisher == "Activision")|(data.Publisher == "Electronic Arts"))]


plt.figure(figsize=(20,15))

plt.subplot(221)
sns.lineplot(y = df.Global_Sales,x = df.Year,hue=df.Publisher,err_style=None)
plt.title("Global Sales")

plt.subplot(222)
sns.lineplot(y = df.EU_Sales,x = df.Year,hue=df.Publisher,err_style=None)
plt.title("European Sales")

plt.subplot(223)
sns.lineplot(y = df.NA_Sales,x = df.Year,hue=df.Publisher,err_style=None)
plt.title("North American Sales")

plt.subplot(224)
sns.lineplot(y = df.JP_Sales,x = df.Year,hue=df.Publisher,err_style=None)
plt.title("Japan Sales")

plt.show()

import numpy as np 
import pandas as pd 
from collections import defaultdict,OrderedDict
from operator import itemgetter

import matplotlib.pyplot as plt
import seaborn as sns 

import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))
data = pd.read_csv("../input/googleplaystore.csv")
data.info()
data.head()
data.describe()
data.sort_values("Rating",ascending=False).head()
overline = data[data.Rating == data.Rating.sort_values(ascending = False).max()] 
print(overline)
data.drop(index = overline.index, inplace=True)
group = data[["Genres","Category"]].groupby("Category")
dic = defaultdict(list)
for i in data.Category.unique():
    count = group.get_group(i).Genres.unique()
    groupName = i
    dic[groupName].append(count)

pd.DataFrame(index=[k for k,v in dic.items()],data=[[len(i) for i in v] for k,v in dic.items()],columns=["count"])

pd.DataFrame({"Count":dic})
data.Rating = data.Rating.fillna(0)
category = data[["Rating","Category"]].groupby("Category")

dic = defaultdict(list)
for i in data.Category.unique():
    rating = category.get_group(i).Rating
    
    dic[i].append([i  for i in rating])

df = pd.DataFrame(index = [k for k,v in dic.items()], data = [[sum(i)/len(i) for i in v]  for k,v in dic.items()],columns=["Avg. Rating"])
df
plt.figure(figsize = (25,7))
sns.barplot(data.Category.value_counts().index,data.Category.value_counts().values)
plt.xlabel("Category")
plt.ylabel("Counts")
plt.xticks(rotation = 90)
plt.show()
plt.figure(figsize = (25,10))
sns.swarmplot(x = data.Category, y = data.Rating, hue = data.Type)
plt.xticks(rotation = 90)
plt.show()
df = data[(data.Category == "HEALTH_AND_FITNESS") & (data.Price == "0")&([len(i) > 10 for i in data.Installs.values])].sort_values("Rating",ascending=False).head(20)

plt.figure(figsize = (20,7))
sns.barplot(x = df.App , y = df.Rating)
plt.xticks(rotation = 90)
plt.show()
d = defaultdict(list)

for i in data.Rating.unique():
    rating = data[data.Rating == i].Rating
    reviews = data[data.Rating == i].Reviews.astype(int)
    for k,v in zip(list(rating),list(reviews)):
        d[k].append(v)

plt.figure(figsize=(20,7))
sns.barplot(x = [k for k,v in d.items()], y = [sum(v)/len(v) for k,v in d.items()])
plt.xlabel("Rating")
plt.ylabel("Avarage Reviews")
plt.show()
di = defaultdict(list)
                 
for i in data["Content Rating"].unique():
    content = data[data["Content Rating"] == i]["Content Rating"]
    count =  data[data["Content Rating"] == i].Installs
    for k,v in zip(list(content),list(count)):
        di[k].append(int(v.replace(",","").split("+")[0]))
        
                         
labels = []
explode = [0,0,0,0,0,0]
sizes = []

for i in zip([k for k,v in di.items()],[sum(v)/len(v) for k,v in di.items()]):
    sizes.append(i[1])
    labels.append(i[0])
    
plt.figure(figsize = (8,8))
plt.pie(sizes,labels = labels,explode=explode,autopct='%1.1f%%')
plt.show()
data.Price = data.Price.apply(lambda x:x.replace("$","")).astype(float)
data.sort_values("Price",ascending=False).head(10)
d = defaultdict(list)

for i in data.Category.unique():
    category = data[data.Category == i].Category
    price = data[data.Category == i].Price
    for k,v in zip(list(category),list(price)):
        d[k].append(v)


orderList = dict()
for k,v in d.items():
   orderList[k]=(sum(v)/len(v))

orderList = OrderedDict(sorted(orderList.items() ,key=itemgetter(1),reverse=True))

plt.figure(figsize = (20,7))
sns.barplot([k for k,v in orderList.items()],[v for k,v in orderList.items()])
plt.xlabel("Category")
plt.ylabel("Average Price")
plt.xticks(rotation = 90)
plt.show()
plt.figure(figsize = (20,10))
sns.lineplot(x = data[data.Rating >= 3.0].Rating,y = data[data.Price != 0.0].Price, hue = data.Category,palette = sns.color_palette("Paired",len(data.Category.unique())))
plt.xlabel("Rating")
plt.ylabel("Price")
plt.show()
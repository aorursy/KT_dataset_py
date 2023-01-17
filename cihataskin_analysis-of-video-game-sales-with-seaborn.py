# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from collections import Counter

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/vgsales.csv',encoding="windows-1252")
data.info()
data.head()
data.shape
data.isnull().sum()
# marking pokemon games

data["Pokemon"]=[True if each.startswith("Pokemon") else False for each in data.Name]#creating new column

data1=data[data.Pokemon]#pokemon games' data
print(data1.columns)
data1.head()
best=data1.nlargest(5,'Global_Sales')#best 5 pokemon games
sns.set(font_scale=1.5)

plt.figure(figsize=(22,10))

sns.barplot(x=best.Name, y=best.Global_Sales)

plt.xticks(rotation= 20)

plt.xlabel('Pokemon Games')

plt.ylabel('Millions')

plt.title('Best 5 Pokemon Games Sales')
len(data["Publisher"].unique())
rpgames=data[data["Genre"]=="Role-Playing"]#only role playing game data

publisher=list(rpgames["Publisher"].unique())# and its publishers



rpg_publishers=[]

rpg_sales=[]



for i in publisher:

    x=rpgames[rpgames["Publisher"]==i]

    if len(x)!=0:

        sales_ratio=sum(x["Global_Sales"])/len(x)

        rpg_sales.append(sales_ratio)

        rpg_publishers.append(i)#Maybe no need

rpgData=pd.DataFrame({"Publisher":rpg_publishers, "Sales":rpg_sales})#converting dataframe

sorting=rpgData.nlargest(30,'Sales')#selecting best 30 rpg sales ratio
plt.figure(figsize=(15,10))

ax= sns.barplot(x=sorting.Publisher, y=sorting.Sales,palette = sns.cubehelix_palette(len(sorting)))

plt.xticks(rotation= 90)

plt.xlabel('RPG Publisher')

plt.ylabel('Millions')

plt.title('Mean of RPG Sales in The World')
#Game Sales for platform

platforms=list(data["Platform"].unique())

str(platforms)

na=[]

ua=[]

jp=[]

other=[]

glbl=[]



for i in platforms:

    x=data[data["Platform"]==i]

    

    na.append(sum(x["NA_Sales"]))

    ua.append(sum(x["EU_Sales"]))

    jp.append(sum(x["JP_Sales"]))

    other.append(sum(x["Other_Sales"]))

    glbl.append(sum(x["Global_Sales"]))

 

valuablePlatform=pd.DataFrame({"Platform":platforms, "NA":na, "UA":ua, "JP":jp, "Other":other, "Global":glbl})
valuablePlatform.head()
f, ax = plt.subplots(figsize=(15, 15))



sortedData =valuablePlatform.sort_values("Global", ascending=False)



sns.set_color_codes("pastel")

sns.barplot(x="Global", y="Platform", data=sortedData, label="Global Sales", color="r")



sns.set_color_codes("muted")

sns.barplot(x="UA", y="Platform", data=sortedData, label="European Sales", color="b",alpha=0.5)



sns.set_color_codes("muted")

sns.barplot(x="NA", y="Platform", data=sortedData, label="North American Sales", color="r",alpha=0.5)



sns.set_color_codes("muted")

sns.barplot(x="JP", y="Platform", data=sortedData, label="Japan Sales", color="g",alpha=0.3)



# Add a legend and informative axis label

ax.legend(ncol=2, loc="center", frameon=True)

ax.set(xlim=(0, 1300), ylabel="Platforms", xlabel="Million")

sns.despine(left=True, bottom=True)
sns.pairplot(valuablePlatform,diag_kind="kde",kind="reg",palette="husl")
dataYear=data["Year"].dropna().unique()#cleaning data

dataYear.sort()
na=[]

eu=[]

jp=[]

other=[]

glbl=[]



for i in dataYear:

    x=data[data["Year"]==i]

    

    na.append(sum(x["NA_Sales"]))

    eu.append(sum(x["EU_Sales"]))

    jp.append(sum(x["JP_Sales"]))

    other.append(sum(x["Other_Sales"]))

    glbl.append(sum(x["Global_Sales"]))

 

yearSales=pd.DataFrame({"Year":dataYear, "NA":na, "EU":eu, "JP":jp, "Other":other, "Global":glbl})

yearSales["Year"].astype("int64")



yearSales.info()
yearSales["Year"]=yearSales.Year.astype("int64")



f,ax1 = plt.subplots(figsize =(20,10))

sns.pointplot(x="Year",y="NA" ,data=yearSales,color='lime',alpha=0.8)

sns.pointplot(x="Year",y="EU",data=yearSales,color='gold',alpha=0.8)

sns.pointplot(x="Year",y="JP",data=yearSales,color='purple',alpha=0.8)

sns.pointplot(x="Year",y="Global",data=yearSales,color='red',alpha=0.8)

plt.text(10,670,'Annual Global Sales',color='red',fontsize = 18,style = 'italic')

plt.text(10,640,'Annual NA Sales',color='lime',fontsize = 18,style = 'italic')

plt.text(10,610,'Annual EU Sales',color='gold',fontsize = 18,style = 'italic')

plt.text(10,580,'Annual Japan Sales',color='purple',fontsize = 18,style = 'italic')

plt.xticks(rotation=90)

plt.xlabel('Years',fontsize = 20,color='blue')

plt.ylabel('Sales',fontsize = 20,color='blue')

plt.title('ANNUAL SALES',fontsize = 20,color='blue')

plt.grid()
g=sns.jointplot(yearSales.JP,yearSales.EU, kind="kde",size=8)

plt.savefig("graph.png")

plt.show()
sns.lmplot(x="NA",y="EU", data=yearSales)

plt.show()
sns.kdeplot(yearSales.JP,yearSales.NA,shade=True,cut=5)
f,ax2=plt.subplots(figsize=(13,10))

sns.heatmap(yearSales.corr(),annot=True,linecolor="green", linewidths=.5,center=1.2, fmt=".1f",ax=ax2)

plt.show()
sales=[sum(yearSales["NA"]),sum(yearSales["EU"]),sum(yearSales["JP"]),sum(yearSales["Other"])]

explode=[0,0,0,0]

labels=["NA Sales","EU Sales","JP Sales","Other Sales"]

colors=["lime","gold","purple","gray"]



plt.figure(figsize=(7,7))

plt.pie(sales, explode=explode, labels=labels, colors=colors, autopct="%1.1f%%")

plt.title("Global Sales", color="blue",fontsize=18)
f,ax2=plt.subplots(figsize=(15,7))

sns.violinplot(x="Genre",y="Year", data=data, inner="points", scale="width")

plt.xticks(rotation=90)

plt.show()
f,ax2=plt.subplots(figsize=(10,7))

data_nintendo=data[data["Publisher"]=="Nintendo"]

sns.boxplot(x="Genre",y="Year",data=data_nintendo,palette="PRGn")

plt.xticks(rotation=90)
sns.countplot(data_nintendo.Genre)

plt.xticks(rotation=90)

plt.ylabel("Count of Games")

plt.title("Genres of Nintendo Games",color="black",fontsize=15)
f,ax2=plt.subplots(figsize=(20,10))

platform_count=Counter(data["Platform"])

most=platform_count.most_common(12)

firstTwelvePlatform,val=zip(*most)# return tuple

firstTwelvePlatform=list(firstTwelvePlatform)



data6=data[data["Platform"].isin(firstTwelvePlatform)]

data6=data6[(data6["Genre"]=="Action") | (data6["Genre"]=="Strategy")]



sns.swarmplot(x="Platform", y="Year",hue="Genre",data=data6)



plt.xticks(rotation=90)
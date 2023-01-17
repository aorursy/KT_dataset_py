
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#seaborn
import seaborn as sns
#matplot
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import warnings            
warnings.filterwarnings("ignore")
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")
data.head()
data.info()
data.isnull().any() #look for nan values
data.isna().sum() # sum of nan values
data.dropna(axis=0,inplace=True) #dropping nan values
def div(n):
    return n/10
data.Critic_Score=data.Critic_Score.apply(div) # div 10 to Critic score to make similiar User score

data["Year_of_Release"]=data["Year_of_Release"].astype("int") # changing data types
data["User_Count"]=data["User_Count"].astype("int")
data["Critic_Count"]=data["Critic_Count"].astype("int")
data["User_Score"]=data["User_Score"].astype("float")
data=data.reset_index(drop=True)
data.head(10)
# Plot for games' count on each platform (classic bar plot)
x=data.Platform.unique()
y=data.Platform.value_counts()
plt.figure(figsize=(16,9))
sns.barplot(x=x,y=y,edgecolor="black")
plt.xlabel("Platforms")
plt.ylabel("Games count")
plt.show()
# Sequential bar plot for games' genre count
x=data.Genre.unique()
y=data.Genre.value_counts().sort_values(ascending=True)
plt.figure(figsize=(16,9))
sns.barplot(x=x,y=y,palette="rocket")
plt.xlabel("Genres")
plt.ylabel("Games count")
plt.show()
top_publisher=pd.DataFrame(data.Publisher.value_counts()[:10]) # top 10 game publishers
top_publisher
top_publisher.index
# Horizontal bar plot for top 10 game publishers sale
publisher_list=list(top_publisher.index)

NA_Sales = []
EU_Sales = []
JP_Sales = []
Other_Sales = []

# Groupped sales on 4 type

for i in publisher_list:
    x = data[data.Publisher==i]
    NA_Sales.append(sum(x.NA_Sales))
    EU_Sales.append(sum(x.EU_Sales))
    JP_Sales.append(sum(x.JP_Sales))
    Other_Sales.append(sum(x.Other_Sales))

# visualization
f,ax = plt.subplots(figsize = (16,9))
sns.barplot(x=NA_Sales,y=publisher_list,label='NA',color="r" )
sns.barplot(x=EU_Sales,y=publisher_list,label='EU',color="b")
sns.barplot(x=JP_Sales,y=publisher_list,label='JP',color="g")
sns.barplot(x=Other_Sales,y=publisher_list,label='Other',color="y")

ax.legend(loc='lower right',frameon = True)     
ax.set(xlabel='Total Sales', ylabel='Publisher',title = "Total Sales According to Publishers ")
plt.show()
release_sales=data.groupby("Year_of_Release")["Global_Sales","Name"].max() # max sales in each year
release_sales
# max gloabal sales in each year
sns.set(style="whitegrid")
f,ax=plt.subplots(figsize=(16,9))
sns.pointplot(x=release_sales.index,y=release_sales.Global_Sales,alpha=0.5,color="red")
plt.text(21,82,'Sales(millions)',fontsize = 17,style = 'italic')
plt.xlabel('Year of Release',fontsize = 15)
plt.ylabel('Max Sales',fontsize = 15)
plt.title('Max Sales of Each Year',fontsize = 20)
plt.xticks(rotation= 45)
plt.show()
data.corr()
# kde jointplot
g=sns.jointplot(x="User_Score",y="Critic_Score",kind="kde",size=10,space=0,data=data)
# grid jointplot
grid = sns.jointplot(x=data.User_Score, y=data.Critic_Score, space=0, size=10, ratio=5,color="g")
ratings=pd.DataFrame(data.Rating.value_counts()) 
ratings
labels=ratings[:4].index # I ignored last 3 value because too small
colors = ['grey','blue','red','yellow']
sizes = ratings[:4].values

# visual
plt.figure(figsize = (9,9))
plt.pie(sizes, labels=labels, colors=colors, autopct='%1.2f%%')
plt.title("Games' Ratings",color = 'blue',fontsize = 15)
plt.show()
sns.lmplot(x="User_Count", y="Critic_Count", data=data,size=10,)
plt.show()
f, ax = plt.subplots(figsize=(10, 10))
ax = sns.kdeplot(data.User_Score, data.Critic_Score,cmap="Reds", shade=True, shade_lowest=False)
red = sns.color_palette("Reds")[-2]
ax.text(3.8, 4.5, "Score", size=16, color="b")
plt.show()
f, ax = plt.subplots(figsize=(20, 10))
pal = sns.hls_palette(2, l=.7, s=.8)
sns.violinplot(data=data[["User_Score","Critic_Score"]],palette=pal, inner="points")
plt.show()
f, ax = plt.subplots(figsize=(10, 10))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(data.iloc[:,3:].corr(), cmap=cmap, vmax=.3, center=0,square=True,annot=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()
f, ax = plt.subplots(figsize=(15, 10))
sns.boxplot(x=data.Platform[data.Platform=="PSV"], y=data.Global_Sales,hue=data.Rating)
plt.show()
f,ax = plt.subplots(figsize=(20, 10))
sns.swarmplot(x=data.Genre, y=data.Critic_Score,hue=data.Rating)
plt.show()
sns.pairplot(data.iloc[:,10:14])
plt.show()
plt.figure(figsize=(15,10))
sns.countplot(data.Genre)
plt.title("Genres",color = 'g',fontsize=15)
plt.show()
plt.figure(figsize=(10,8))
above_5 =['above_5' if i >= 5 else 'below_5' for i in data.User_Score]
df = pd.DataFrame({'score':above_5})
sns.countplot(x=df.score)
plt.ylabel('Number of User Scores')
plt.title('Distribution of Scores',color = 'r',fontsize=15)
plt.show()
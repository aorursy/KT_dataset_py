# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
%matplotlib inline
import sys 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Any results you write to the current directory are saved as output.
gpsa=pd.read_csv("../input/googleplaystore.csv")
gpsa.columns = gpsa.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')
gps=gpsa.copy()
gps.installs = gps.installs.str.strip().str.replace("+","").str.replace(",","").str.replace("Free","0")
gps.installs=gps.installs.astype(float)
#gps.installs.unique()
gps.head(10)
#gps.info()
gps.dropna(inplace=True)
gps.info()
#1
app_category_list= list(gps["category"].unique())
#app_category
category_rating=[]
for i in app_category_list:
    x=gps[gps["category"]==i]
    category_rating.append(sum(x.rating)/len(x))
#sorting
data=pd.DataFrame({"Category":app_category_list,"Category_Rating":category_rating})
new_index=(data["Category_Rating"].sort_values(ascending=False)).index.values
sort_data=data.reindex(new_index)
#Visualization
plt.figure(figsize=(20,8))
sns.barplot(x=sort_data["Category"],y=sort_data["Category_Rating"],alpha=0.5)
plt.xticks(rotation=90)
plt.xlabel("Category",color="r",fontsize=15)
plt.ylabel("Category Rating",color="r",fontsize=15)
plt.title("Rating rates by Category",color="r",fontsize=15)
plt.show()
#2 
#app_category_list ----> using
gps.reviews=gps.reviews.astype(float)
category_reviews=[]
for i in app_category_list:
    x=gps[gps["category"]==i]
    category_reviews.append(sum(x.reviews)/len(x))
#sorting
data=pd.DataFrame({"Category":app_category_list,"Category_Reviews_Ratings":category_reviews})
new_index=(data["Category_Reviews_Ratings"].sort_values(ascending=False)).index.values
sort_data2=data.reindex(new_index)
#Visualization
current_palette =sns.color_palette("Paired")
plt.figure(figsize=(20,8))
sns.barplot(x=sort_data2["Category"],y=sort_data2["Category_Reviews_Ratings"],alpha=0.5,palette=current_palette)
plt.xticks(rotation=90)
plt.xlabel("Category",color="r",fontsize=15)
plt.ylabel("Category Reviews Ratings",color="r",fontsize=15)
plt.title("Number of reviews by categories",color="r",fontsize=15)
plt.show()
#3 
#Türlere göre rating oranları 
#app_category_list ----> using
genres_list=list(gps.genres.unique())
genres_rating=[]
for i in genres_list:
    x=gps[gps["genres"]==i]
    genres_rating.append(sum(x.rating)/len(x))

#sorting
data=pd.DataFrame({"Genres_List":genres_list,"Genres_Rating":genres_rating})
new_index=(data["Genres_Rating"].sort_values(ascending=True)).index.values
sorted_data=data.reindex(new_index)

#Visualization
plt.figure(figsize=(20,8))
sns.barplot(x=sorted_data["Genres_List"],y=sorted_data["Genres_Rating"])
plt.xticks(rotation=90)
plt.xlabel("Genres",color="r",fontsize=15)
plt.ylabel("Genres Ratings",color="r",fontsize=15)
plt.title("Rating rates by Genres",color="r",fontsize=15)
plt.show()
#4
category_installs=[]
for i in app_category_list:
    x=gps[gps["category"]==i]
    category_installs.append(sum(x.installs)/len(x))
#sorting
data=pd.DataFrame({"Category":app_category_list,"Category_Installs_Ratings":category_installs})
new_index=(data["Category_Installs_Ratings"].sort_values(ascending=False)).index.values
sort_data3=data.reindex(new_index)
#Visualization
current_palette =sns.color_palette("Paired")
plt.figure(figsize=(20,8))
sns.barplot(x=sort_data3["Category"],y=sort_data3["Category_Installs_Ratings"],alpha=0.5,palette=current_palette)
plt.xticks(rotation=90)
plt.xlabel("Category",color="r",fontsize=15)
plt.ylabel("Category Installs Ratings",color="r",fontsize=15)
plt.title("Number of Installs by categories",color="r",fontsize=15)
plt.show()
#1
#gps.tail()
#gps.info()
#Normalization
sort_data2["Category_Reviews_Ratings"]=sort_data2["Category_Reviews_Ratings"]/max(sort_data2["Category_Reviews_Ratings"])
sort_data3["Category_Installs_Ratings"]=sort_data3["Category_Installs_Ratings"]/max(sort_data3["Category_Installs_Ratings"])

#sorting
data_concat=pd.concat([sort_data2,sort_data3["Category_Installs_Ratings"]],axis=1)
data_concat.sort_values("Category_Reviews_Ratings",inplace=True)


plt.figure(figsize=(20,8))
sns.pointplot(x=data_concat["Category"],y=data_concat["Category_Reviews_Ratings"],alpha=0.5,color="red")
sns.pointplot(x=data_concat["Category"],y=data_concat["Category_Installs_Ratings"],alpha=0.9,color="orange")
plt.xticks(rotation=90)
plt.text(20,0.50,"Category Reviews Ratings",color="orange",fontsize=15,style="italic")
plt.text(20,0.55,"Category Installs Ratings",color="red",fontsize=15,style="italic")
plt.title("Categorical - Installs and Review",color="r",fontsize=15)
plt.show()
#data_concat.head()
#1
sns.jointplot(x=data_concat["Category_Reviews_Ratings"],y=data_concat["Category_Installs_Ratings"],kind="kde")
plt.show()
#2
sns.jointplot(x=data_concat["Category_Reviews_Ratings"],y=data_concat["Category_Installs_Ratings"],kind="scatter")
plt.show()
#3
sns.jointplot(x=data_concat["Category_Reviews_Ratings"],y=data_concat["Category_Installs_Ratings"],kind="hex")
plt.show()
#1 type rates
#gps.head()
app_type=gps.type.unique()
app_type_value=gps.type.value_counts()

colors=["orange","red"]
explode=[0,0]

#Visualization
plt.figure(figsize=(8,8))
plt.pie(app_type_value,explode=explode,labels=app_type,colors=colors,autopct="%1.1f%%")
plt.title("Type Rates",color="black",fontsize=20)
plt.show()

#1 Categorical Reviews and Installs
sns.lmplot(x="Category_Reviews_Ratings",y="Category_Installs_Ratings",data=data_concat)
plt.xlabel("Category Reviews Rating")
plt.ylabel("Category Installs Rating")
plt.title("LMPLOT - Categorical Reviews and Installs")
plt.show()
#1 Categorical Reviews and Installs
plt.figure(figsize=(7,7))
sns.kdeplot(data_concat["Category_Reviews_Ratings"],data_concat["Category_Installs_Ratings"],cut=3,shade=True)
plt.xlabel("Category Reviews Rating")
plt.ylabel("Category Installs Rating")
plt.title("KDEPLOT - Categorical Reviews and Installs")
plt.show()
#1
plt.figure(figsize=(8,8))
sns.violinplot(data=data_concat,inner="points")
plt.ylabel("Frequency")
plt.title("VIOLINPLOT - Categorical Reviews and Installs")
plt.show()
#1 Reviews and Installs
plt.figure(figsize=(8,8))
sns.heatmap(data_concat.corr(),annot=True,linewidth=.5,linecolor="red",fmt=".1f")
plt.title("HEATMAP - Categorical Reviews and Installs")
plt.show()
#1 Types of payments by rating and comments
plt.figure(figsize=(24,8))
sns.boxplot(x=gps.installs,y=gps.rating,hue=gps.type,palette="PRGn")
plt.title("BOXPLOT",color="red",fontsize=15)
plt.show()
#1 Types of payments by rating and comments
plt.figure(figsize=(24,8))
sns.swarmplot(x=gps.installs,y=gps.rating,hue=gps.type,palette="PRGn")
plt.title("SWARMPLOT",color="red",fontsize=15)
plt.show()
#1 Categorical Install and Reviews
sns.pairplot(data_concat)
plt.title("PAİRPLOT - Categorical Reviews and Installs")
plt.show()
#1 
sns.countplot(gps.type)
plt.title("Numbers of Type",color="r")
plt.show()
#2
above=["above4" if i>4 else "under4"  for i in gps.rating]
new_df=pd.DataFrame({"rating":above})
sns.countplot(x=new_df.rating)
plt.ylabel("Frequency")
plt.title("Rating rates")
plt.show()
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Let's read our data and analyzing it
video = pd.read_csv("../input/Video_Games_Sales_as_at_22_Dec_2016.csv")
video.shape
video.info()
video.head(10)
video.columns
# we have to get rid of all nan values so we use dropna() function
video.dropna(inplace = True)
video.reset_index(drop = True, inplace=True)
# Lets look at our unique platforms, genres
print(video.Platform.unique())
print("")
print(video.Genre.unique())
# Creating sales ratio of platforms

global_sales_ratio = []
na_sales_ratio = []
eu_sales_ratio = []
jp_sales_ratio = []
other_sales_ratio = []
critic_score_ratio = []
platform_list = list(video.Platform.unique())

for i in platform_list:
    x = video[video.Platform == i]
    global_sales_ratio.append(sum(x.Global_Sales)/len(x))
    na_sales_ratio.append(sum(x.NA_Sales)/len(x))
    eu_sales_ratio.append(sum(x.EU_Sales)/len(x))
    jp_sales_ratio.append(sum(x.JP_Sales)/len(x))
    other_sales_ratio.append(sum(x.Other_Sales)/len(x))
    
sales_ratio = pd.DataFrame({"platform":platform_list,
                            "global_sales_ratio":global_sales_ratio,
                            "na_sales_ratio":na_sales_ratio,
                            "eu_sales_ratio":eu_sales_ratio,
                            "jp_sales_ratio":jp_sales_ratio,
                            "other_sales_ratio":other_sales_ratio})
sales_ratio
# visualization
plt.figure(figsize=(20,20))
sns.color_palette("dark")
sns.barplot(x="platform", y="global_sales_ratio", data=sales_ratio,
            palette=sns.color_palette("cubehelix", 25))

plt.title("Global Sales Ratio of Platforms", fontsize=25)
plt.xlabel("Platforms", fontsize=15)
plt.ylabel("Global Sales Ratio", fontsize=15)
plt.show()

f,ax = plt.subplots(figsize=(15,10))
sns.barplot(x=global_sales_ratio, y=platform_list, label="Global", color="g")
sns.barplot(x=na_sales_ratio, y=platform_list, label="North America", color="b")
sns.barplot(x=eu_sales_ratio, y=platform_list, label="European Union", color="cyan")
sns.barplot(x=jp_sales_ratio, y=platform_list, label="Japan", color="red")
sns.barplot(x=other_sales_ratio, y=platform_list, label="Other", color="yellow")

ax.legend(frameon=True)
# ax.set(xlabel="Platforms", ylabel="Sales' Ratio", title= "Sales Ratio vs Platforms")
plt.xlabel("Platform", fontsize=12)
plt.ylabel("Sales' Ratio", fontsize=12)
plt.title("Platforms vs Sales' Ratio", fontsize=15, color="black")
video.Genre.value_counts()
# genre count
genre_count = Counter(video.Genre)
data2 = genre_count.most_common(12)
genre,count= zip(*data2)
genre,count = list(genre), list(count)

#visualization
plt.figure(figsize=(20,20))
sns.barplot(x = genre, y = count, palette = sns.color_palette("dark"))
plt.title("Frequency of Genre", fontsize=25)
plt.xlabel("Genre", fontsize=15)
plt.ylabel("Frequency", fontsize = 15)
plt.show()
na_sales_ratio = []
eu_sales_ratio = []
platform_list = list(video.Platform.unique())

for i in platform_list:
    x = video[video.Platform == i]
    na_sales = sum(x.NA_Sales)/len(x)
    na_sales_ratio.append(na_sales)
    eu_sales = sum(x.EU_Sales)/len(x)
    eu_sales_ratio.append(eu_sales)
    
data = pd.DataFrame({"na_sales_ratio":na_sales_ratio, "eu_sales_ratio":eu_sales_ratio})

# Visualization
f,ax = plt.subplots(figsize=(15,8))
sns.pointplot(x=platform_list, y= data.na_sales_ratio, color="red", ax=ax )
sns.pointplot(x=platform_list, y= data.eu_sales_ratio, color="lime", ax=ax )

plt.text(14, 0.6, 'na sales ratio', color='red', fontsize = 15, style = 'italic')
plt.text(14, 0.65, 'eu sales ratio', color='lime', fontsize = 15, style = 'italic')
ax.set(xlabel="Platforms", ylabel = "Sales Ratio", title= "Sales Ratio of Platforms" )
plt.grid()
# Show the joint distribution using kernel density estimation 

sns.jointplot(x=global_sales_ratio, y=na_sales_ratio, data=sales_ratio, kind="kde", size=7)
plt.show()   
# you can change parameters of joint plot
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
# Different usage of parameters but same plot with previous one

sns.jointplot(x=global_sales_ratio, y=na_sales_ratio, data=sales_ratio, size=7, ratio=4 , color="red")
plt.show()
# preparing the data for visualization
labels = video.Genre.value_counts().index
colors = ["lime", "coral", "forestgreen", "gold", "grey", "crimson", "r", "yellow", "maroon", "darkorange", "deepskyblue", "c"]
explode = [0,0,0,0,0,0,0,0,0,0,0,0]
sizes = video.Genre.value_counts().values

plt.figure(figsize=(10,10))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.show()
# Visualization of global sales rate vs NA sales rate of each platform with different style of seaborn code
# lmplot
# Show the results of a linear regression within each dataset
sns.lmplot(x="global_sales_ratio", y="eu_sales_ratio", data=sales_ratio)
plt.show()
# Visualization of global sales rate vs NA sales rate of each platform with different style of seaborn code
sns.kdeplot(sales_ratio.global_sales_ratio, sales_ratio.eu_sales_ratio, shade=True, cut=4)
plt.show()
# Show each distribution with both violins and points
plt.figure(figsize=(10,4))
sns.violinplot(data=data, colors = ("red","blue","yellow"), inner="points")
plt.show()
data.corr()
plt.subplots(figsize=(5,5))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, linecolor="black")
plt.show()
new_data = pd.DataFrame({"name": ("charles","max","moe","jason","sue","abby","amanda","alexa"),
                         "occupation": ("doctor","teacher","teacher","doctor","doctor","teacher","doctor","teacher"),
                         "gender":("male","male","male","male","female","female","female","female"),
                         "age": (30,25,45,50,41,20,58,37)})
new_data
plt.subplots(figsize=(7,7))
sns.boxplot(x="occupation", y="age", hue="gender", data=new_data, palette="Set2")
video.head()
plt.subplots(figsize=(7,7))
sns.swarmplot(x="occupation", y="age", hue="gender", data=new_data, palette="Set2")
data.head()
sns.pairplot(data)
plt.show()
video.head()
video.Year_of_Release = video.Year_of_Release.astype(int)  # change "flot year" to "int year"
release = video.loc[:100, "Year_of_Release"]
#visualization
plt.subplots(figsize=(15,10))
sns.countplot(release)
plt.xlabel("Year")
plt.title("Year of Release", fontsize=15)

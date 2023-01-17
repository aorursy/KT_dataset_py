# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dataset=pd.read_csv("/kaggle/input/videogamesales/vgsales.csv")
dataset.info()
dataset.describe()
dataset.head()
dataset.tail()
dataset.info()
def  bar_plot(variable):

    var=dataset[variable]

    varValue=var.value_counts()

    fig=plt.figure(figsize=(9,3))

    plt.bar(varValue[:6].index,varValue[:6].values)

    plt.xticks(varValue[:6].index,varValue[:6].index.values,rotation=45)

    plt.ylabel("Frequency")

    plt.title(variable)
Category=["Genre","Publisher"]

for c in Category:

    bar_plot(c)
Category_a=["Platform"]

for v in Category_a:

    bar_plot(v)
def dist_plot(variable):

    fig=plt.figure(figsize=(9,3))

    x=dataset[variable]

    sns.distplot(x,bins=25,kde=True)

    plt.ylabel("Frequency")

    plt.title(variable)

    plt.show()
numvalue=["Year","NA_Sales","EU_Sales","JP_Sales","Other_Sales","Global_Sales"]

for c in numvalue:

    dist_plot(c)
dataset.isnull().sum()
dataset.dropna(axis=0,inplace=True)
dataset.isnull().sum()
platform_list=list(dataset["Platform"].unique())

platform_global_sal=[]

for i in platform_list:

    x=dataset[dataset["Platform"]==i]

    platform_ratio=sum(x.JP_Sales)/len(x)

    platform_global_sal.append(platform_ratio)

data=pd.DataFrame({"platform list":platform_list,"EU_Sales":platform_global_sal})

new_index=data["EU_Sales"].sort_values(ascending=False).index.values

sorted_data=data.reindex(new_index)

#visualiton

fig=plt.Figure(figsize=(15,10))

sns.barplot(x=sorted_data["platform list"],y=sorted_data["EU_Sales"])

plt.xticks(rotation=90)

plt.xlabel("Platforms")

plt.ylabel("EU  Sales")

plt.title("EU Sales of Each Platforms")
platform_list=list(dataset["Platform"].unique())

platform_global_sal=[]

for i in platform_list:

    x=dataset[dataset["Platform"]==i]

    platform_ratio=sum(x.JP_Sales)/len(x)

    platform_global_sal.append(platform_ratio)

data=pd.DataFrame({"platform list":platform_list,"Global_Sales":platform_global_sal})

new_index=data["Global_Sales"].sort_values(ascending=False).index.values

sorted_data=data.reindex(new_index)

#visualiton

fig=plt.Figure(figsize=(15,10))

sns.barplot(x=sorted_data["platform list"],y=sorted_data["Global_Sales"])

plt.xticks(rotation=90)

plt.xlabel("Platforms")

plt.ylabel("Global  Sales")

plt.title("Global Sales of Each Platforms")
platform_list=list(dataset["Platform"].unique())

platform_global_sal=[]

for i in platform_list:

    x=dataset[dataset["Platform"]==i]

    platform_ratio=sum(x.JP_Sales)/len(x)

    platform_global_sal.append(platform_ratio)

data=pd.DataFrame({"platform list":platform_list,"JP_Sales":platform_global_sal})

new_index=data["JP_Sales"].sort_values(ascending=False).index.values

sorted_data=data.reindex(new_index)

#visualiton

fig=plt.Figure(figsize=(15,10))

sns.barplot(x=sorted_data["platform list"],y=sorted_data["JP_Sales"])

plt.xticks(rotation=90)

plt.xlabel("Platforms")

plt.ylabel("JP  Sales")

plt.title("JP Sales of Each Platforms")
platform_list=list(dataset["Platform"].unique())

platform_NA_sal=[]

for i in platform_list:

    x=dataset[dataset["Platform"]==i]

    platform_ratio=sum(x.NA_Sales)/len(x)

    platform_NA_sal.append(platform_ratio)

data2=pd.DataFrame({"platform list":platform_list,"NA_Sales":platform_NA_sal})

new_index2=data2["NA_Sales"].sort_values(ascending=False).index.values

sorted_data2=data2.reindex(new_index2)

#visualiton

fig=plt.Figure(figsize=(15,10))

sns.barplot(x=sorted_data2["platform list"],y=sorted_data2["NA_Sales"])

plt.xticks(rotation=90)

plt.xlabel("Platforms")

plt.ylabel("NA_Sales")

plt.title("NA Sales of Each Platforms")
data=pd.concat([sorted_data,sorted_data2["NA_Sales"]],axis=1)

data.sort_values("JP_Sales",inplace=True)

f,ax1=plt.subplots(figsize=(15,10))

sns.pointplot(x='platform list',y='JP_Sales',data=data,color='lime',alpha=0.8)

sns.pointplot(x='platform list',y='NA_Sales',data=data,color='red',alpha=0.8)

plt.text(40,0.6,'Na Sales',color='red',fontsize = 17,style = 'italic')

plt.text(40,0.55,'JP Sales',color='lime',fontsize = 18,style = 'italic')

plt.xlabel('Platfotm',fontsize = 15,color='blue')

plt.ylabel('Sales',fontsize = 15,color='blue')

plt.title('JP sales VS  NA Sales',fontsize = 20,color='blue')

plt.grid()
a=dataset.Genre.value_counts()[0:6]

sizes=a.values

labels=a.index

explode=[0,0,0,0,0,0]

colors=["orange","red","blue","green","yellow","violet"]

plt.figure(figsize=(7,7))

plt.pie(sizes,explode=[0.1]*6,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Genre',color = 'blue',fontsize = 15)
a=dataset.Platform.value_counts()[0:6]

sizes=a.values

labels=a.index

explode=[0,0,0,0,0,0]

colors=["orange","red","blue","green","yellow","violet"]

plt.figure(figsize=(7,7))

plt.pie(sizes,explode=[0.1]*6,labels=labels,colors=colors,autopct='%1.1f%%')

plt.title('Platforms',color = 'blue',fontsize = 15)
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)

sns.violinplot(data=data,palette=pal,inner="points")

plt.show()
sns.countplot(dataset.Genre)

plt.xticks(rotation=90)

plt.title("Genre",color="blue",fontsize=15)
sns.countplot(dataset.Platform)

plt.xticks(rotation=90)

plt.title("Platforms",color="blue",fontsize=15)
from wordcloud import WordCloud

xdata=dataset.Name

wordcloud=WordCloud(

               background_color="white",

               width=1200,

               height=800 ).generate(" ".join(xdata))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
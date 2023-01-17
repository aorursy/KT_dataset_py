# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
vgsales=pd.read_csv("../input/videogamesales/vgsales.csv",sep=",")
df=vgsales.copy()
df.head()
df.info()
df.Year=df.Year.astype(float)
df.isnull().sum()
df.dropna(inplace=True)
df.isnull().sum()
df.Name.value_counts()
df.Publisher.value_counts()
df.Platform.value_counts() # 31
df.Year.value_counts()
df.Genre.value_counts() # 12
genre_list=list(df.Genre.unique())
year_list2=list(df.Year.unique())
publisher_list= list(df.Publisher.unique())
platform_list=list(df.Platform.unique())
name_list=list(df.Name.unique())
year_list2
data1=pd.DataFrame({"Release_Year":year_list,"Num_of_released_game":df.Year.value_counts().values})
data1.sort_values(by="Release_Year",ascending=True,inplace=True)
data1.reset_index(drop=True)
plt.figure(figsize=(15,15))
sns.pointplot(x='Release_Year',y='Num_of_released_game',data=data1,color='green',alpha=0.8)
plt.xticks(rotation=90)
plt.xlabel("Year")
plt.ylabel("Number of Game Released ")
plt.title("Number of Released Game by Year")
plt.grid()

sales_mean=[]

for i in year_list:
    x=df[df["Year"]==i]
    sales_mean.append(x.Global_Sales.sum())
    
data2=pd.DataFrame({"Year":year_list,"Global_Sales":sales_mean})
data2.sort_values(by="Year",ascending=True,inplace=True)
data2.reset_index(drop=True)


plt.figure(figsize=(15,15))
sns.pointplot(x='Year',y='Global_Sales',data=data2,color='red',alpha=0.8)
plt.xticks(rotation=90)
plt.xlabel("Year")
plt.ylabel(" Global Sales in million ")
plt.title("Global Sales in million by Year")
plt.grid()



    
labels = df.Genre.value_counts().index
colors = ['blue','red','yellow','green','brown',"purple","pink","gray","orange","darkblue","lime","cyan"]
explode = [0,0,0,0,0,0,0,0,0,0,0,0]
sizes = df.Genre.value_counts().values

# visual
plt.figure(figsize = (10,10))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Preference of Game Genre',color = 'blue',fontsize = 15)
plt.figure(figsize=(20,20))
sns.barplot(x=df.Publisher.value_counts().index[:20],y=df.Publisher.value_counts().values[:20],data=df)
plt.xticks(rotation=90)
plt.xlabel("Name of Game Company")
plt.ylabel("Number of released game")
plt.title("Number of released game from Game companies Top 20",color="blue",fontsize=15)
name_lenght=[]

for i in name_list:
    name_lenght.append(len(i))
    

data3=pd.DataFrame({"Game_Name":name_list[:25],"Name_Lenght":name_lenght[:25]})
data3.sort_values(by="Name_Lenght",ascending=False,inplace=True)
data3.reset_index(drop=True,inplace=True)

plt.figure(figsize=(15,15))
sns.barplot(x="Game_Name",y="Name_Lenght",data=data3,palette = sns.cubehelix_palette(len(x)))
plt.xticks(rotation=90)
plt.xlabel("Game Name")
plt.ylabel("Name Lenght")
plt.title("Game's name lenght Top 25",fontsize=15,color="blue")
year_list=year_list.sort()


global_sales = []
eu_sales = []
jp_sales = []
na_sales = []
other_sales = []
for i in year_list2:
    x = df[df['Year']==i]
    global_sales.append(sum(x.Global_Sales))
    eu_sales.append(sum(x.EU_Sales)) 
    jp_sales.append(sum(x.JP_Sales))
    na_sales.append(sum(x.NA_Sales))
    other_sales.append(sum(x.Other_Sales)) 
    
    
#Visualization

f,ax = plt.subplots(figsize = (20,20))
sns.barplot(x=year_list2,y=global_sales,color='purple',alpha = 0.5,label='Global Sales' ) # alpha =saydamlık
sns.barplot(x=year_list2,y=na_sales,color='blue',alpha = 0.7,label='Na Sales')
sns.barplot(x=year_list2,y=eu_sales,color='cyan',alpha = 0.6,label='EU Sales')
sns.barplot(x=year_list2,y=jp_sales,color='yellow',alpha = 0.6,label='JP Sales')
sns.barplot(x=year_list2,y=other_sales,color='red',alpha = 0.6,label='Other Sales')
plt.xticks(rotation=90)

ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu yani sağ alttaki kutucuğum yeri ve şekli
ax.set(xlabel='Year', ylabel='Number of Sales',title = "Number of Sales in million by Year")
sns.lmplot(x="Global_Sales", y="EU_Sales", data=df)
plt.show()
sns.lmplot(x="JP_Sales", y="EU_Sales", data=df)
plt.show()
sns.jointplot("Global_Sales", "NA_Sales", data=df,size=5, ratio=3, color="r")
df3=df.drop(columns=["Rank","Year"])

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=df3, palette=pal, inner="points")
plt.show()
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(df3.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
sns.pairplot(df3)
plt.show()

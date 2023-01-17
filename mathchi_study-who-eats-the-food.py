# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
from subprocess import check_output             # utf-8 e uygun olup olmadigini arastirmak icin bunu aratiyoruz
print(check_output(["ls","../input"]).decode("utf8"))
        

# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/world-foodfeed-production/FAO.csv', encoding='cp1254')          
data.head()
data.info()   
data.Y2012 = data.Y2012.astype(float)
data.Y2013 = data.Y2013.astype(float)
data.info()
data.rename(columns={'Area Abbreviation': 'Area_Abbreviation', 'Item Code': 'Item_Code', 'Element Code': 'Element_Code', 'Area Code': 'Area_Code'}, inplace=True)
data.head()
area_list = list(data['Area'].unique())
# print(area_list)

area_Y1961_ratio = []
for i in area_list:
    x = data[data['Area']==i]
    area_Y1961_rate = sum(x.Y1961)/len(x)
    area_Y1961_ratio.append(area_Y1961_rate)
data1 = pd.DataFrame({'area_list': area_list,'area_Y1961_ratio':area_Y1961_ratio})

# Duzenlenen datayi yeniden index lememiz gerekiyor ki duzenli gorunsun!!! Bunun icin:

new_index = (data1['area_Y1961_ratio'].sort_values(ascending=False)).index.values
sorted_data = data1.reindex(new_index)
print(sorted_data)
data1.head()
# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['area_list'].head(20), y=sorted_data['area_Y1961_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('States')
plt.ylabel('Y1961')
plt.title('Y1961 States')
data.head()
data.Item.value_counts()
name_count = Counter(data.Item)
print(name_count)

most_common_names = name_count.most_common(15)  
x,y = zip(*most_common_names)
x,y = list(x),list(y)
# PLOT unu cizelim

plt.figure(figsize=(25,10))
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Item Names')
plt.ylabel('Frequency')
plt.title('Most common 15 Item Names')
print(most_common_names)
data.head()
# area_list = list(data['Area'].unique())
# print(area_list)

area_Y2013_ratio = []
for i in area_list:
    x = data[data['Area']==i]
    area_Y2013_rate = sum(x.Y2013)/len(x)
    area_Y2013_ratio.append(area_Y2013_rate)
data2 = pd.DataFrame({'area_list': area_list,'area_Y2013_ratio':area_Y2013_ratio})

# Duzenlenen datayi yeniden index lememiz gerekiyor ki duzenli gorunsun!!! Bunun icin:

new_index = (data2['area_Y2013_ratio'].sort_values(ascending=False)).index.values
sorted_data2 = data2.reindex(new_index)
# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['area_list'].head(20), y=sorted_data2['area_Y2013_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('States')
plt.ylabel('Y2013')
plt.title('Y2013 States')
data.head()
Item_Code = []
Element_Code = []
Y1961 = []
Y2013 = []

for i in area_list:
    x = data[data['Area']==i]
    Item_Code.append(sum(x.Item_Code)/len(x))
    Element_Code.append(sum(x.Element_Code)/len(x))
    Y1961.append(sum(x.Y1961)/len(x))
    Y2013.append(sum(x.Y2013)/len(x))

# visualization
f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=Item_Code,y=area_list,color='green',alpha = 0.5,label='Item Code' )
sns.barplot(x=Element_Code,y=area_list,color='blue',alpha = 0.7,label='Element Code')
sns.barplot(x=Y1961,y=area_list,color='cyan',alpha = 0.6,label='Year 1961')
sns.barplot(x=Y2013,y=area_list,color='yellow',alpha = 0.6,label='Year 2013')

ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Percentage of Items, Elements', ylabel='States',title = "Percentage of State's Food Using ")
data.head()
data.sort_values('Item').head()
sorted_data['area_Y1961_ratio'] = sorted_data['area_Y1961_ratio']/max( sorted_data['area_Y1961_ratio'])
sorted_data2['area_Y2013_ratio'] = sorted_data2['area_Y2013_ratio']/max( sorted_data2['area_Y2013_ratio'])
data3 = pd.concat([sorted_data,sorted_data2['area_Y2013_ratio']],axis=1)
data3.sort_values('area_Y1961_ratio',inplace=True)
# visualize
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='Area_Abbreviation',y='Y1961',data=data,color='lime',alpha=0.8)
sns.pointplot(x='Area_Abbreviation',y='Y2013',data=data,color='red',alpha=0.8)
plt.text(40,0.6,'Foods of Y1961',color='red',fontsize = 17,style = 'italic')   # burdaki degerler yazilarin konumlari istedigimiz gibi degistirebiliriz
plt.text(40,0.55,'Foods of Y2013',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('States',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('FOODS for Y1961  VS  Y2013',fontsize = 20,color='blue')
plt.grid()
data.Y1961.plot(kind= 'line', color= 'b', label= 'Y1961', linewidth= 2, alpha= 0.5, grid= True, linestyle= '-')
data.Y2013.plot(color= 'r', label= 'Y2013', linewidth= 2, alpha= 0.5, grid= True, linestyle= '-')
plt.legend(loc= 'upper right')          #legend= puts label into plot
plt.xlabel('x axis')                    #label= name of label
plt.ylabel('y axis')                    #label= name of label
plt.title('Line Plot')                  #title= title of plot
plt.show()
data.head()
g = sns.jointplot(data.Y1961, data.Y2013, kind="kde", size=7)
plt.savefig('graph.png')                          # bu bize sekilleri kaydetmemizi saglar
plt.show()
# you can change parameters of joint plot
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
# Different usage of parameters but same plot with previous one

g = sns.jointplot("Y1961", "Y2013", data=data,size=5, ratio=3, color="r")
data.Item.dropna(inplace = True)
labels = data.Item.value_counts().index
colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,0,0,0]
sizes = data.Item.value_counts().values

# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=None, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Foods According to Items',color = 'blue',fontsize = 15)
data3.head()
# Visualization of Foods Y1961 vs Y2013 of each state with different style of seaborn code
# lmplot 
# Show the results of a linear regression within each dataset


sns.lmplot(x="area_Y1961_ratio", y="area_Y2013_ratio", data=data3)
plt.show()
data.head()
# cubehelix plot
# 2013 teki itemslerin cubhelix cizelim

sns.kdeplot(data.Area_Code, data.Y2013, shade=True, cut=3)      # burada 'cut' sekil arasi genisligi verir
plt.show()
# Show each distribution with both violins and points
# Use cubehelix to get a custom sequential palette

pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data, palette=pal, inner="points")
plt.show()
# Ilk once korelasyonuna bakalim

data3.corr()
#correlation map
# Visualization of Y1961 rate vs Y2013 rate of each state with different style of seaborn code

f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data3.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
data.Item.head().unique()
data.head()
# Food Yiyecek
# Feed: Besleme
# Plot the orbital period with horizontal boxes

sns.boxplot(x="Area_Code", y="Element_Code", hue="Element", data=data, palette="PRGn")
plt.show()
# swarm plot

sns.swarmplot(x="Item", y="Y2013",hue="Element", data=data)
plt.show()
data3.head()
# pair plot
sns.pairplot(data3)
plt.show()
data.Element.value_counts()

sns.countplot(data.Element)
#sns.countplot(data.Item)                 # istersek bunuda ayri ayri cizdirebiliriz
plt.title("Element",color = 'blue',fontsize=15)
data3.head()
area_list = data3.area_list.value_counts()
#print(area_list)

plt.figure(figsize=(10,7))
sns.barplot(x=area_list[:7].index,y=area_list[:7].values)
plt.ylabel('Number of Foods')
plt.xlabel('Areas')
plt.title('Countries',color = 'blue',fontsize=15)
data.head()
above2800 =['above2800' if i >= 2800 else 'below2800' for i in data.Item_Code]
df = pd.DataFrame({'Item_Code':above2800})
sns.countplot(x=df.Item_Code)
plt.ylabel('Number of Item_Code')
plt.title('Item_Code of Foods',color = 'blue',fontsize=15)
sns.countplot(data=data.head(1000), x='Area')
plt.title('Number of foods for Countries',color = 'blue',fontsize=15)
# Most dangerous cities
most_common_names
Item = data.Item.value_counts()
plt.figure(figsize=(10,7))
sns.barplot(x=Item[:12].index,y=Item[:12].values)                 # 12. sutunu aliyoruz
plt.xticks(rotation=45)
plt.title('Most common item names',color = 'blue',fontsize=15)
sns.countplot(data.Element)
plt.xlabel('Element Types')
plt.title('Element types',color = 'blue', fontsize = 15)
# Korelasyon matrix

f = plt.figure(figsize=(19, 15))
plt.matshow(data.corr(), fignum=f.number)
plt.xticks(range(data.shape[1]), data.columns, fontsize=10, rotation=45)
plt.yticks(range(data.shape[1]), data.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=19);
data.plot(kind = 'scatter', x = 'Y2000', y= 'Y2013', alpha = 0.5, color = 'red')
plt.xlabel('Y2000')
plt.ylabel('Y2013')
plt.title('Y2000 & Y2013 Scatter Plot')
plt.show()
data.Y2013.plot(kind = 'hist',bins = 70, figsize = (15,15))
plt.show()
data.Y1961.plot(kind = 'hist',bins = 50)
plt.clf()
series = data['Y1961']                           
print(type(series))     
data_frame = data[['Y2013']]  
print(type(data_frame))
x = data['Y2010'] < 200                  # Defans değeri 200'den KUCUK olan verileri x değişkenine atıyoruz.
print(x)                                 # bu sekilde true false olanlarin hepsini gosterir
data[x]
data[np.logical_and(data['Y1961'] > 200, data['Y2013'] < 2030 )]

# lets classify foods whether they have high or low use. Our threshold is average food use.
threshold = sum(data.Y2013)/len(data.Y2013)
data["use_level"] = ["high" if i > threshold else "low" for i in data.Y2013]
data.loc[:33,["use_level","Y2013"]] # we will learn loc more detailed later
data.info()
data.describe()
data.dropna(inplace = True)  
data.describe()

# bos degerler olmadigindan yukaridakinin ayni degerlerini verir. Bos degerler olsaydi pokemondaki gibi filtrelerdik
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Area_Code',by = 'Y2013')
data_new = data.head()    # I only take 5 rows into new data
data_new

# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Item', value_vars= ['Y1961','Y2013'])
melted
# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'value', columns = 'variable',values='Item')
# Firstly lets create 2 data frame
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data3 = data['Item'].head()
data1 = data['Y2000'].head()
data2= data['Y2013'].head()
conc_data_col = pd.concat([data3,data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col
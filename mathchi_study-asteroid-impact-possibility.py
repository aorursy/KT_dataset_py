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
data1 = pd.read_csv('/kaggle/input/asteroid-impacts/impacts.csv')
data2 = pd.read_csv('/kaggle/input/asteroid-impacts/orbits.csv')
data1.info()
data2.info()
             
data = pd.concat([data1, data2], axis=1, sort=False)
data.columns = ['Object_Name', 'Period_Start', 'Period_End', 'Possible_Impacts',
       'Cumulative_Impact_Probability', 'Asteroid_Velocity',
       'Asteroid_Magnitude', 'Asteroid_Diameter(km)',
       'Cumulative_Palermo_Scale', 'Maximum_Palermo_Scale',
       'Maximum_Torino_Scale', 'Object_Name', 'Object_Classification',
       'Epoch(TDB)', 'Orbit_Axis(AU)', 'Orbit_Eccentricity',
       'Orbit_Inclination(deg)', 'Perihelion_Argument(deg)',
       'Node_Longitude(deg)', 'Mean_Anomoly(deg)',
       'Perihelion_Distance(AU)', 'Aphelion_Distance(AU)',
       'Orbital_Period(yr)', 'Minimum_Orbit_Intersection_Distance(AU)',
       'Orbital_Reference', 'Asteroid_Magnitude']
data.columns
# print(data)
data1.columns = ['Object_Name', 'Period_Start', 'Period_End', 'Possible_Impacts',
       'Cumulative_Impact_Probability', 'Asteroid_Velocity',
       'Asteroid_Magnitude', 'Asteroid_Diameter(km)',
       'Cumulative_Palermo_Scale', 'Maximum_Palermo_Scale',
       'Maximum_Torino_Scale']
data1.head()
data1.info()
data1.Period_End.value_counts()
data1.Object_Name.head(50).unique()
object_list = list(data1['Object_Name'].head(50).unique())
# print(object_list)

Period_Start_ratio = []
for i in object_list:
    x = data1[data1['Object_Name']==i]
    Period_Start_rate = sum(x.Period_Start)/len(x)
    Period_Start_ratio.append(Period_Start_rate)
data3 = pd.DataFrame({'object_list': object_list,'Period_Start_ratio':Period_Start_ratio})


# Duzenlenen datayi yeniden index lememiz gerekiyor ki duzenli gorunsun!!! Bunun icin:

new_index = (data3['Period_Start_ratio'].sort_values(ascending=False)).index.values
sorted_data = data3.reindex(new_index)

# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['object_list'], y=sorted_data['Period_Start_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('Object Names')
plt.ylabel('Period Start')
plt.title('Period Start for Object')
data2.columns = ['Object_Name', 'Object_Classification',
       'Epoch(TDB)', 'Orbit_Axis(AU)', 'Orbit_Eccentricity',
       'Orbit_Inclination(deg)', 'Perihelion_Argument(deg)',
       'Node_Longitude(deg)', 'Mean_Anomoly(deg)',
       'Perihelion_Distance(AU)', 'Aphelion_Distance(AU)',
       'Orbital_Period(yr)', 'Minimum_Orbit_Intersection_Distance(AU)',
       'Orbital_Reference', 'Asteroid_Magnitude']

data2.head()
data2.info()
data2.Object_Name.value_counts()
name_count = Counter(data2.Object_Name)
# print(name_count)
most_common_names = name_count.most_common(15)  
x,y = zip(*most_common_names)
x,y = list(x),list(y)
# print(most_common_names)
# PLOT unu cizelim

plt.figure(figsize=(30,15))
ax= sns.barplot(x=x, y=y,palette = sns.cubehelix_palette(len(x)))
plt.xlabel('Object Names')
plt.ylabel('Frequency')
plt.title('Most common 15 Object Name')
print(most_common_names)
data2.head()
Period_End_ratio = []
for i in object_list:
    x = data1[data1['Object_Name']==i]
    Period_End_rate = sum(x.Period_End)/len(x)
    Period_End_ratio.append(Period_End_rate)
data4 = pd.DataFrame({'object_list': object_list,'Period_End_ratio': Period_End_ratio})


# Duzenlenen datayi yeniden index lememiz gerekiyor ki duzenli gorunsun!!! Bunun icin:

new_index = (data4['Period_End_ratio'].sort_values(ascending=False)).index.values
sorted_data2 = data4.reindex(new_index)
# visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data2['object_list'], y=sorted_data2['Period_End_ratio'])
plt.xticks(rotation= 45)
plt.xlabel('Object Names')
plt.ylabel('Period End')
plt.title('Period End for Object')
data1.head()
# print(object_list)

Cumulative_Impact_Probability = []
Asteroid_Velocity = []
Asteroid_Magnitude = []
# Asteroid_Diameter = []
Possible_Impacts = []
for i in object_list:
    x = data1[data1['Object_Name']==i]
    Cumulative_Impact_Probability.append(sum(x.Cumulative_Impact_Probability)/len(x))
    Asteroid_Velocity.append(sum(x.Asteroid_Velocity) / len(x))
    Asteroid_Magnitude.append(sum(x.Asteroid_Magnitude) / len(x))
#     Asteroid_Diameter.append(sum(x.Asteroid_Diameter) / len(x))
    Possible_Impacts.append(sum(x.Possible_Impacts) / len(x))
# visualization
f,ax = plt.subplots(figsize = (9,15))
sns.barplot(x=Cumulative_Impact_Probability,y=object_list,color='green',alpha = 0.5,label='Cumulative_Impact_Probability' )
sns.barplot(x=Asteroid_Velocity,y=object_list,color='blue',alpha = 0.7,label='Asteroid_Velocity')
sns.barplot(x=Asteroid_Magnitude,y=object_list,color='cyan',alpha = 0.6,label='Asteroid_Magnitude')
sns.barplot(x=Possible_Impacts,y=object_list,color='yellow',alpha = 0.6,label='Possible_Impacts')
# sns.barplot(x=share_hispanic,y=object_list,color='red',alpha = 0.6,label='Hispanic')

ax.legend(loc='lower right',frameon = True)     # legendlarin gorunurlugu
ax.set(xlabel='Percentage of Asteroids', ylabel='Asteroids',title = "Percentage of Asteroid Names ")
data3.head()
sorted_data['Period_Start_ratio'] = sorted_data['Period_Start_ratio']/max( sorted_data['Period_Start_ratio'])
sorted_data2['Period_End_ratio'] = sorted_data2['Period_End_ratio']/max( sorted_data2['Period_End_ratio'])
data4 = pd.concat([sorted_data,sorted_data2['Period_End_ratio']],axis=1)
data4.sort_values('Period_Start_ratio',inplace=True)
data4.head()
# visualize
f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='object_list',y='Period_Start_ratio',data=data4,color='lime',alpha=0.8)
sns.pointplot(x='object_list',y='Period_End_ratio',data=data4,color='red',alpha=0.8)
plt.text(40,0.6,'Period_End ratio',color='red',fontsize = 17,style = 'italic')   # burdaki degerler yazilarin konumlari istedigimiz gibi degistirebiliriz
plt.text(40,0.55,'Period_Start ratio',color='lime',fontsize = 18,style = 'italic')
plt.xlabel('Object Names',fontsize = 15,color='blue')
plt.ylabel('Values',fontsize = 15,color='blue')
plt.title('Period_Start ratio  VS  Period_End ratio',fontsize = 20,color='blue')
plt.grid()
data4.head()
g = sns.jointplot(data4.Period_Start_ratio, data4.Period_End_ratio, kind="kde", size=7)
plt.savefig('graph.png')                          # bu bize sekilleri kaydetmemizi saglar
plt.show()
# you can change parameters of joint plot
# kind : { “scatter” | “reg” | “resid” | “kde” | “hex” }
# Different usage of parameters but same plot with previous one

g = sns.jointplot("Period_Start_ratio", "Period_End_ratio", data=data4,size=5, ratio=3, color="r")
data2.Object_Name.head(15)
data2.Object_Name.value_counts()
data2.Object_Name.dropna(inplace = True)
labels = data2.Object_Name.value_counts().index
colors = ['grey','blue','red','yellow','green','brown']
explode = [0,0,0,0,0,0]
sizes = data2.Object_Name.value_counts().values
# visual
plt.figure(figsize = (7,7))
plt.pie(sizes, explode=None, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Asteroids According to Names',color = 'blue',fontsize = 15)
data4.head()
sns.lmplot(x="Period_Start_ratio", y="Period_End_ratio", data=data4)
plt.show()
sns.kdeplot(data4.Period_Start_ratio, data4.Period_End_ratio, shade=True, cut=3)      # burada 'cut' sekil arasi genisligi verir
plt.show()
pal = sns.cubehelix_palette(2, rot=-.5, dark=.3)
sns.violinplot(data=data4, palette=pal, inner="points")
plt.show()
data4.corr()
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(data4.corr(), annot=True, linewidths=0.5,linecolor="red", fmt= '.1f',ax=ax)
plt.show()
data4.head()
# pair plot
sns.pairplot(data4)
plt.show()
data2.head()
sns.countplot(data2.Object_Classification)
#sns.countplot(kill.manner_of_death)                 # istersek bunuda ayri ayri cizdirebiliriz
plt.title("Amor Asteroid",color = 'blue',fontsize=15)

Object_Classification = data2.Object_Classification.value_counts()
#print(Object_Classification)

plt.figure(figsize=(10,7))
sns.barplot(x=Object_Classification[:7].index,y=Object_Classification[:7].values)
plt.ylabel('Number of Objects')
plt.xlabel('Object Names')
plt.title('Object Names and numbers',color = 'blue',fontsize=15)
data.corr()
f,ax = plt.subplots(figsize=(10,10))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
plt.show()         
# carpisma yili ilk carpisicaklara gore sirala ve en bastaki 10 unu goster

data.sort_values('Period_End').head(10)
data.Period_Start.plot(kind= 'line', color= 'b', label= 'Period_Start', linewidth= 2, alpha= 0.5, grid= True, linestyle= '-')
data.Period_End.plot(color= 'y', label= 'Period_End', linewidth= 2, alpha= 0.5, grid= True, linestyle= '-.')
plt.legend(loc= 'upper right')          #legend= puts label into plot
plt.xlabel('x axis')                    #label= name of label
plt.ylabel('y axis')                    #label= name of label
plt.title('Line Plot')                  #title= title of plot
plt.show()
# Korelasyon matrix

f = plt.figure(figsize=(19, 15))
plt.matshow(data.corr(), fignum=f.number)
plt.xticks(range(data.shape[1]), data.columns, fontsize=14, rotation=90)
plt.yticks(range(data.shape[1]), data.columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=19);
data.plot(kind = 'scatter', x = 'Period_Start', y= 'Period_End', alpha = 0.5, color = 'red')
plt.xlabel('Period_Start')
plt.ylabel('Period_End')
plt.title('Object_Name & Period_End Scatter Plot')
plt.show()
data.Period_End.plot(kind = 'hist',bins = 70, figsize = (12,12))
plt.show()
data.Period_End.plot(kind = 'hist',bins = 50)
plt.clf()
series = data['Period_Start']                           
print(type(series))     
data_frame = data[['Period_End']]  
print(type(data_frame))
x = data['Period_End'] < 2030                      # Defans değeri 200'den büyük olan verileri x değişkenine atıyoruz.
print(x)                                       # bu sekilde true false olanlarin hepsini gosterir
data[x]                                        # sinirlama yapilan hangi pokemonlar ve ozelliklerini verir
data[np.logical_and(data['Period_Start'] > 2010, data['Period_End'] < 2030 )] 
data[(data['Period_Start'] > 2010) & (data['Period_End'] < 2030 )]
# lets return pokemon csv and make one more list comprehension example
# lets classify pokemons whether they have high or low speed. Our threshold is average speed.
threshold = sum(data.Period_End)/len(data.Period_End)
data["close_level"] = ["far away" if i > threshold else "close" for i in data.Period_End]
data.loc[:10,["close_level","Period_End"]]
data.info()

# ciktidan anlasilacagi uzere bos data mevcut degildir. Bos data mevcut olsaydi pokemonda yaptigimiz gibi filtrelememiz gerekirdi
data.describe()
# Bos datamiz olmadigi icin burasi yukardakinin aynisini verir

data.dropna(inplace = True)  
data.describe()
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Period_End',by = 'Asteroid_Diameter(km)')

# burada cok fazla asteroid old icin yakindan bakmak gerekir ama asagida yakin zamanli carpisacak olanlari secildi orada daha net anlasilabilir


data_new = data.head(10)    # I only take 5 rows into new data
data_new

data_new.boxplot(column='Period_End',by = 'Asteroid_Diameter(km)')
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Asteroid_Velocity', value_vars= ['Period_Start','Period_End'])
melted


# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Asteroid_Velocity', columns = 'variable',values='value')
# Firstly lets create 2 data frame
data1 = data.head()
data2= data.tail()
conc_data_row = pd.concat([data1,data2],axis =0,ignore_index =True) # axis = 0 : adds dataframes in row
conc_data_row
data3 = data['Asteroid_Velocity'].head()
data1 = data['Period_Start'].head()
data2= data['Period_End'].head()
conc_data_col = pd.concat([data3,data1,data2],axis =1) # axis = 0 : adds dataframes in row
conc_data_col
data.dtypes

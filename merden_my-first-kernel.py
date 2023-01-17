import pandas as pd

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches

import seaborn as sns

import numpy as np

plt.style.use('fivethirtyeight')

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from mpl_toolkits.basemap import Basemap

import folium

import folium.plugins

from matplotlib import animation,rc

import io

import base64

from IPython.display import HTML, display

import warnings

warnings.filterwarnings('ignore')

from scipy.misc import imread

import codecs

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
terror=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')   # read data 

terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)

terror=terror[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]

terror['Casualties']=terror['Killed']+terror['Wounded']   # add new info .. casualities = killed + woundered

terror.head(6)  # look first 6 data in table
terror.info()   # take the information about data how many data in the dataset, which type data in the dataset
![](https://www.iaspaper.net/wp-content/uploads/2017/08/Terrorism-Word-Cloud.jpg)
terror.corr()  #corresponding between datas. 
#correlation map

f,ax = plt.subplots(figsize=(12, 12))

sns.heatmap(terror.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)

plt.show()
terror.head(10)   # look firs 10 data in dataset about information what type data , which type items is here
terror.columns   # look columns name in dataset
# Line Plot

# color = color, label = label, linewidth = width of line, alpha = opacity, grid = grid, linestyle = sytle of line

terror.Killed.plot(kind = 'line', color = 'g',label = 'Killed',linewidth=1,alpha = 0.5,grid = True,linestyle = ':',figsize=(20, 20))

terror.Wounded.plot(color = 'r',label = 'Wounded',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')

plt.legend(loc='upper right')     # legend = puts label into plot

plt.xlabel('Number of Terror Attack')              # label = name of label

plt.ylabel('Number of Casualties')

plt.title('Terror')            # title = title of plot

plt.show()
# Scatter Plot 

# x = Killed, y = Wounded

terror.plot(kind='scatter', x='Killed', y='Wounded',alpha = 0.5,color = 'red')

plt.xlabel('Killed')              # label = name of label

plt.ylabel('Wounded')

plt.title('Killed - Wounded Scatter Plot')            # title = title of plot
# Scatter Plot 

# x = Killed, y = Year

terror.plot(kind='scatter', x='Year', y='Killed',alpha = 0.5,color = 'red')

plt.xlabel('Year')              # label = name of label

plt.ylabel('Killed')

plt.title('Killed - Year Scatter Plot')            # title = title of plot
# Histogram

# bins = number of bar in figure

terror.Year.plot(kind = 'hist',bins = 20,figsize = (12,12))

plt.show()
plt.subplots(figsize=(15,6))

sns.countplot('Year',data=terror,palette='RdYlGn_r',edgecolor=sns.color_palette('dark',7))

plt.xticks(rotation=90)

plt.title('Number Of Terrorist Activities Each Year')

plt.show()
#create dictionary and look its keys and values

dictionary = {'fruit' : 'apple','vegatables' : 'tomato'}

print(dictionary.keys())

print(dictionary.values())
# Keys have to be immutable objects like string, boolean, float, integer or tubles

# List is not immutable

# Keys are unique



dictionary['fruit'] = "orange"    # update existing entry

print(dictionary)



dictionary['tree'] = "poplar"       # Add new entry

print(dictionary)



del dictionary['vegatables']              # remove entry with key 'vegatables'

print(dictionary)



print('tree' in dictionary)        # check include or not

dictionary.clear()                   # remove all entries in dict

print(dictionary)









# In order to run all code you need to take comment this line

# del dictionary         # delete entire dictionary    



print(dictionary)       # it gives error because dictionary is deleted
terror=pd.read_csv('../input/globalterrorismdb_0718dist.csv',encoding='ISO-8859-1')   # read data 





terror.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)

terror=terror[['Year','Month','Day','Country','Region','city','latitude','longitude','AttackType','Killed','Wounded','Target','Summary','Group','Target_type','Weapon_type','Motive']]

terror['Casualties']=terror['Killed']+terror['Wounded']   # add new info .. casualities = killed + woundered

series = terror['Year']        # data['Year'] = series

print(type(series))



data_frame = terror[['Year']]  # data[['Defense']] = data frame

print(type(data_frame))

# Comparison operator

print(6 > 4)

print(3!=2)

print(5 < 4)



# Boolean operators

print(True and False)

print(True or False)

# 1 - Filtering Pandas data frame

x = terror['Killed']>1000     # There are only 4 event which have higher killed value than 1000

terror[x]
# 2 - Filtering pandas with logical_and

# There are only 2 event which have higher killed value than 1000 and wounded  value than 1000

terror[np.logical_and(terror['Killed']>1000, terror['Wounded']>1000 )]
# This is also same with previous code line. Therefore we can also use '&' for filtering.

terror[(terror['Killed']>1000) & (terror['Wounded']>1000)]
# Stay in loop if condition( i is not equal 4) is true

i = 10

while i != 4 :

    print('i is: ',i)

    i -=1 

print(i,' is equal to 4')
# Stay in loop if condition( i is not equal 13) is true

i = 1

while i != 13 :

    print('i is: ',i)

    i +=2 

print(i,' is equal to 13')
# Stay in loop if condition( i is not equal 8) is true

lis = [1,2,3,4,5,6,7,8]

for i in lis:

    print('i is: ',i)

print('')



# Enumerate index and value of list

# index : value = 0:1, 1:2, 2:3, 3:4, 4:5

for index, value in enumerate(lis):

    print(index," : ",value)

print('')   



# For dictionaries

# We can use for loop to achive key and value of dictionary. We learnt key and value at dictionary part.

dictionary = {'fruit' : 'apple','vegatables' : 'tomato'}

for key,value in dictionary.items():

    print(key," : ",value)

print('')



# For pandas we can achieve index and value

for index,value in terror[['Killed']][0:1].iterrows():

    print(index," : ",value)

    

for index,value in terror[['Killed']][73126:73127].iterrows():

    print(index," : ",value)

    

    

    

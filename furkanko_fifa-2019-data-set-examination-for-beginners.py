import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt # plotting tool

import seaborn as sns # visualization tool



# to see .csv folders

import os 

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

data = pd.read_csv('../input/fifa19/data.csv') # provides to read data by using pandas library
data.info() # to see more details about data
data.columns #gives us columns seperately.
data.corr() # gives us "correlation between features"
#correlation map

f,ax = plt.subplots(figsize=(40,40))

sns.heatmap(data.corr(), annot= True, linewidth= 1, fmt= '.2f', ax=ax) # heatmap (visualization fucntion from "seaborn" library)

plt.show()
data.head(18207) # gives us informations about football players. "18207" can change, i just wanted to see all data.
fig, ax = plt.subplots(figsize=(8,4)) #adjusment of figure size

plt.grid() #grid on



x = data['Name'].head(10) #x axis of the chart 

y = data['Age'].head(10) #y axis of the chart

 

ax.barh(x,y) #creat horizontal bar chart 

ax.set_xlabel('Age') #label of x

ax.set_ylabel('Name') #label of y

ax.set_title('Ages of Football Players') #title of chart



plt.show()

plt.figure(figsize=(16,6)) #to adjusment size of plot

plt.grid() #to open grid



plt.scatter(data['Name'].head(10), data['Club'].head(10),s= 100, c='k', alpha= 1) #s for size of points,

                                                                                  #c for color of points,

                                                                                  #alpha for transparency of points

plt.show()
f20p= data.sort_values('Unnamed: 0', ascending = True)[['Club','Name']].head(20)

print(f20p)
sns.set(style ="dark", palette="colorblind", color_codes=True) #some adjustments

plt.figure(figsize=(20,8)) #to adjusment size of plot

ax = sns.countplot(x = 'Club', data = data.head(20), palette = 'hls') #x->data, data->from

ax.set_xlabel(xlabel="Name of Clubs", fontsize=16) #x axis label and font size

ax.set_ylabel(ylabel='Number of players', fontsize=16) #y axis label and font size

ax.set_title(label='Distribution of the top 20 players by teams', fontsize=20) #title of plot

plt.show()

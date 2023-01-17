import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#read the train.csv file into a data frame called df. 
df=pd.read_csv("/kaggle/input/train.csv")
df
#Renaming the wilderness area columns for better clarity.
df.rename(columns = {'Wilderness_Area1':'Rawah', 'Wilderness_Area2':'Neota','Wilderness_Area3':'Comanche_Peak','Wilderness_Area4':'Cache_la_Poudre'}, inplace = True) 
df.columns
df.info()
#Combining the four wilderness area columns and fourty soil type columns to Wild_area and Soil_type respectively, and removing already existing ones.
df['Wild_area'] = (df.iloc[:, 11:15] == 1).idxmax(1)
df['Soil_type'] = (df.iloc[:, 15:55] == 1).idxmax(1)
df=df.drop(columns=['Id','Rawah', 'Neota',
       'Comanche_Peak', 'Cache_la_Poudre', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',
       'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',
       'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13',
       'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17',
       'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',
       'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',
       'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29',
       'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33',
       'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',
       'Soil_Type38', 'Soil_Type39', 'Soil_Type40'])
df
df.describe()
plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=.5)

plt.subplot(3,2,1)
plt.hist(df['Elevation'],color="red")
plt.xlabel('Elevation')

plt.subplot(3,2,2)
plt.hist(df['Slope'],color="blue")
plt.xlabel('Slope')

plt.subplot(3,2,3)
plt.hist(df['Horizontal_Distance_To_Hydrology'],color="green")
plt.xlabel('Horizontal_Distance_To_Hydrology')

plt.subplot(3,2,4)
plt.hist(df['Horizontal_Distance_To_Roadways'],color="yellow")
plt.xlabel('Horizontal_Distance_To_Roadways')

plt.subplot(3,2,5)
plt.hist(df['Horizontal_Distance_To_Fire_Points'],color="violet")
plt.xlabel('Horizontal_Distance_To_Fire_Points')

plt.subplot(3,2,6)
plt.hist(df['Aspect'],color="orange")
plt.xlabel('Aspect')
Rawah=df.loc[df.Wild_area == 'Rawah',:]
Rawah.describe()
plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=.5)
plt.suptitle('RAWAH WILD AREA')

plt.subplot(3,2,1)
plt.hist(Rawah['Elevation'],color="red")
plt.xlabel('Elevation')

plt.subplot(3,2,2)
plt.hist(Rawah['Slope'],color="blue")
plt.xlabel('Slope')

plt.subplot(3,2,3)
plt.hist(Rawah['Horizontal_Distance_To_Hydrology'],color="green")
plt.xlabel('Horizontal_Distance_To_Hydrology')

plt.subplot(3,2,4)
plt.hist(Rawah['Horizontal_Distance_To_Roadways'],color="yellow")
plt.xlabel('Horizontal_Distance_To_Roadways')

plt.subplot(3,2,5)
plt.hist(Rawah['Horizontal_Distance_To_Fire_Points'],color="violet")
plt.xlabel('Horizontal_Distance_To_Fire_Points')

plt.subplot(3,2,6)
plt.hist(Rawah['Aspect'],color="orange")
plt.xlabel('Aspect')
Neota=df.loc[df.Wild_area == 'Neota',:]
Neota.describe()
plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=.5)
plt.suptitle('Neota Wilderness Area')

plt.subplot(3,2,1)
plt.hist(Neota['Elevation'],color="red")
plt.xlabel('Elevation')

plt.subplot(3,2,2)
plt.hist(Neota['Slope'],color="blue")
plt.xlabel('Slope')

plt.subplot(3,2,3)
plt.hist(Neota['Horizontal_Distance_To_Hydrology'],color="green")
plt.xlabel('Horizontal_Distance_To_Hydrology')

plt.subplot(3,2,4)
plt.hist(Neota['Horizontal_Distance_To_Roadways'],color="yellow")
plt.xlabel('Horizontal_Distance_To_Roadways')

plt.subplot(3,2,5)
plt.hist(Neota['Horizontal_Distance_To_Fire_Points'],color="violet")
plt.xlabel('Horizontal_Distance_To_Fire_Points')

plt.subplot(3,2,6)
plt.hist(Neota['Aspect'],color="orange")
plt.xlabel('Aspect')
Comanche=df.loc[df.Wild_area == 'Comanche_Peak',:]
Comanche.describe()
plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=.5)
plt.suptitle('Comanche Peak Wilderness Area')

plt.subplot(3,2,1)
plt.hist(Comanche['Elevation'],color="red")
plt.xlabel('Elevation')

plt.subplot(3,2,2)
plt.hist(Comanche['Slope'],color="blue")
plt.xlabel('Slope')

plt.subplot(3,2,3)
plt.hist(Comanche['Horizontal_Distance_To_Hydrology'],color="green")
plt.xlabel('Horizontal_Distance_To_Hydrology')

plt.subplot(3,2,4)
plt.hist(Comanche['Horizontal_Distance_To_Roadways'],color="yellow")
plt.xlabel('Horizontal_Distance_To_Roadways')

plt.subplot(3,2,5)
plt.hist(Comanche['Horizontal_Distance_To_Fire_Points'],color="violet")
plt.xlabel('Horizontal_Distance_To_Fire_Points')

plt.subplot(3,2,6)
plt.hist(Comanche['Aspect'],color="orange")
plt.xlabel('Aspect')
cache=df.loc[df.Wild_area == 'Cache_la_Poudre',:]
cache.describe()
plt.figure(figsize=(15,15))
plt.subplots_adjust(hspace=.5)
plt.suptitle('Cache la Poudre Wilderness Area')
plt.subplot(3,2,1)
plt.hist(cache['Elevation'],color="red")
plt.xlabel('Elevation')

plt.subplot(3,2,2)
plt.hist(cache['Slope'],color="blue")
plt.xlabel('Slope')

plt.subplot(3,2,3)
plt.hist(cache['Horizontal_Distance_To_Hydrology'],color="green")
plt.xlabel('Horizontal_Distance_To_Hydrology')

plt.subplot(3,2,4)
plt.hist(cache['Horizontal_Distance_To_Roadways'],color="yellow")
plt.xlabel('Horizontal_Distance_To_Roadways')

plt.subplot(3,2,5)
plt.hist(cache['Horizontal_Distance_To_Fire_Points'],color="violet")
plt.xlabel('Horizontal_Distance_To_Fire_Points')

plt.subplot(3,2,6)
plt.hist(cache['Aspect'],color="orange")
plt.xlabel('Aspect')
sns.catplot(y="Soil_type",hue="Wild_area",kind="count",palette="pastel",height=10,data=df);
sns.boxplot(y=df['Elevation'],x=df['Cover_Type'],palette="husl");
sns.boxplot(y=df['Aspect'],x=df['Cover_Type'],palette='husl');
sns.boxplot(y=df['Slope'],x=df['Cover_Type'],palette='husl');
sns.boxplot(y=df['Horizontal_Distance_To_Hydrology'],x=df['Cover_Type'],palette='husl');
sns.boxplot(y=df['Vertical_Distance_To_Hydrology'],x=df['Cover_Type'],palette='husl');
sns.boxplot(y=df['Horizontal_Distance_To_Roadways'],x=df['Cover_Type'],palette='husl');
sns.boxplot(y=df['Hillshade_9am'],x=df['Cover_Type'],palette='husl');
sns.boxplot(y=df['Hillshade_Noon'],x=df['Cover_Type'],palette='husl');
sns.boxplot(y=df['Hillshade_3pm'],x=df['Cover_Type'],palette='husl');
sns.boxplot(y=df['Horizontal_Distance_To_Fire_Points'],x=df['Cover_Type'],palette='husl');
p1=sns.countplot(data=df,x='Wild_area',hue="Cover_Type",palette='husl')
p1.set_xticklabels(p1.get_xticklabels(),rotation=15);
a=df[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Cover_Type','Wild_area']]
sns.pairplot(a,hue='Wild_area')
b=df[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Cover_Type','Wild_area']]
sns.pairplot(b,hue='Cover_Type')
sns.lmplot(data=df,x='Horizontal_Distance_To_Hydrology',y='Elevation',scatter=False,hue="Cover_Type")
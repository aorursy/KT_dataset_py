import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib

import os

import matplotlib.pyplot as plt

%matplotlib inline
os.chdir('../input')
train= pd.read_csv('train.csv')
train.head(2)
train.shape
train.columns
train.rename(columns= {'Wilderness_Area1': 'Rawah', 'Wilderness_Area2': 'Neota', 'Wilderness_Area3': 'Comanche Peak', 'Wilderness_Area4':'Cache la Poudre'}, inplace= True)
train.columns
#Combining the four wilderness area columns and fourty soil type columns to Wild_area and Soil_type respectively, and removing already existing ones.

train['Wild'] = (train.iloc[:, 11:15] == 1).idxmax(1)

train['Soil'] = (train.iloc[:, 15:55] == 1).idxmax(1)

train13 = train.drop(columns=['Id','Rawah', 'Neota',

       'Comanche Peak', 'Cache la Poudre', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',

       'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',

       'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13',

       'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17',

       'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',

       'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',

       'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29',

       'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33',

       'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',

       'Soil_Type38', 'Soil_Type39', 'Soil_Type40'])

train13
train13.describe() 
sns.countplot(train13['Cover_Type'], color= 'silver') 

#gives the number of entries in each cover type

# ==> all cover types appear to have the same frequency
sns.countplot(train13['Wild'], palette= 'spring');  
sns.scatterplot(train13['Elevation'], train13['Aspect'], hue= train13['Cover_Type'], palette= 'plasma')  

# ==> relationship between differnet variables and cover type
sns.scatterplot(train13['Vertical_Distance_To_Hydrology'], train13['Horizontal_Distance_To_Hydrology'], hue= train13['Cover_Type'], palette= 'seismic')
sns.boxplot(x= train13['Cover_Type'], y= train13['Elevation'], palette= 'magma') 
sns.boxplot(x= train13['Cover_Type'], y= train13['Slope'], palette= 'rainbow')  
sns.boxplot(x= train13['Cover_Type'], y= train13['Aspect'],notch= True, palette= 'nipy_spectral') 
ax= sns.boxplot(x= train13['Cover_Type'], y= train13['Aspect'],palette= 'nipy_spectral') 

ax= sns.swarmplot(x= train13['Cover_Type'], y= train13['Aspect'],color= 'silver') 
sns.boxplot(x= train13['Cover_Type'], y= train13['Wild'], palette= 'Dark2') 
tt= pd.read_csv('train.csv')
sns.boxplot(x= tt['Cover_Type'], y= tt['Wilderness_Area1'], palette= 'Dark2') 
sns.boxplot(x= tt['Cover_Type'], y= tt['Wilderness_Area2'], palette= 'Dark2') 
sns.boxplot(x= tt['Cover_Type'], y= tt['Wilderness_Area3'], palette= 'Dark2') 
sns.boxplot(x= tt['Cover_Type'], y= tt['Wilderness_Area4'], palette= 'Dark2') 
sns.catplot(y= 'Wild', hue= 'Cover_Type', kind= 'count', palette='viridis', height= 15, data=train13) 
X= train13[['Elevation', 'Slope', 'Aspect', 'Cover_Type']] 
sns.pairplot(X, hue= 'Cover_Type', palette= 'magma') 
X1= train13[['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',

       'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points', 'Cover_Type']] 
sns.pairplot(X1, hue= 'Cover_Type', palette= 'spring') 
sns.boxplot(x= 'Cover_Type', y= 'Horizontal_Distance_To_Hydrology', palette= 'rainbow', data= train13)
sns.boxplot(x= 'Cover_Type', y= 'Horizontal_Distance_To_Roadways', palette= 'rainbow', data= train13)
sns.boxplot(x= 'Cover_Type', y= 'Horizontal_Distance_To_Fire_Points', palette= 'rainbow', data= train13) 
sns.boxplot(x= 'Cover_Type', y= 'Vertical_Distance_To_Hydrology', palette= 'summer', data= train13)
sns.boxplot(x= 'Wild', y= 'Elevation', data= train13, palette= 'gist_earth')
sns.catplot(y = 'Elevation',x= 'Wild',hue= 'Cover_Type', data=train13) 
sns.boxplot(x= 'Wild', y= 'Horizontal_Distance_To_Hydrology', data= train13, palette= 'ocean')
sns.catplot(y = 'Horizontal_Distance_To_Hydrology',x= 'Wild',hue= 'Cover_Type', data=train13) 
sns.boxplot(x= 'Wild', y= 'Horizontal_Distance_To_Roadways', data= train13, palette= 'ocean')
sns.catplot(y= 'Horizontal_Distance_To_Roadways', x= 'Wild', hue= 'Cover_Type', data= train13)
sns.boxplot(x= 'Wild',y= 'Horizontal_Distance_To_Fire_Points',data= train13, palette= 'ocean')
sns.catplot(x= 'Wild', y= 'Horizontal_Distance_To_Fire_Points', hue= "Cover_Type", data= train13)
sns.lmplot(x= 'Elevation', y='Horizontal_Distance_To_Roadways', scatter= False, data= train13);
sns.lmplot(x='Elevation',y='Horizontal_Distance_To_Fire_Points',scatter=False, data= train13);
sns.lmplot(x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False, data= train13);
sns.lmplot(x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',hue="Wild", scatter=False,data=train13, palette= 'ocean')
sns.lmplot(x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',hue="Cover_Type",scatter=False,data= train13, palette= 'winter');
sns.boxplot(y= train13['Horizontal_Distance_To_Hydrology'],x= train13['Wild'], palette= 'viridis_r');
sns.catplot(x='Wild',y="Horizontal_Distance_To_Hydrology",hue="Cover_Type",data= train13, palette= 'viridis_r');
sns.lmplot(x='Elevation',y='Horizontal_Distance_To_Hydrology',data= train13,scatter=False);
sns.boxplot(y= train13['Vertical_Distance_To_Hydrology'],x= train13['Wild'], palette= 'viridis');
sns.catplot(x='Wild',y="Vertical_Distance_To_Hydrology",hue="Cover_Type", data= train13, palette= 'viridis');
X2= train[['Hillshade_9am', 'Hillshade_Noon',

       'Hillshade_3pm', 'Cover_Type']]
sns.pairplot(X2, hue= 'Cover_Type', palette= 'winter')
sns.boxplot(x= 'Cover_Type', y= 'Hillshade_9am', palette= 'cividis', data= train13)
sns.boxplot(x= 'Cover_Type', y= 'Hillshade_Noon', palette= 'cividis', data= train13)
sns.boxplot(x= 'Cover_Type', y= 'Hillshade_3pm', palette= 'cividis', data= train13)
sns.boxplot(x= train13['Wild'],y= train13['Hillshade_9am'], palette= 'cool');
sns.boxplot(x= train13['Wild'],y= train13['Hillshade_Noon'], palette= 'cool');
sns.boxplot(x= train13['Wild'],y= train13['Hillshade_3pm'], palette= 'cool');
sns.boxplot(x= train13['Wild'],y= train13['Slope'], palette= 'cool');
sns.boxplot(x= train13['Wild'],y= train13['Aspect'], palette= 'cool');
sns.catplot(y= 'Soil', hue= 'Cover_Type', kind= 'count', palette='viridis_r', height= 15, data=train13) 
sns.catplot(y="Soil",hue="Wild",kind="count",palette="Dark2_r",height=10,data= train13);
import numpy as np

import pandas as pd

import os

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline 

sns.set(color_codes=True)
os.chdir("../input/")
forest=pd.read_csv('eda.csv')
forest.head(10)

forest.dtypes
forest.columns
forest_new=forest.drop(['Id'],axis=1)

forest_new.head()
forest_new.shape
print(forest_new.isnull().sum())
# Therefore its clear from the above data that there are no  missing values in the given dataset which is pretty much impressive as it indicates a clean data

forest_new.rename(columns = {'Wilderness_Area1':'Rawah', 'Wilderness_Area2':'Neota','Wilderness_Area3':'Comanche_Peak ','Wilderness_Area4':'Cache_la_Poudre'}, inplace = True) 
forest_new.columns
#Combining the four wilderness area columns and fourty soil type columns to Wild_area and Soil_type respectively, and removing already existing ones.

forest_new['Wild_area'] = (forest_new.iloc[:, 11:15] == 1).idxmax(1)

forest_new['Soil_type'] = (forest_new.iloc[:, 15:55] == 1).idxmax(1)

forestnew=forest_new.drop(columns=['Rawah','Neota',

       'Comanche_Peak ', 'Cache_la_Poudre', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3', 'Soil_Type4',

       'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8', 'Soil_Type9',

       'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13',

       'Soil_Type14', 'Soil_Type15', 'Soil_Type16', 'Soil_Type17',

       'Soil_Type18', 'Soil_Type19', 'Soil_Type20', 'Soil_Type21',

       'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type25',

       'Soil_Type26', 'Soil_Type27', 'Soil_Type28', 'Soil_Type29',

       'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33',

       'Soil_Type34', 'Soil_Type35', 'Soil_Type36', 'Soil_Type37',

       'Soil_Type38', 'Soil_Type39', 'Soil_Type40'])
forestnew.columns
forestnew.describe()
#Count of number of entries of each cover type.

sns.countplot(forestnew['Cover_Type'],color="red");
#Count of the entries from different wilderness areas.

sns.countplot(forestnew['Wild_area'], palette= 'spring');
#Distribution of elevation values in the data.

sns.distplot(forestnew['Elevation'],kde=False,color='green', bins=100);

plt.ylabel('Frequency',fontsize=10)
#Distribution of aspect values in the data

sns.distplot(forestnew['Aspect'],kde=False,color='green', bins=100);

plt.ylabel('Frequency',fontsize=10)
#Distribution of frequency of various soil types in the data.

forestnew['Soil_type'].value_counts().plot(kind='barh',figsize=(10,10));

plt.xlabel('Frequency',fontsize=10)

plt.ylabel('Soil_types',fontsize=10)
#Boxplot between elevation and Cover type

sns.boxplot(y=forest_new['Elevation'],x=forest_new['Cover_Type'],palette='rainbow')
#Boxplot between Aspect and Cover type

sns.boxplot(y=forest_new['Aspect'],x=forest_new['Cover_Type'],palette='spring')
#Boxplot between Slope and Cover type

sns.boxplot(y=forest_new['Slope'],x=forest_new['Cover_Type'],palette='spring')
#Boxplot between Horizontal_Distance_To_Hydrology and Cover type

sns.boxplot(y=forest_new['Horizontal_Distance_To_Hydrology'],x=forest_new['Cover_Type'],palette='rainbow')
#Boxplot between Vertical_Distance_To_Hydrology and Cover type

sns.boxplot(y=forest_new['Vertical_Distance_To_Hydrology'],x=forest_new['Cover_Type'],palette='rainbow')
#Boxplot between Horizontal_Distance_To_Roadways and Cover type

sns.boxplot(y=forest_new['Horizontal_Distance_To_Roadways'],x=forest_new['Cover_Type'],palette='spring')
#Boxplot between Hillshade_9am and Cover type

sns.boxplot(y=forest_new['Hillshade_9am'],x=forest_new['Cover_Type'],palette='rainbow')
#Boxplot between Hillshade_Noon and Cover type

sns.boxplot(y=forest_new['Hillshade_Noon'],x=forest_new['Cover_Type'],palette='rainbow')
##Hillshade at noon values across forest types are similar, lying between 220 to 240.
#Boxplot between Hillshade_3pm and Cover type

sns.boxplot(y=forest_new['Hillshade_3pm'],x=forest_new['Cover_Type'],palette='rainbow')
#Boxplot between Horizontal_Distance_To_Fire_Points and Cover type

sns.boxplot(y=forest_new['Horizontal_Distance_To_Fire_Points'],x=forest_new['Cover_Type'],palette='rainbow')
##The median horizontal distance to firepoints varies from 1000-2000 units across all the foresttypes except for type 7.

## close proximity to firepoints could be due to the influence of roadways and thus human interference.
#Creating data frame for Degree Variables 

X_deg=forest_new[['Elevation','Aspect','Slope','Cover_Type']]
X_deg
#Creating pairplot for Degree Variables

sns.pairplot(X_deg,hue='Cover_Type',palette='ocean')
X_dist=forest_new[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways','Horizontal_Distance_To_Fire_Points','Cover_Type']]
#Creating pairplot for Degree Variables

sns.pairplot(X_dist,hue='Cover_Type',palette='spring')
#Creating data frame for Hillshade Variables 

X_hs=forest_new[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Cover_Type']]

#Creating data frame for Hillshade Variables 

X_wild=forest_new[['Wilderness_Area1','Wilderness_Area2','Wilderness_Area3','Wilderness_Area4','Cover_Type']]
#Creating pairplot for Hillshade Variables

sns.pairplot(X_wild,hue='Cover_Type',palette='spring')

#Elevation values across wilderness areas.

sns.boxplot(y=forestnew['Elevation'],x=forestnew['Wild_area']);
#Elevation values across wilderness areas and forest cover types.

sns.catplot(data=forestnew,x='Wild_area',y="Elevation",hue="Cover_Type");
#Horizontal distance to fire points across wilderness areas.

sns.catplot(data=forestnew,x='Wild_area',y="Horizontal_Distance_To_Fire_Points");
#Horizontal distance to fire points across wilderness areas and forest cover types.

sns.catplot(data=forestnew,x='Wild_area',y="Horizontal_Distance_To_Fire_Points",hue="Cover_Type");
##Horizontal distance to roadways across wilderness areas.

sns.boxplot(y=forestnew['Horizontal_Distance_To_Roadways'],x=forestnew['Wild_area']);
#Elevation values across horizontal distance to fire points.

sns.lmplot(data= forestnew,x='Elevation',y='Horizontal_Distance_To_Fire_Points',scatter=False,palette='magma');
#Elevation values across horizontal distance to roadways and elevation.

sns.lmplot(data=forestnew,x='Elevation',y='Horizontal_Distance_To_Roadways',scatter=False);
#Relationship between horizontal distance to roadways and firepoint.

sns.lmplot(data=forestnew,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False);
#Relationship between distance to firepoints and roadways across wilderness types.

sns.lmplot(data=forestnew,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False,hue="Wild_area");
sns.lmplot(x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',hue="Cover_Type",scatter=False,data= forestnew, palette= 'winter');
sns.catplot(y= 'Soil_type', hue= 'Cover_Type', kind= 'count', palette='spring', height= 15, data=forestnew) 
sns.catplot(y="Soil_type",hue="Wild_area",kind="count",palette="winter",height=10,data= forestnew);
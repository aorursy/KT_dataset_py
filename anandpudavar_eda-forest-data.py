#Importing the required libraries.
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#Reading the given csv file into a dataframe.
df=pd.read_csv("/kaggle/input/forest_train.csv")
#Getting the initial five values of the dataframe.
df.head()
#Getting the dimensionality of the dataframe.
df.shape
#Getting the summary of the data types in the data frame
df.info()
#Renaming the wilderness area columns for better clarity.
df.rename(columns = {'Wilderness_Area1':'Rawah', 'Wilderness_Area2':'Neota','Wilderness_Area3':'Comanche_Peak ','Wilderness_Area4':'Cache_la_Poudre'}, inplace = True) 
#Checking the column names
df.columns
#Combining the four wilderness area columns and fourty soil type columns to Wild_area and Soil_type respectively, and removing already existing ones.
df['Wild_area'] = (df.iloc[:, 11:15] == 1).idxmax(1)
df['Soil_type'] = (df.iloc[:, 15:55] == 1).idxmax(1)
df_forest=df.drop(columns=['Id','Rawah', 'Neota',
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
df_forest
#Checking the columns in the modified dataframe.
df_forest.columns
#Provide the general descriptive statistical values.
df_forest.describe()
#Count of number of entries of each cover type.
sns.countplot(df_forest['Cover_Type'],color="grey");
#Count of the entries from different wilderness areas.
sns.countplot(df_forest['Wild_area']);
#Distribution of elevation values in the data.
sns.distplot(df_forest['Elevation'],kde=False,color='red', bins=100);
plt.ylabel('Frequency',fontsize=10)
#Distribution of aspect values in the data
sns.distplot(df_forest['Aspect'],kde=False,color='red', bins=100);
plt.ylabel('Frequency',fontsize=10)
#Distribution of values of slope in the data.
sns.distplot(df_forest['Slope'],kde=False,color='red', bins=100);
plt.ylabel('Frequency',fontsize=10)
#Distribution of values of the horizontal distance to roadways.
sns.distplot(df_forest['Horizontal_Distance_To_Roadways'],kde=False,color='blue', bins=100);
plt.ylabel('Frequency',fontsize=10)
#Distribution of values of the horizontal distance to fire points.
sns.distplot(df_forest['Horizontal_Distance_To_Fire_Points'],kde=False,color='blue', bins=100);
plt.ylabel('Frequency',fontsize=10)
#Distribution of values of the horizontal distance to nearest surface water features.
sns.distplot(df_forest['Horizontal_Distance_To_Hydrology'],kde=False,color='green', bins=100);
plt.ylabel('Frequency',fontsize=10)
#Distribution of values of the vertical distances to nearest surface water features.
sns.distplot(df_forest['Vertical_Distance_To_Hydrology'],kde=False,color='green', bins=100);
plt.ylabel('Frequency',fontsize=10)
#Distribution of values of the hillshade index at 9am during summer solstice on an index from 0-255.
sns.distplot(df_forest['Hillshade_9am'],kde=False,color='black', bins=100);
plt.ylabel('Frequency',fontsize=10)
#Distribution of values of the hillshade index at noon during summer solstice on an index from 0-255.
sns.distplot(df_forest['Hillshade_Noon'],kde=False,color='black', bins=100);
plt.ylabel('Frequency',fontsize=10)
#Distribution of values of the hillshade index at 3pm during summer solstice on an index from 0-255.
sns.distplot(df_forest['Hillshade_3pm'],kde=False,color='black', bins=100);
plt.ylabel('Frequency',fontsize=10)
#Distribution of frequency of various soil types in the data.
df_forest['Soil_type'].value_counts().plot(kind='barh',figsize=(10,10));
plt.xlabel('Frequency',fontsize=10)
plt.ylabel('Soil_types',fontsize=10)
#Forest cover types in each wilderness areas.
a1=sns.countplot(data=df_forest,x='Wild_area',hue="Cover_Type");
a1.set_xticklabels(a1.get_xticklabels(),rotation=15);
#Elevation values across forest types.
sns.boxplot(y=df_forest['Elevation'],x=df_forest['Cover_Type']);
#Aspect values across forest types.
sns.boxplot(y=df_forest['Aspect'],x=df_forest['Cover_Type']);
#Slope values across forest types.
sns.boxplot(y=df_forest['Slope'],x=df_forest['Cover_Type']);
#Soil types across forest types.
a2=sns.catplot(y="Soil_type",hue="Cover_Type",kind="count",palette="pastel",height=15,data=df_forest);

#Horizontal distance to fire points across forest types.
sns.boxplot(data=df_forest,x='Cover_Type',y="Horizontal_Distance_To_Fire_Points");
#Horizontal distance to roadways across forest types.
sns.boxplot(y=df_forest['Horizontal_Distance_To_Roadways'],x=df_forest['Cover_Type']);
#Horizontal distance to surface water features across forest types.
sns.boxplot(y=df_forest['Horizontal_Distance_To_Hydrology'],x=df_forest['Cover_Type']);
#Vertical distance to surface water features across forest types.
sns.boxplot(y=df_forest['Vertical_Distance_To_Hydrology'],x=df_forest['Cover_Type']);
#Hillshade at 9am values across forest types.
sns.boxplot(y=df_forest['Hillshade_9am'],x=df_forest['Cover_Type']);
#Hillshade at noon values across forest types.
sns.boxplot(y=df_forest['Hillshade_Noon'],x=df_forest['Cover_Type']);
#Hillshade at 3pm values across forest types.
sns.boxplot(y=df_forest['Hillshade_3pm'],x=df_forest['Cover_Type']);
#Elevation values across wilderness areas.
a2=sns.boxplot(y=df_forest['Elevation'],x=df_forest['Wild_area']);
a2.set_xticklabels(a2.get_xticklabels(),rotation=15);
#Elevation values across wilderness areas and forest cover types.
a3=sns.catplot(data=df_forest,x='Wild_area',y="Elevation",hue="Cover_Type");
a3.set_xticklabels(rotation=65, horizontalalignment='right');
#Horizontal distance to fire points across wilderness areas.
a4=sns.boxplot(data=df_forest,x='Wild_area',y="Horizontal_Distance_To_Fire_Points");
a4.set_xticklabels(a4.get_xticklabels(),rotation=15);
#Horizontal distance to fire points across wilderness areas and forest cover types.
a5=sns.catplot(data=df_forest,x='Wild_area',y="Horizontal_Distance_To_Fire_Points",hue="Cover_Type");
a5.set_xticklabels(rotation=65, horizontalalignment='right');
##Horizontal distance to roadways across wilderness areas.
a6=sns.boxplot(y=df_forest['Horizontal_Distance_To_Roadways'],x=df_forest['Wild_area']);
a6.set_xticklabels(a6.get_xticklabels(),rotation=15);
#Relationship between wild areas and distance to roadways across forest cover types.
a7=sns.catplot(data=df_forest,x='Wild_area',y="Horizontal_Distance_To_Roadways",hue="Cover_Type");
a7.set_xticklabels(rotation=65, horizontalalignment='right');
#Elevation values across horizontal distance to fire points.
sns.lmplot(data= df_forest,x='Elevation',y='Horizontal_Distance_To_Fire_Points',scatter=False);
#Elevation values across horizontal distance to roadways and elevation.
sns.lmplot(data=df_forest,x='Elevation',y='Horizontal_Distance_To_Roadways',scatter=False);
#Relationship between horizontal distance to roadways and firepoint.
sns.lmplot(data=df_forest,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False);
#Relationship between distance to firepoints and roadways across wilderness types.
sns.lmplot(data=df_forest,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False,hue="Wild_area");
#Relationship between distance to firepoints and roadways across forest cover types.
sns.lmplot(data=df_forest,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False,hue="Cover_Type");
##Horizontal distance to hydrology across wilderness areas.
a8=sns.boxplot(y=df_forest['Horizontal_Distance_To_Hydrology'],x=df_forest['Wild_area']);
a8.set_xticklabels(a8.get_xticklabels(),rotation=15);
#Relationship between wild areas and distance to hydrology across forest cover types.
a9=sns.catplot(data=df_forest,x='Wild_area',y="Horizontal_Distance_To_Hydrology",hue="Cover_Type");
a9.set_xticklabels(rotation=65, horizontalalignment='right');
#Elevation values across horizontal distance to fire points.
sns.lmplot(data= df_forest,x='Elevation',y='Horizontal_Distance_To_Hydrology',scatter=False);
##Vertical distance to hydrology across wilderness areas.
a10=sns.boxplot(y=df_forest['Vertical_Distance_To_Hydrology'],x=df_forest['Wild_area']);
a10.set_xticklabels(a10.get_xticklabels(),rotation=15);
#Relationship between wild areas and vertical distance to hydrology across forest cover types.
a11=sns.catplot(data=df_forest,x='Wild_area',y="Vertical_Distance_To_Hydrology",hue="Cover_Type");
a11.set_xticklabels(rotation=65, horizontalalignment='right');
##Hillshade at 9am across wilderness areas.
a12=sns.boxplot(y=df_forest['Hillshade_9am'],x=df_forest['Wild_area']);
a12.set_xticklabels(a10.get_xticklabels(),rotation=15);
##Hillshade at noon across wilderness areas.
a13=sns.boxplot(y=df_forest['Hillshade_Noon'],x=df_forest['Wild_area']);
a13.set_xticklabels(a10.get_xticklabels(),rotation=15);
##Hillshade at 3pm across wilderness areas.
a14=sns.boxplot(y=df_forest['Hillshade_3pm'],x=df_forest['Wild_area']);
a14.set_xticklabels(a10.get_xticklabels(),rotation=15);
##Slope values across wilderness areas.
a15=sns.boxplot(y=df_forest['Slope'],x=df_forest['Wild_area']);
a15.set_xticklabels(a10.get_xticklabels(),rotation=15);
##Aspect values across wilderness areas.
a16=sns.boxplot(y=df_forest['Aspect'],x=df_forest['Wild_area']);
a16.set_xticklabels(a10.get_xticklabels(),rotation=15);
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train=pd.read_csv("/kaggle/input/forest-train/train.csv")
train.head()
train.columns
train.head()

#Rename the wilderness area columns.
train.rename(columns = {'Wilderness_Area1':'Rawah', 'Wilderness_Area2':'Neota','Wilderness_Area3':'Comanche_Peak ','Wilderness_Area4':'Cache_la_Poudre'}, inplace = True) 
train.columns
train['Wild_area'] = (train.iloc[:, 10:15] == 1).idxmax(1)
#Combining the four wilderness area columns
train['Soil_type'] = (train.iloc[:, 15:55] == 1).idxmax(1)
#combining fourty soil type columns
# Removing the already existing ones
train_forest=train.drop(columns=['Rawah', 'Neota',
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

train_forest.describe()
unique, counts = np.unique(train.Cover_Type, return_counts=True)
(unique, counts) # equal number of Type
fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))
train[['Cover_Type', 'Elevation']].groupby(['Cover_Type']).mean().plot(kind='bar', ax=ax1, color='k')
train[['Cover_Type', 'Aspect']].groupby(['Cover_Type']).mean().plot(kind='bar', ax=ax2, color='b')
train[['Cover_Type', 'Slope']].groupby(['Cover_Type']).mean().plot(kind='bar', ax=ax3, color='r')
label=['Cover ' + str(x) for x in range(1,8)]
for i in range(7):
    ax = plt.hist(train.Elevation[train.Cover_Type==i+1],label=label[i], bins=20,stacked=True)
plt.legend()
plt.xlabel('Elevation (m)')
# Elevation feature has an important weight.
#since we can already distinguish the type 3, 4 and 7 with only this attribute.
#entries from different wilderness areas.
sns.countplot(train_forest['Wild_area']);

# Comanche Peak wild Area occur the most and Neota occurs least

# number of entries of each cover type.
sns.countplot(train_forest['Cover_Type'],palette="rainbow");

#shows same frequency
#Distribution of aspect values:
sns.distplot(train_forest['Aspect'],kde=False,color='blue', bins=100);
plt.ylabel('Frequency',fontsize=10)

#most of the values lying between 0-100 and 275-350.
#Distribution of elevation values:
sns.distplot(train_forest['Elevation'],kde=False,color='pink', bins=100);
plt.ylabel('Frequency',fontsize=10)
#data collected from relatively high altitude areas
#peaks at the intervals of 2000m-2500m, 2500m-3000m,3000m-3500m and tapering at both ends.
#Distribution of values of slope
sns.distplot(train_forest['Slope'],kde=False,color='maroon', bins=100);
plt.ylabel('Frequency',fontsize=10)

#Right skewed(positive), highest frequency around 10 at the slope
#Distribution of values of the horizontal distance to roadways.
sns.distplot(train_forest['Horizontal_Distance_To_Roadways'],kde=False,color='green', bins=100);
plt.ylabel('Frequency',fontsize=10)
#most of the samples are within the range of 0-2000 distance to the roadways.
#This indicates chances of commercial exploitation of the forests by human activities.
#Distribution of values of the horizontal distance to fire points.
sns.distplot(train_forest['Horizontal_Distance_To_Fire_Points'],kde=False,color='red', bins=100);
plt.ylabel('Frequency',fontsize=10)
#This positive skewed distribution indicates the horizontal distance to wildfire ignition point mostly around 0-2000
#thus indicates some influence of human activities in the proximities
#Distribution of values of the horizontal distance to nearest surface water features.
sns.distplot(train_forest['Horizontal_Distance_To_Hydrology'],kde=False,color='aqua', bins=100);
plt.ylabel('Frequency',fontsize=10)
#peak near zero: This means that most of the samples are present very close to water sources.
#Distribution of values of the vertical distances to nearest surface water features.
sns.distplot(train_forest['Vertical_Distance_To_Hydrology'],kde=False,color='blue', bins=100);
plt.ylabel('Frequency',fontsize=10)

# shows a positively skewed distribution with a sharp peak at 0.
#Distribution of values of the hillshade index at 9am during summer solstice on an index from 0-255.
sns.distplot(train_forest['Hillshade_9am'],kde=False,color='orange', bins=100);
plt.ylabel('Frequency',fontsize=10)

# shows a negtively skewed distribution, peaking at around 225 in the range 100 to 250.
sns.distplot(train_forest['Hillshade_Noon'],kde=False,color='orange', bins=100);
plt.ylabel('Frequency',fontsize=10)

#negtively skewed distribution, peak around 225 in the range 125-250.
sns.distplot(train_forest['Hillshade_3pm'],kde=False,color='orange', bins=100);
plt.ylabel('Frequency',fontsize=10)

# shows a more or less symmetric distribution

color = ['b','r','k','y','m','c','g']
for i in range(7):
    plt.scatter(train.Hillshade_Noon[train.Cover_Type==i+1], train.Hillshade_3pm[train.Cover_Type==i+1], color=color[i], label='Type' +str(i+1))
plt.xlabel('Hillshade_Noon')
plt.ylabel('Hillshade_3pm')
plt.legend()
#Shows a scatterplot where several values are equal to 0 which may be the missing values and replaced with mean of the attribute.
sns.countplot(data=train_forest,x='Wild_area',hue="Cover_Type");
#Boxplot between elevation and Cover type
sns.boxplot(y=train['Elevation'],x=train['Cover_Type'],palette='rainbow')
#Boxplot between slope and Cover type
sns.boxplot(y=train['Slope'],x=train['Cover_Type'],palette='rainbow')

#Boxplot between Aspect and Cover type
sns.boxplot(y=train['Aspect'],x=train['Cover_Type'],palette='rainbow')

#Creating data frame for Degree Variables 
train_deg=train[['Elevation','Aspect','Slope','Cover_Type']]
sns.pairplot(train_deg,hue='Cover_Type')
#Boxplot between Horizontal_Distance_To_Roadways and Cover type
sns.boxplot(y=train['Horizontal_Distance_To_Roadways'],x=train['Cover_Type'],palette='nipy_spectral')
#Boxplot between Horizontal_Distance_To_Hydrology and Cover type
sns.boxplot(y=train['Horizontal_Distance_To_Hydrology'],x=train['Cover_Type'],palette='nipy_spectral')
#Boxplot between Vertical_Distance_To_Hydrology and Cover type
sns.boxplot(y=train['Vertical_Distance_To_Hydrology'],x=train['Cover_Type'],palette='nipy_spectral')
#Boxplot between Horizontal_Distance_To_Fire_Points and Cover type
sns.boxplot(y=train['Horizontal_Distance_To_Fire_Points'],x=train['Cover_Type'],palette='nipy_spectral')
#Creating data frame for Distance Variables 
train_dist=train[['Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology','Horizontal_Distance_To_Roadways',
          'Horizontal_Distance_To_Fire_Points','Cover_Type']]

#Creating pairplot for Degree Variables
sns.pairplot(train_dist,hue='Cover_Type')
#Boxplot between Hillshade_Noon and Cover type
sns.boxplot(y=train['Hillshade_Noon'],x=train['Cover_Type'],palette='plasma')
#All values are lying between 220 to 240.
#Boxplot between Hillshade_3pm and Cover type
sns.boxplot(y=train['Hillshade_3pm'],x=train['Cover_Type'],palette='rainbow')
#values are ranging between 100 to 150.
#Boxplot between Hillshade_3pm and Cover type
sns.boxplot(y=train['Hillshade_9am'],x=train['Cover_Type'],palette='rainbow')
#values are in similar range of around 200 to 250.
#Creating data frame for Hillshade Variables 
train_hs=train[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Cover_Type']]
#Creating pairplot for Hillshade Variables
sns.pairplot(train_hs,hue='Cover_Type')
#Elevation values across wilderness areas and forest cover types.
sns.catplot(data=train_forest,x='Wild_area',y="Elevation",hue="Cover_Type");
#Horizontal distance to fire points across wilderness areas and forest cover types.
sns.catplot(data=train_forest,x='Wild_area',y="Horizontal_Distance_To_Fire_Points",hue="Cover_Type");
#Relationship between wild areas and distance to roadways across forest cover types.
sns.catplot(data=train_forest,x='Wild_area',y="Horizontal_Distance_To_Roadways",hue="Cover_Type");
#Relationship between distance to firepoints and roadways across forest cover types.
sns.lmplot(data=train_forest,x='Horizontal_Distance_To_Fire_Points',y='Horizontal_Distance_To_Roadways',scatter=False,hue="Cover_Type");
# plotting Soil types based on forest cover types.
sns.catplot(y="Soil_type",hue="Cover_Type",kind="count",palette="viridis",height=15,data=train_forest);
# plotting Soil types based on Wilderness areas.
sns.catplot(y="Soil_type",hue="Wild_area",kind="count",palette="plasma",height=15,data=train_forest);
# Heatmap 
size = 10
mat = dataset_train.iloc[:,:size].corr()
f, ax = plt.subplots(figsize = (10,8))
sns.heatmap(mat,vmax=0.8,square=True);



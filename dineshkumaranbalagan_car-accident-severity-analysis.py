#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
import matplotlib.cm as cm
import matplotlib.colors as colors
print('Libraries Imported')
#importing datasets
Accidents= pd.read_csv('../input/dft-accident-data/Accidents0515.csv',index_col='Accident_Index')
Casualities=pd.read_csv('../input/dft-accident-data/Casualties0515.csv',error_bad_lines=False,index_col='Accident_Index',warn_bad_lines=False)
Vehicles=pd.read_csv('../input/dft-accident-data/Casualties0515.csv',error_bad_lines=False,index_col='Accident_Index',warn_bad_lines=False)
print('Datasets Imported')
Accidents.head()
Accidents.shape
Accidents.columns
#dropping unwanted columns:
Accidents.drop(['Location_Easting_OSGR', 'Location_Northing_OSGR','LSOA_of_Accident_Location',
                'Junction_Control' ,'2nd_Road_Class','Did_Police_Officer_Attend_Scene_of_Accident'], axis=1, inplace=True)
Accidents.info()
#checking for any null values
Accidents.isnull().sum()
#dropping all rows that contains null values in it:
#Here dropping null rows doesn't affect the processing: 
Accidents=Accidents.dropna()
Accidents.shape
Casualities.head()
Casualities.isnull().sum()
Casualities.info()
Casualities.shape
Casualities.columns
Correlation = Accidents.corr()
plt.figure(figsize=(20,10))
sns.heatmap(Correlation, annot=True)
#Distribution of accients based on week:
plt.figure(figsize=(10,5))
Accidents['Day_of_Week'].hist(color='purple')
plt.grid(alpha=0.4)
#Accidents severity based on road type:
plt.figure(figsize=(10,5))
ax=sns.countplot('Road_Type',hue='Accident_Severity',data=Accidents)
ax.set_xticklabels(['Roundabout','One_way_street ','Dual_carriageway','Single carriageway','Slip road','Unknown'])
plt.xticks(rotation=90)
plt.legend(['Fatal','Serious','Slight'])
plt.grid(alpha=0.4)
#Scatter plot of Longitude/Latitude
plt.figure(figsize=(10,5))
plt.scatter(x='Latitude',y='Longitude',data=Accidents,c='orange')
plt.xlabel('Latitude',fontsize=12)
plt.ylabel('Longitude',fontsize=12)
plt.grid(alpha=0.4)
plt.figure(figsize=(10,15))
sns.jointplot(x='Weather_Conditions',y='Number_of_Casualties',data=Accidents)
plt.figure(figsize=(10,5))
ax=sns.countplot('Light_Conditions',data=Accidents,color='orange') 
ax.set_xticklabels(['Daylight','Darkness - lights lit',
                    'Darkness - lights unlit','Darkness - no lighting','Darkness - lighting unknown'])
plt.xticks(Rotation=90)
plt.title('ACCIDENT RATES BASED ON LIGHT CONDITIONS',fontsize=15)
plt.grid(alpha=0.4)
plt.show()
correlation=Casualities.corr()
plt.figure(figsize=(25,8))
sns.heatmap(Correlation,annot=True)
plt.figure()
Casualities.hist(figsize=(15,15));
#Distrubution of casualities based on age:
plt.figure(figsize=(20,5))
sns.countplot('Age_of_Casualty',data=Casualities)
plt.title('CASUALITY DISTRIBUTION BASED ON AGE', fontsize=15)
plt.xticks(rotation=90)
plt.grid(alpha=0.4)
plt.show()
plt.figure(figsize=(40,10))
sns.countplot('Age_of_Casualty',hue='Sex_of_Casualty',data=Casualities)
plt.xticks(fontsize=15,rotation=90)
plt.legend(['Missing data','Male','Female'],prop={'size': 30}, loc=1)
plt.grid(alpha=0.4)
plt.xlabel('AGE_OF_CASUALITIES', fontsize=25)
plt.ylabel('COUNT', fontsize=25)
plt.show()
plt.figure()
ax=sns.countplot('Casualty_Class', data=Casualities)
ax.set_xticklabels(['Driver or rider','Passenger','Pedestrian'])
plt.grid(alpha=0.4)
plt.show()
Vehicles.head()
Vehicles.drop(['Vehicle_Reference',
       'Casualty_Reference', 'Casualty_Class', 'Sex_of_Casualty',
       'Age_of_Casualty', 'Age_Band_of_Casualty', 'Casualty_Severity',
       'Pedestrian_Location', 'Pedestrian_Movement', 'Car_Passenger',
       'Bus_or_Coach_Passenger', 'Pedestrian_Road_Maintenance_Worker',
       'Casualty_Type', 'Casualty_Home_Area_Type'], axis=1, inplace=True)
Dataframe1=Accidents.merge(Casualities, right_index=True, left_index=True)
Dataframe2=Dataframe1.merge(Vehicles, right_index=True, left_index=True)
Dataframe2.head()
Dataframe2.columns
plt.figure(figsize=(10,5))
ax=sns.countplot('Accident_Severity',hue='Sex_of_Casualty',data=Dataframe1)
ax.set_xticklabels(['Fatal','Serious','Slight'])
plt.legend(['Unlabelled','Male','Female'],fontsize=12)
plt.title('ACCIDENT SEVERITY DISTRIBUTION BASED ON SEX', fontsize=15)
plt.grid(alpha=0.4)
plt.figure(figsize=(10,5))
ax=sns.countplot('Accident_Severity',hue='Casualty_Class',data=Dataframe1)
plt.legend(['Driver or rider','Passenger','Pedestrian'],fontsize=12)
ax.set_xticklabels(['Fatal','Serious','Slight'])
plt.title('ACCIDENT SEVERITY DISTRIBUTION BASED ON CASUALITY CLASS', fontsize=15)
plt.grid(alpha=0.4)
plt.figure(figsize=(10,5))
ax=sns.countplot('Accident_Severity', hue='Light_Conditions', data=Dataframe2)
plt.legend(['Daylight','Darkness - lights lit','Darkness - lights unlit','Darkness - no lighting','Darkness - lighting unknown'],fontsize=12)
ax.set_xticklabels(['Fatal','Serious','Slight'])
plt.title('ACCIDENT SEVERITY DISTRIBUTION BASED ON LIGHT CONDITIONS', fontsize=15)
plt.grid(alpha=0.4)
plt.figure(figsize=(10,5))
sns.countplot('Accident_Severity', hue='Road_Type', data=Dataframe2)
ax.set_xticklabels(['Fatal','Serious','Slight'])
plt.legend(['Roundabout','One way street','Dual carriageway','Single carriageway','Slip road','Unknown'], fontsize=12)
plt.title('ACCIDENT SEVERITY DISTRIBUTION BASED ON ROAD TYPE', fontsize=15)
plt.grid(alpha=0.4)
#Scatter plot of Longitude/Latitude
plt.figure(figsize=(10,5))
sns.jointplot(x='Latitude',y='Longitude',kind = 'scatter',data=Dataframe2)
plt.xlabel('Latitude',fontsize=12)
plt.ylabel('Longitude',fontsize=12)
plt.grid(alpha=0.4)
plt.figure(figsize=(10,5))
ax=sns.countplot('Accident_Severity', hue='Speed_limit', data=Dataframe2)
plt.grid(alpha=0.4)
ax.set_xticklabels(['Fatal','Serious','Slight'])
plt.legend(loc=2) 
Dataframe3=Dataframe2[['Latitude','Longitude']].dropna()
locationlist = Dataframe3.values.tolist()
len(locationlist)
locationlist[7]
Dataframe3.shape
import folium
from folium.plugins import MarkerCluster
m = folium.Map(location=[51.5085300,-0.1257400], tiles='openstreetmap', zoom_start=15)
marker_cluster = MarkerCluster().add_to(m)
for i in range(0,len(locationlist)):
    folium.CircleMarker(locationlist[i],radius = float(Dataframe2["Accident_Severity"].values[0]/1e7),
                        popup="Accident Severity : %s"%Dataframe2["Accident_Severity"].values[0],color="red",fill_color='pink').add_to(m)
m
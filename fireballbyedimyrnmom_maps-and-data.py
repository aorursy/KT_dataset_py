# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd



#Area schools include

Schools = pd.read_csv("../input/tacoma-schools/Schools.csv")

Schools.head(3)
Hate_Biases = pd.read_csv("../input/hatebias-tacoma/City_of_Tacoma_Hate_Bias__2013-2019_.csv")

Fires = pd.read_csv("../input/tacoma-fires/Tacoma_Fire_Department_-_Fire_Incident_Data_Table.csv")
#Tacoma-specific

Fires.head(2)
#Tacoma-specific

Hate_Biases.head(3)
a=pd.concat([Schools, Fires, Hate_Biases], axis=1)

a.head(3)
#drop excess columns

b=a.drop(['OBJECTID', 'Case Number','DIST_NO','Occurred On Day', 'Occurred On','Offense Status','GRADE','Disposition','PRS_ID', 'District', 'Age', 'Gender', 'Entity Name', 'WEBSITE', 'PHONE', 'Block', 'Sector', 'Resident Status', 'Offense Type', 'Ethnicity', 'Address of Incident (XY)'], axis=1)

b.head(2)   
#Still too large, but also need several of these columns 



c=b.drop( ['Mobile_VehicleYear', 'Fire_OutsideAreaAffected','Structure_FireFloorOfOrigin','IncidentNumber', 'EstimatedContentLoss', 'Structure_BuildingStatus', 'IncidentID', 'Mobile_VehicleMake', 'Mobile_VehicleModel', 'X_COORD', 'IncidentDate', 'Structure_NumberOfOccupants', 'Structure_NumberOfBusinesses', 'Y_COORD', 'Structure_TotalSQFootBurned', 'Structure_TotalSQFootSmokeDamage', 'Structure_ConstructionType','Structure_AlarmInvolved', 'Structure_FloorOfOriginDescription', 'Structure_BurnDamage', 'Structure_SmokeDamage', 'Structure_AlarmType', 'Structure_NumberOfBasementLevels', 'Structure_NumberOfStories', 'Structure_AlarmEffectiveness', 'Structure_TotalSqFootage'], axis=1)

c.head(2)
#Pick specific desired columns for a dataframe

df = c[['Address','Location', 'ZIP','ZipCode','Latitude','Longitude','NAME','Hate Bias', 'CITY', 'EstimatedTotalFireLoss']].copy()

df.head(3)
df.describe()
#Explore the columns

df.apply (pd.Series.value_counts)
df['NAME'].value_counts(dropna=False)
df['ZipCode'].value_counts(dropna=False)
df['CITY'].value_counts(dropna=False)
import numpy as np

df=df.fillna(0)

#find and change NaNs to a zero
df1=df[df.CITY == 'TACOMA']

df1.tail(3) #reflects the bottom rows
#These entries are in Tacoma but lack the LAT LONGS 

error1= df1[df1.Latitude == 0]

error1.head(2)
df1.Location[280]
#Update data to include the Lat and Long availiavle in the Location column

df1.at[280,'Latitude']= 47.18027

df1.at[280,'Longitude']=-122.426748

df1.tail(3)
#Repeat for other LAT/LON rows 

df1.Location[265]
#Update data to include the Lat and Long available in the Location column

df1.at[104,'Latitude']= 47.174562

df1.at[104,'Longitude']=-122.43407

df1.at[105,'Latitude']=47.216222

df1.at[105,'Longitude']=-122.425035



df1.at[129,'Latitude']= 47.222461

df1.at[129,'Longitude']=-122.477576



df1.at[268,'Latitude']=47.121624

df1.at[268,'Longitude']=-122.493964

df1.at[265,'Latitude']=47.220936

df1.at[265,'Longitude']=-122.337246
#Verify changes

error2= df1[df1.Latitude == 0]

error2.head(2)

#Still some to go..
##df1.Location[228]

df1.Location[245]
df1.at[173,'Latitude']= 47.25396

df1.at[173,'Longitude']=-122.419891

df1.at[174,'Latitude']= 47.204561

df1.at[174,'Longitude']=-122.433263

df1.at[179,'Latitude']=47.228756

df1.at[179,'Longitude']=-122.499617



df1.at[186,'Latitude']=47.211272

df1.at[186,'Longitude']=-122.449777

df1.at[201,'Latitude']=47.261692

df1.at[201,'Longitude']=-122.463316

df1.at[228,'Latitude']=47.252557

df1.at[228,'Longitude']=-122.495463



df1.at[240,'Latitude']=47.200213

df1.at[240,'Longitude']=-122.468269

df1.at[245,'Latitude']=47.230702

df1.at[245,'Longitude']=-122.31735
df1.head(3)
df2=df1.drop(['Location'], axis=1)

df2.head(3)
#Clean dataframe

errs1= df2[df2.ZipCode == 0]

errs1.head()
#Update the ZipCode column

df2.at[64,'ZipCode']= 98405

df2.at[71,'ZipCode']= 98406

df2.at[79,'ZipCode']= 98409

df2.at[83,'ZipCode']= 98445

df2.at[90,'ZipCode']= 98443

df2.at[92,'ZipCode']= 98405

df2.at[93,'ZipCode']= 98422

df2.at[94,'ZipCode']= 98444

df2.at[95,'ZipCode']= 98404

df2.at[96,'ZipCode']= 98444
df2.at[3,'ZipCode']= 98403

df2.at[4,'ZipCode']= 98409

df2.at[7,'ZipCode']= 98408

df2.at[9,'ZipCode']= 98405

df2.at[10,'ZipCode']= 98405

df2.at[11,'ZipCode']= 98409

df2.at[19,'ZipCode']= 98408

df2.at[20,'ZipCode']= 98404

df2.at[23,'ZipCode']= 98404

df2.at[24,'ZipCode']= 98444
df2.at[26,'ZipCode']= 98422

df2.at[27,'ZipCode']= 98405

df2.at[33,'ZipCode']= 98446

df2.at[38,'ZipCode']= 98446

df2.at[41,'ZipCode']= 98467

df2.at[45,'ZipCode']= 98444

df2.at[46,'ZipCode']= 98402

df2.at[51,'ZipCode']= 98446

df2.at[55,'ZipCode']= 98404

df2.at[58,'ZipCode']= 98422
#Verified changes

errs2= df2[df2.ZipCode == 0]

errs2.head()
import matplotlib.pyplot as plt

import seaborn as sns



sns.countplot(df2['ZipCode'])
sns.scatterplot(x='ZipCode',y='EstimatedTotalFireLoss',data=df2, color='red')
df3=df2.drop(['ZIP'], axis=1)

df3.head(3)
df3['Hate Bias'].value_counts()
from sklearn.preprocessing import LabelEncoder

#It requires the category ‘object’ column to become of ‘category’ type before running it.



df3['Hate Bias']= df3['Hate Bias'].astype('category')# Assigning numerical value 

df3['Hate Bias catg'] = df3['Hate Bias'].cat.codes

df3.head(3)
import folium

#Black and white map

Tacoma = folium.Map(location=[47.2529, -122.4443],

                   tiles = "Stamen Toner", zoom_start = 12)



Tacoma 
Tac = folium.Map(location=[47.2529, -122.4443],

                   zoom_start = 12)

#Schools as markers

##Mark the first school in df3= Annie Wright School

folium.Marker([47.248468, -122.444963], popup='Annie Wright').add_to(Tac)

Tac
#repeat for all schools using a loop

for index, row in df.iterrows(): #using df 

    if row['NAME']!=0: #to avoid an error      

        folium.Marker([row['Latitude'], row['Longitude']], popup=row['NAME']).add_to(Tac)

Tac
#Heat map

from folium import plugins

from folium.plugins import HeatMap



# Ensure the data is float (float64 may not work as is)

df3['Latitude'] = df3['Latitude'].astype(float)

df3['Longitude'] = df3['Longitude'].astype(float)

#Filters

##Hate Bias reports

heatMap = df3[df3['Hate Bias catg']>0] 

heatMap.head(3)
#a new map without school markers

Tac1 = folium.Map(location=[47.2529, -122.4443],

                   zoom_start = 12)
# List comprehension to make out list of lists

heat_data = [[row['Latitude'],row['Longitude']] for index, row in heatMap.iterrows()]



# Plot it on the map

HeatMap(heat_data).add_to(Tac1)



Tac1
heatMap2 = df3[df3['EstimatedTotalFireLoss']>0] 

heatMap2.head()
#a new map for fires

Tac2 = folium.Map(location=[47.2529, -122.4443],

                   zoom_start = 12)
heatMap2 = df3[df3['EstimatedTotalFireLoss']>0] 

heat_data1 = [[row['Latitude'],row['Longitude']] for index, row in heatMap2.iterrows()]



# Plot 

HeatMap(heat_data1).add_to(Tac2)



Tac2
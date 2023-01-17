!pip install KeplerGl
import pandas as pd

data = pd.read_csv("../input/nypd-complaint/NYPD_Complaint_Map__Year_to_Date_.csv")

#data.head()

print(data.shape)



import geopandas as gpd

#df = gpd.read_file("../input/nypd-complaint/NYPD_Complaint_Map__Year_to_Date_.csv")

#df.head()



data = data.drop(columns=['PARKS_NM'])

data = data.drop(columns=['HADEVELOPT'])



data = data[pd.notnull(data['Latitude'])]

data = data[pd.notnull(data['Longitude'])]



data.dropna(axis='columns')

print(data.shape)

print(data.sample(5))



########get sample data set ######

data = data.sample(100000)

############## default map

from keplergl import KeplerGl

map = KeplerGl()

########## map for the csv data 



map.add_data(data=data)

map

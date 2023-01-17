!pip install hdbscan
!pip install pycountry-convert

from geopy.geocoders import Nominatim
import pycountry_convert
import geopandas as gp
import pandas as pd
import numpy as np
import hdbscan
valid_countries_dict = pycountry_convert.map_countries(cn_name_format="default")
valid_countries = [(k, v['alpha_2'], v['alpha_3']) for k, v in valid_countries_dict.items()]
countryDF = pd.DataFrame(valid_countries, columns=["country","alpha2","alpha3"])
countryDF.drop_duplicates('alpha2', inplace=True)
countryDF.reset_index(inplace=True)
countryDF.loc[countryDF['alpha3']=='BOL','country']='Bolivia'
countryDF.loc[countryDF['alpha3']=='VAT','country']='Vatican'
countryDF.loc[countryDF['alpha3']=='TWN','country']='Taiwan'
countryDF.loc[countryDF['alpha3']=='VIR','country']='Virgin Islands'
geoloc = Nominatim(user_agent="area_clustering")

def getCoordinates(countryName):    
    try:
        info = geoloc.geocode(countryName)    
        return [info.latitude, info.longitude]
    except:
        print('Error: Country {} not found'.format(countryName))        
        return [0, 90]
coordinates = countryDF["country"].apply(getCoordinates)
countryDF = pd.concat([countryDF, pd.DataFrame(coordinates.tolist(), columns=['latitude', 'longitude'])], axis=1)
countryDF["latitude"] = countryDF["latitude"].apply(np.radians)
countryDF["longitude"] = countryDF["longitude"].apply(np.radians)
def cluster_data(dataframe, min_cluster_size, min_samples):
    clus = hdbscan.HDBSCAN(metric='haversine', min_cluster_size=min_cluster_size, min_samples=min_samples)    
    dataframe.loc[:,'groups'] = clus.fit_predict(dataframe[["latitude","longitude"]])    
    print('n_groups: {}, unclustered objects: {}'.format(max(dataframe['groups']), sum(dataframe['groups']==-1)))
    return dataframe
countryDF = cluster_data(dataframe=countryDF, min_cluster_size=5, min_samples=3)
rec = countryDF.loc[countryDF['groups']==-1].copy()
rec = cluster_data(dataframe=rec, min_cluster_size=3, min_samples=1)
rec.loc[rec['groups']>=0,'groups']+=np.max(countryDF['groups']+1)
countryDF[countryDF['groups']==-1]=rec
countryDF['groups'].value_counts()
countryDF[countryDF['groups']==-1]
countryDF['groups']=countryDF['groups']+1
countryDF.head()
countryDF.to_csv('area_mapping.csv', columns=['alpha3','groups'], index=False)
our_mapping = pd.read_csv('area_mapping.csv')
our_mapping.head()
our_dict = pd.Series(our_mapping.groups.values, index = our_mapping.alpha3).to_dict()

def get_area(iso_code):
    try:
        return our_dict[iso_code]
    except KeyError:
        return 0
print(get_area('POL'))
print(get_area('not valid'))
world = gp.read_file(gp.datasets.get_path('naturalearth_lowres'))
world['groups']=0
world.head(10)
for _id in range(len(world)):
    number=countryDF.index[countryDF['alpha3']==world.loc[_id,'iso_a3']]
    
    if len(number)>0:        
        tmp = countryDF.loc[number,'groups']
        world.at[_id,'groups'] = tmp
world.head(10)
n_groups = max(our_mapping.groups)+1
shuffled = np.random.permutation(n_groups)
categories_dict = {_id: shuffled[_id]  for _id in range(n_groups)}
world['groups'] = world['groups'].replace(categories_dict)
world.head(10)
ax = world.plot(color='white', edgecolor='black', figsize=(20,20))
plot = world.plot(ax=ax, column='groups', cmap='jet')
wun=world[world['groups']==categories_dict[0]]
ax = world.plot(color='white', edgecolor='black', figsize=(20,20))
plot = wun.plot(ax=ax, column='groups', cmap='jet', vmin=0,vmax=1)
world.loc[world.iso_a3=='RUS']
world.loc[world.groups==world.loc[world.iso_a3=='RUS'].groups.values[0]]
def reGroup(iso_a3_list):
    new_group_id = max(world.groups)+1
    
    for item in iso_a3_list:
        world.loc[world.iso_a3==item,'groups']=new_group_id    
reGroup(['RUS', 'MNG'])
ax = world.plot(color='white', edgecolor='black', figsize=(20,20))
plot = world.plot(ax=ax, column='groups', cmap='jet')
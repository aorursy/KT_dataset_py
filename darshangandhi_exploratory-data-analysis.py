import pandas as pd
import numpy as np 
!type pip
!type python
data=pd.read_csv("ZomatoRestaurantsIndia.csv")
data
data.shape
data.head(5)
# data.locality_verbose
data.columns

#country_id
#city
#country_id
#locality_verbose
#currency
#opentable_support

data=data.drop(['country_id','locality_verbose','currency','opentable_support'],axis=1)

data.shape
data
data.dtypes
#TO CATEGORICAL FORMAT

data=data.astype('category')

#TO NUMERICAL FORMAT

# average_cost_for_two
# aggregate_rating
# photo_count
# votes

data.average_cost_for_two=data.average_cost_for_two.astype('int64')
data.aggregate_rating=data.aggregate_rating.astype('float64')
data.photo_count=data.photo_count.astype('int64')
data.votes=data.votes.astype('int64')



data.dtypes
#this is done based on all columns in the data
data_duplicated=data[data.duplicated()]
data_duplicated
#let us do for city column 

data_city_dup=data[data['city'].duplicated()]
data_city_dup
data.name
data.establishment
data.city
data.city.value_counts()

data.locality
data.latitude
data.latitude.isna().sum()
data.latitude=data.latitude.astype('Float64')
data.loc[(data.latitude<8.4)| (data.latitude >37.6),['latitude']]=None
data.latitude.isna().sum()
data.longitude
data.longitude.isna().sum()
data.longitude=data.longitude.astype('Float64')
data.loc[(data.longitude<68.7)| (data.longitude >97.25),['longitude']]=None
data.longitude.isna().sum()
!pip install git+git://github.com/geopandas/geopandas.git
import matplotlib.pyplot as plt
import geopandas

gdf = geopandas.GeoDataFrame(
    data, geometry=geopandas.points_from_xy(data.longitude, data.latitude))
gdf

world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
ax = world[world.name == 'India'].plot(
    color='black', edgecolor='white')
gdf.plot(ax=ax, color='yellow',figsize=(50,50))

data.cuisines
def cuisines(x):
    save_cuisines=(list(map(str,(x))))
    result_cuisines=set()
    for i in save_cuisines:
        inter=(i.split(", "))
        for j in inter:
            result_cuisines.add(j)


    return (result_cuisines)


result_cuisines=cuisines(data.cuisines)

print(result_cuisines)
store_cuisine_inter=[]

df_cuisine=pd.DataFrame(result_cuisines,columns=['Cuisines'])
for i in result_cuisines:
    store_cuisine_inter.append(data['cuisines'].str.count(i).sum())


df_cuisine['frequency']=store_cuisine_inter
df_cuisine
data.average_cost_for_two
data.price_range
data.price_range.value_counts().plot.pie()
data.highlights
def highlights(x):
    save_highlights=(list(map(str,(x))))
    result_highlights=set()
    for i in save_highlights:
        inter=(i.split(", "))
        for j in inter:
            result_highlights.add(j)


    return (result_highlights)
ans_highlights=highlights(data.highlights)
store_inter=[]

df_highlights=pd.DataFrame(ans_highlights,columns=['Facility'])
for i in ans_highlights:
    store_inter.append(data['highlights'].str.count(i).sum())


df_highlights
# print(store_inter)
df_highlights['frequency']=store_inter
df_highlights
save_df=df_highlights.sort_values(by=['frequency'],ascending=False).head(1)
save_df['Facility'].head(1)
data.aggregate_rating
data.rating_text
data.rating_text.unique()

data.loc[((data.aggregate_rating>=0) & (data.aggregate_rating<=1)),'rating_text_new']='poor'
data.loc[((data.aggregate_rating>=1) & (data.aggregate_rating<=2)),'rating_text_new']='average'
data.loc[((data.aggregate_rating>=2) & (data.aggregate_rating<=3)),'rating_text_new']='good'
data.loc[((data.aggregate_rating>=3) & (data.aggregate_rating<=4)),'rating_text_new']='very good'
data.loc[((data.aggregate_rating>=4) & (data.aggregate_rating<=5)),'rating_text_new']='excellent'
data['rating_text_new'].unique()
data.votes
data.photo_count
data.delivery
data.isna().sum()
data.describe()
import seaborn as sns
corr=data.corr()
sns.heatmap(corr,annot=True)

data.plot.box()

data.res_id=data.res_id.astype('int64')
data['res_id'].plot.box()
data['photo_count'].plot.box()
data['votes'].plot.box()
data['aggregate_rating'].plot.box()
east_zone=['Arunachal Pradesh', 'Assam', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Sikkim' , 'Tripura','Bihar', 'Orissa', 'Jharkhand', 'West Bengal ']

west_zone=['Rajasthan' , 'Gujarat', 'Goa', 'Maharashtra', 'Daman and Diu',
'Dadra and Nagar Haveli','Madhya Pradesh','Chhattisgarh']

north_zone=['Jammu and Kashmir', 'Himachal Pradesh','Chandigarh','Delhi', 'Punjab', 'Uttarakhand' , 'Uttar Pradesh','Haryana']

south_zone=['Andhra Pradesh', 'Karnataka', 'Kerala','Tamil Nadu','Telangana','puducherry']


# data.locality
dict_region={'Agra':"north", 'Ahmedabad':'west', 'Ajmer':'west', 'Alappuzha':'south', 'Allahabad':'north', 'Amravati':"west",
       'Amritsar':'north', 'Aurangabad':'west', 'Bangalore':'south', 'Bhopal':'west', 'Bhubaneshwar':'east',
       'Chandigarh':'north', 'Chennai':'south', 'Coimbatore':'south', 'Cuttack':'east', 'Darjeeling':'east',
       'Dehradun':'north', 'Dharamshala':'north', 'Faridabad':"north", 'Gandhinagar':"west", 'Gangtok':'east',
       'Ghaziabad':"north", 'Goa':"west", 'Gorakhpur':"north", 'Greater Noida':"north", 'Guntur':"south",
       'Gurgaon':"north", 'Guwahati':"east", 'Gwalior':"east", 'Haridwar':"north", 'Howrah':"east",
       'Hyderabad':"south", 'Indore':"west", 'Jabalpur':"west", 'Jaipur':"west", 'Jalandhar':"north", 'Jammu':"north",
       'Jamnagar':"west", 'Jamshedpur':"east", 'Jhansi':"north", 'Jodhpur':"west", 'Junagadh':"west",
       'Kanpur':"north", 'Kharagpur':"east", 'Kochi':"south", 'Kolhapur':"west", 'Kolkata':"east", 'Kota':"west",
       'Lucknow':"north", 'Ludhiana':"north", 'Madurai':"south", 'Manali':"north", 'Mangalore':"south", 'Manipal':"south",
       'Meerut':"north", 'Mohali':"north", 'Mumbai':"west", 'Mussoorie':"north", 'Mysore':"south", 'Nagpur':"west",
       'Nainital':"north", 'Nashik':"west", 'Navi Mumbai':"west", 'Nayagaon':"north", 'Neemrana':"west",
       'New Delhi':"north", 'Noida':"north", 'Ooty':"south", 'Palakkad':"south", 'Panchkula':"north", 'Patiala':"north",
       'Patna':"north", 'Puducherry':"south", 'Pune':"west", 'Pushkar':"west", 'Raipur':"west", 'Rajkot':"west",
       'Ranchi':"east", 'Rishikesh':"north", 'Salem':"south", 'Secunderabad':"south", 'Shimla':"north",
       'Siliguri':"east", 'Srinagar':"north", 'Surat':"west", 'Thane':"west", 'Thrissur':"south", 'Tirupati':"south",
       'Trichy':'south', 'Trivandrum':'south', 'Udaipur':'west', 'Udupi':"south", 'Vadodara':"west", 'Varanasi':"north",
       'Vellore':"south", 'Vijayawada':"south", 'Vizag':"south", 'Zirakpur':"north"}
data.drop(data[data.city=='north'].index,inplace=True)
save_region=[]
for i in data.city:
    save_region.append(dict_region[i])
# print(len(save_region))
data['region']=save_region
data

city_data=data.groupby('region')['city'].unique()
city_data
data[['name','city','average_cost_for_two','price_range']].head(100)
save_costly=data[['city','average_cost_for_two']].groupby(['city']).mean()

values_for_graph=save_costly.sort_values(by=['average_cost_for_two'],ascending=False).head(5)

values_for_graph['city']=values_for_graph.index

values_for_graph
sns.barplot(x='city',y='average_cost_for_two',data=values_for_graph)
data_high_north=(data[['highlights']][data.region=='north'])
ans_north=(highlights(data_high_north.highlights))
store_inter_north=[]

df_highlights_north=pd.DataFrame(ans_north,columns=['Facility'])

for i in ans_north:
    store_inter_north.append(data_high_north['highlights'].str.count(i).sum())

df_highlights_north['frequency']=store_inter_north
df_highlights_north.sort_values(by=['frequency'],ascending=False)
data_high_east=(data[['highlights']][data.region=='east'])
ans_east=(highlights(data_high_east.highlights))
store_inter_east=[]

df_highlights_east=pd.DataFrame(ans_east,columns=['Facility'])
for i in ans_east:
    store_inter_east.append(data_high_east['highlights'].str.count(i).sum())

df_highlights_east['frequency']=store_inter_east
df_highlights_east.sort_values(by=['frequency'],ascending=False)
data_high_south=(data[['highlights']][data.region=='south'])
ans_south=(highlights(data_high_south.highlights))
store_inter_south=[]

df_highlights_south=pd.DataFrame(ans_south,columns=['Facility'])
for i in ans_south:
    store_inter_south.append(data_high_south['highlights'].str.count(i).sum())

df_highlights_south['frequency']=store_inter_south
df_highlights_south.sort_values(by=['frequency'],ascending=False)
data_high_west=(data[['highlights']][data.region=='west'])
ans_west=(highlights(data_high_west.highlights))
store_inter_west=[]

df_highlights_west=pd.DataFrame(ans_west,columns=['Facility'])
for i in ans_west:
    store_inter_west.append(data_high_west['highlights'].str.count(i).sum())

df_highlights_west['frequency']=store_inter_west
df_highlights_west.sort_values(by=['frequency'],ascending=False)
#for NORTH

import seaborn as sns
save_north=df_highlights_north.sort_values(by=['frequency'],ascending=False).head(10)
sns.barplot(x='Facility',y='frequency',data=save_north)
#for EAST

import seaborn as sns
save_east=df_highlights_east.sort_values(by=['frequency'],ascending=False).head(10)
sns.barplot(x='Facility',y='frequency',data=save_east)
#for SOUTH
import seaborn as sns
save_south=df_highlights_south.sort_values(by=['frequency'],ascending=False).head(10)
sns.barplot(x='Facility',y='frequency',data=save_south)
#for WEST
import seaborn as sns
save_west=df_highlights_west.sort_values(by=['frequency'],ascending=False).head(10)
sns.barplot(x='Facility',y='frequency',data=save_west)
data_cui_north=(data[['cuisines']][data.region=='north'])
ans_north=(cuisines(data_cui_north.cuisines))
store_inter_north=[]

df_cui_north=pd.DataFrame(ans_north,columns=['cuisines'])

for i in ans_north:
    store_inter_north.append(data_cui_north['cuisines'].str.count(i).sum())

df_cui_north['frequency']=store_inter_north
df_cui_north.sort_values(by=['frequency'],ascending=False)
data_cui_east=(data[['cuisines']][data.region=='east'])
ans_east=(cuisines(data_cui_east.cuisines))
store_inter_east=[]

df_cui_east=pd.DataFrame(ans_east,columns=['cuisines'])
for i in ans_east:
    store_inter_east.append(data_cui_east['cuisines'].str.count(i).sum())

df_cui_east['frequency']=store_inter_east
df_cui_east.sort_values(by=['frequency'],ascending=False)
data_cui_south=(data[['cuisines']][data.region=='south'])
ans_south=(highlights(data_cui_south.cuisines))
store_inter_south=[]

df_cui_south=pd.DataFrame(ans_south,columns=['cuisines'])
for i in ans_south:
    store_inter_south.append(data_cui_south['cuisines'].str.count(i).sum())

df_cui_south['frequency']=store_inter_south
df_cui_south.sort_values(by=['frequency'],ascending=False)
data_cui_west=(data[['cuisines']][data.region=='west'])
ans_west=(cuisines(data_cui_west.cuisines))
store_inter_west=[]

df_cui_west=pd.DataFrame(ans_west,columns=['cuisines'])
for i in ans_west:
    store_inter_west.append(data_cui_west['cuisines'].str.count(i).sum())

df_cui_west['frequency']=store_inter_west
df_cui_west.sort_values(by=['frequency'],ascending=False)
#for NORTH

import seaborn as sns
save_north=df_cui_north.sort_values(by=['frequency'],ascending=False).head(10)
sns.barplot(x='cuisines',y='frequency',data=save_north)
#for EAST

import seaborn as sns
save_east=df_cui_east.sort_values(by=['frequency'],ascending=False).head(10)
sns.barplot(x='cuisines',y='frequency',data=save_east)
#for SOUTH

import seaborn as sns
save_south=df_cui_south.sort_values(by=['frequency'],ascending=False).head(10)
sns.barplot(x='cuisines',y='frequency',data=save_south)
#for WEST

import seaborn as sns
save_west=df_cui_west.sort_values(by=['frequency'],ascending=False).head(10)
sns.barplot(x='cuisines',y='frequency',data=save_west)

df_cui_north.sort_values(by=['frequency'],ascending=False).head(10)
data_north=data[data.region=='north']
data_north.head(5)

data_north[['aggregate_rating','rating_text','photo_count','votes']].sort_values(by=['photo_count','votes'],ascending=False).head(50)
data_for_boxplot=data_north[['aggregate_rating','photo_count','votes']]
ax = sns.boxplot(data=data_for_boxplot, orient="h", palette="Set2")
sns.boxplot(x='photo_count',y='aggregate_rating',data=data_for_boxplot)
sns.boxplot(x='votes',y='aggregate_rating',data=data_for_boxplot)
df_mum=data[data.city=='Mumbai']
df_mum.head()
df_mum['name'][df_mum.average_cost_for_two>5000].head()
save_ans=df_mum[['name','average_cost_for_two']].sort_values(by=['average_cost_for_two'],ascending=False).drop_duplicates().head(5)
save_ans
# sns.barplot(x='name',y='average_cost_for_two',data=save_ans)

ans_plot=save_ans.plot.bar(x='name',y='average_cost_for_two')
ans_plot
mumbai_cuisines=cuisines(df_mum['cuisines'])
print(mumbai_cuisines)

store_cuisine_inter_mum=[]

df_cuisine_mumbai=pd.DataFrame(mumbai_cuisines,columns=['Cuisines_Of_Mumbai'])
for i in mumbai_cuisines:
    store_cuisine_inter_mum.append(df_mum['cuisines'].str.count(i).sum())


df_cuisine_mumbai['frequency']=store_cuisine_inter_mum
df_cuisine_mumbai.sort_values(by=['frequency'],ascending=False).head(20)
df_mum_local=df_mum[['locality']]
df_mum_local['locality'].value_counts().head(5)
import seaborn as sns
new_corr=df_mum[['aggregate_rating','average_cost_for_two']]
corr_new=new_corr.corr()
sns.heatmap(corr_new,annot=True)
sns.set(rc={'figure.figsize':(11,8)})
sns.boxplot(x='photo_count',y='establishment',data=data)
sns.set(rc={'figure.figsize':(30,1)})
#for WEST
data_high_west=(data[['highlights']][data.region=='west'])
ans_west=(highlights(data_high_west.highlights))
store_inter_west=[]

df_highlights_west=pd.DataFrame(ans_west,columns=['Facility'])
for i in ans_west:
    store_inter_west.append(data_high_west['highlights'].str.count(i).sum())

df_highlights_west['frequency']=store_inter_west
df_highlights_west.sort_values(by=['frequency'],ascending=False)


import seaborn as sns
save_west=df_highlights_west.sort_values(by=['frequency'],ascending=False).head(10)
sns.barplot(x='Facility',y='frequency',data=save_west)
sns.set(rc={'figure.figsize':(30,10)})

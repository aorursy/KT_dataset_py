import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
plt.style.use('seaborn')

%matplotlib inline
plz_shape_df = gpd.read_file('../input/realstate/plz-gebiete.shp', dtype={'plz': str})
plz_shape_df['lat'] = plz_shape_df['geometry'].centroid.y
plz_shape_df['lon'] = plz_shape_df['geometry'].centroid.x
plz_shape_df.head()
plt.rcParams['figure.figsize'] = [16, 11]

# Get lat and lng of Germany's main cities. 
top_cities = {
    'Berlin': (13.404954, 52.520008), 
    'Cologne': (6.953101, 50.935173),
    'Düsseldorf': (6.782048, 51.227144),
    'Frankfurt am Main': (8.682127, 50.110924),
    'Hamburg': (9.993682, 53.551086),
    'Leipzig': (12.387772, 51.343479),
    'Munich': (11.576124, 48.137154),
    'Dortmund': (7.468554, 51.513400),
    'Stuttgart': (9.181332, 48.777128),
    'Nuremberg': (11.077438, 49.449820),
    'Hannover': (9.73322, 52.37052)
}
# Create feature distance to closest big city
# Create feature east or west of germany
def get_closest_city(lat, lon):
    distances = []
    for city in top_cities:
        #print(top_cities[city][0])
        dist = np.sqrt( (lat - top_cities[city][1]) ** 2 + (lon - top_cities[city][0]) ** 2)
        distances.append(dist)
    
    return np.min(distances)
    
plz_shape_df['distance'] =  plz_shape_df.apply(lambda x: get_closest_city(x['lat'], x['lon']), axis=1)
plz_shape_df['east'] =  plz_shape_df.apply(lambda x: 1 if x['lon'] > 10.35 else 0,  axis=1)
plz_shape_df['west'] =  plz_shape_df.apply(lambda x: 1 if x['lon'] <= 10.35 else 0,  axis=1)

plz_shape_df.head()
for name in ['east', 'west']:
    if plz_shape_df[name].isna().values.any():
        print("there is na", name)
# Display on how the segmentation of the country was made
plz_shape_df = plz_shape_df \
    .assign(first_dig_plz = lambda x: x['plz'].str.slice(start=0, stop=1))
plz_shape_df['first_dig_plz'] = plz_shape_df['first_dig_plz'].astype(float)
fig, ax = plt.subplots()

plz_shape_df.plot(
    ax=ax, 
    column='first_dig_plz', 
    categorical=True, 
    legend=True, 
    legend_kwds={'title':'First Digit', 'loc':'lower right'},
    cmap='tab20',
    alpha=0.9
)

for c in top_cities.keys():

    ax.text(
        x=top_cities[c][0], 
        y=top_cities[c][1] + 0.08, 
        s=c, 
        fontsize=12,
        ha='center', 
    )

    ax.plot(
        top_cities[c][0], 
        top_cities[c][1], 
        marker='o',
        c='black', 
        alpha=0.5
    )

ax.set(
    title='Germany First-Digit-Postal Codes Areas', 
    aspect=1.3,
    facecolor='white'
);
# Merge data.
plz_region_df = pd.read_csv(
    '../input/realstate/zuordnung_plz_ort.csv', 
    sep=',', 
    dtype={'plz': str}
)

plz_region_df.drop('osm_id', axis=1, inplace=True)

plz_region_df.head()

germany_df = pd.merge(
    left=plz_shape_df, 
    right=plz_region_df, 
    on='plz',
    how='inner'
)
germany_df.drop(['note'], axis=1, inplace=True)
germany_df[germany_df['ort'].str.match('Kö')]

for name in ['east', 'west']:
    if germany_df[name].isna().values.any():
        print("there is na", name)
germany_df.shape
cities_df = gpd.read_file('../input/realstate/de.csv')
cities_df.index = cities_df.city
cities_df = cities_df.rename(
    index={'Munich': 'München', 'Cologne': 'Köln', 'Frankfurt': 'Frankfurt am Main'},
    columns={'city': 'ort', 'lat': 'lat_city', 'lng': 'lon_city'})
cities_df['ort'] = cities_df.index
cities_df['lat_city'] = cities_df['lat_city'].astype(float)
cities_df['lon_city'] = cities_df['lon_city'].astype(float)
cities_df = cities_df[['ort', 'lat_city', 'lon_city']]
cities_df.head()
cities_df[cities_df['ort'].str.match('Kö')]

print(germany_df.shape)
germany_df = pd.merge(
    left=germany_df, 
    right=cities_df, 
    on='ort',
    how='left'
)
print(germany_df.shape)
germany_df.head()

germany_df['east_city'] =  germany_df.apply(lambda x: 1 if x['lon_city'] < x['lon'] else 0,  axis=1)
germany_df['north_city'] =  germany_df.apply(lambda x: 1 if x['lat_city'] <= x['lat'] else 0,  axis=1)


germany_df[germany_df['ort'].str.match('Münche')]
plz_einwohner_df = pd.read_csv(
    '../input/realstate/plz_einwohner.csv', 
    sep=',', 
    dtype={'plz': str, 'einwohner': int}
)

plz_einwohner_df.head()
master_data = pd.read_csv(
    '../input/realstate/master_data.csv', 
    sep=',', 
    dtype={'plz': str, 
           'einwohner_plz': int
          }
)

master_data.head()

osm_plz_df = pd.read_csv(
    '../input/realstate/osm.csv', 
    sep=',', 
    dtype={'zip_code': str, 
           #'einwohner_plz': int
          }
)
osm_plz_df = osm_plz_df.rename(columns={'zip_code': 'plz'})
amenities_df = osm_plz_df.groupby(['plz', 'amenity']).size().reset_index(name='count')
zip_code = osm_plz_df.groupby(['plz']).size().reset_index(name='osm')
amenities_df.sort_values('plz', ascending=True)
# Merge data.
# habitants per zip code
germany_df2 = pd.merge(
    left=germany_df, 
    right=plz_einwohner_df, 
    on='plz',
    how='left'
)
# macro factors by distric
germany_df2 = pd.merge(
    left=germany_df2, 
    right=master_data, 
    on='plz',
    how='left'
)

amenities = osm_plz_df['amenity'].unique()


for amenity in amenities:
# restuarants, cafe, hospital, doctor, fast_food
    df = amenities_df[amenities_df.amenity == amenity]
    df[amenity] = df['count'].clip(upper=300)
    #df.fillna(0)
    germany_df2 = pd.merge(
        left=germany_df2, 
        right=df[['plz', amenity]], 
        on='plz',
        how='left'
    )
    #germany_df2[amenity] = germany_df2[amenity].fillna(0)
    
# bus, train, university
tags = ['university', 'train_station', "'bus': 'yes'"]
for tag in tags:
    df = osm_plz_df[osm_plz_df['tag'].str.contains(tag)].groupby(['plz']).size().reset_index(name=tag)
    germany_df2 = pd.merge(
        left=germany_df2, 
        right=df[['plz', tag]], 
        on='plz',
        how='left'
    )
    #germany_df2[tag] = germany_df2[tag].fillna(0)
    
tags = ['university', 'train_station', 'bus']
germany_df2 = germany_df2.rename(columns={"'bus': 'yes'": 'bus'})

# total
germany_df2 = pd.merge(
    left=germany_df2, 
    right=zip_code[['plz', 'osm']], 
    on='plz',
    how='left'
)
# total
germany_df2 = pd.merge(
    left=germany_df2, 
    right=zip_code[['plz', 'osm']], 
    on='plz',
    how='left'
)
germany_df2 = germany_df2.drop_duplicates(subset='plz')
for name in ['east', 'east_city']:
    if germany_df2[name].isna().values.any():
        print("there is na", name)
# data should be splitted into train and test before scaling

def fill_missing(df):
    columns = list(df.columns)
    datatypes = ['geometry', 'Gangelt']
    df = df.replace(r'[-|#|x]', np.nan, regex=True)

    for name, types in zip(df.columns, df.dtypes):
        #print(name)
        if name != 'plz' and str(types) not in datatypes:
            try:
                df[name] = df[name].astype(float)
                df[name] = df[name].fillna(df[name].mean())
                #df[name] = (df[name] - df[name].min())/(df[name].max() - df[name].min())
            except:
                pass
        #print(df[name].dtypes)
        #print("there is nan", df[name].isnull().values.any())


    return df

germany_df3 = fill_missing(germany_df2)

def plot_country(germany_df, col_name):
    
    fig, ax = plt.subplots()
    germany_df.plot(
        ax=ax, 
        column=col_name, 
        categorical=False, 
        legend=True, 
        cmap='GnBu',
        #alpha=100
    )

    for c in top_cities.keys():

        ax.text(
            x=top_cities[c][0], 
            y=top_cities[c][1] + 0.08, 
            s=c, 
            fontsize=12,
            ha='center', 
        )

        ax.plot(
            top_cities[c][0], 
            top_cities[c][1], 
            marker='o',
            c='black', 
            alpha=0.5
        )
    ax.set(
        title=('Germany: %s per Postal Code' % col_name), 
        aspect=1.3, 
        facecolor='lightblue'
    );
plot_country(germany_df2, 'east')
def plot_city(df, city_name, col_name, col_name2=None):
    query = 'ort ==  "%s"' % city_name
    print(query)
    berlin_df = df.query(query)
    
    if col_name2 is not None:
        fig, (ax1, ax2) = plt.subplots(1,2)
    else:
        fig, ax1 = plt.subplots()
    berlin_df.plot(
        ax=ax1, 
        column=col_name, 
        categorical=False, 
        legend=True, 
        cmap='GnBu',
        #scheme='quantiles',
        #k=5
    )
    title = '%s: Number of %s per Postal Code' % (city_name, col_name)
    ax1.set(
        title=title, 
        aspect=1.3,
        facecolor='lightblue'
    );
    
    if col_name2 is not None:
        berlin_df.plot(
            ax=ax2, 
            column=col_name2, 
            categorical=False, 
            legend=True, 
            cmap='GnBu',
        )

        ax2.set(
            title = '%s: Number of %s per Postal Code' % (city_name, col_name2),
            aspect=1.3,
            facecolor='lightblue'
        );
    
plot_city(germany_df2, "Frankfurt am Main", "east_city")
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import statsmodels.stats.api as sms
import statsmodels.formula.api as smf
import statsmodels.api as sm
data = pd.read_csv("../input/germany-housing-rent-and-price-data-set-apr-20/apr20_price.csv")

data["obj_yearConstructed"] = data["obj_yearConstructed"].astype(float)

x = data.copy()
x = x[x["obj_yearConstructed"] >= 2000] 

x = x[x["obj_noRooms"] <= 6]
x = x[x["obj_purchasePrice"] >= 10000]
x = x[x["obj_purchasePrice"] < x["obj_purchasePrice"].quantile(0.99) ]


btype =["single_family_house","multi_family_house","semidetached_house","mid_terrace_house"]

x = x[x["obj_buildingType"].isin(btype)]
x['geo_plz'] = x['geo_plz'].astype(int)
x['geo_plz'] = x['geo_plz'].astype(str)
x['geo_plz'] = x['geo_plz'].apply(lambda x: x.zfill(5))
x['plz'] = x['geo_plz']


avg_price_df = x.groupby(['obj_regio1', 'plz']).mean().reset_index(); 


x_copy = x.copy()
x_copy = pd.merge(
    left=x_copy, 
    right=germany_df2, 
    on='plz',
    how='left'
)
  

    
z = pd.merge(
    left=germany_df3, 
    right=avg_price_df, 
    on='plz',
    how='left'
)
x_copy = x_copy[x_copy['obj_purchasePrice'].notna()]
x_copy['log_price'] = np.log(x_copy['obj_purchasePrice'])
x_copy = x_copy.replace(r'[-|#|x]', np.nan, regex=True)
x_copy.dropna(inplace=True, subset=['east', 'east_city', 'north_city'])
print(x_copy.shape)
x_copy.head()

print(x_copy['east'].isna().sum())
print(z['east'].isna().sum())
master_data.columns

print("zip code specif variables")

amenity = amenities[1:-1]
zip_code_columns = [*amenity, *tags, 'east', 'north_city', 'east_city', 'distance', 'einwohner', 'qkm/plz', 'gdp_habitant']


!pip install libpysal
!pip install esda
import libpysal as lps

#print(z.columns)
#z = fill_missing(z)
wq =  lps.weights.Queen.from_dataframe(z[['geometry', *zip_code_columns]])
wq.transform = 'r'
#wq.sparse, y.shape
# remove island, no neighbors 
z1 = z.copy()



z1['log_price'] = np.log(z1['obj_purchasePrice'])
y = z1['log_price']

y_lag = lps.weights.lag_spatial(wq, y)
z1['y_lag'] = y_lag
z1['einwohner_lag'] = lps.weights.lag_spatial(wq, z1['einwohner'])
import mapclassify as mc

plot_city(z1, "Berlin", 'einwohner_lag', 'einwohner')


import esda
z3 = z1[z1['y_lag'] > 2]
z3 = fill_missing(z3) # data filled with average
wq3 =  lps.weights.Queen.from_dataframe(z3[['geometry', *zip_code_columns]])
mi = esda.moran.Moran(z3['log_price'], wq3)
mi.I ## moran i 
import seaborn as sbn
sbn.kdeplot(mi.sim, shade=True)
plt.vlines(mi.I, 0, 1, color='r')
plt.vlines(mi.EI, 0,1)
plt.xlabel("log transform of price")
plt.title("Moran's I")
mi.p_sim
lag_nan_max = np.nanmax(y_lag) ; lag_nan_min = np.nanmin(y_lag) ;lag_nan_mean = np.nanmean(y_lag)

plt.plot(z1['log_price'], z1['y_lag'], '.', color = 'firebrick') 

plt.vlines(z1['log_price'].mean(),lag_nan_min,lag_nan_max,linestyle = '--')
plt.hlines(z1['y_lag'].mean(), z1['log_price'].min(), z1['log_price'].max(), linestyle="--")

#plt.title('Moran Scatterplot')
#plt.ylabel('Spatial Lag Price')
#plt.xlabel('Price')
plt.show()

z2 = z1[z1['y_lag'] > 2]

#plt.plot(z2['log_price'], z2['y_lag'], '.', color = 'firebrick') 
ax = sns.regplot(x=z2['log_price'], y=z2['y_lag'], color="firebrick")

plt.vlines(z2['log_price'].mean(), 
           z2['y_lag'].min(),
           z2['y_lag'].max(),
           linestyle = '--')

plt.hlines(z2['y_lag'].mean(),
           z2['log_price'].min(),
           z2['log_price'].max(),
           linestyle="--")

#plt.title('Moran Scatterplot')
#plt.ylabel('Spatial Lag Price')
#plt.xlabel('Price')
plt.show()
li = esda.moran.Moran_Local(z3['log_price'], wq3)
sig = li.p_sim < 0.05
hotspot = sig * li.q==1
coldspot = sig * li.q==3
doughnut = sig * li.q==2
diamond = sig * li.q==4
spots = ['n.sig.', 'hot spot']
labels = [spots[i] for i in hotspot*1]
z3['labels'] = labels
len(labels)
#from matplotlib import colors
#hmap = colors.ListedColormap(['red', 'lightgrey'])

plot_city(z3, "Berlin", "labels")
residentital_b_type = ['residencial_b_1_p', 'residencial_b_2_p', 'residencial_b_3_p']
new_apartment_permint = [ 'new_apartment_permit_1', 'new_apartment_permit_2_3', 'new_apartment_permit_4_5', 'new_apartment_permit_6']

floor_area = [ 'floor_area_settlement', 'floor_area_traffic', 'floor_area_vegetation']
floor_use = ['floor_use_industry_commerce', 'floor_use_leisure']

people = ['age_3', 'age_3_6', 'age_6_15', 'age_15-18', 'age_18_25', 'age_25_30', 'age_30_40', 'age_40_50', 'age_50_60', 'age_60_75', 'age_75_x']
new_apartment_permit = ['new_apartment_permit', 'new_apartment_permit_1', 'new_apartment_permit_2_3', 'new_apartment_permit_4_5', 'new_apartment_permit_6']

other = ['plz', 'note', 'city', 'landkreis id', 'ags', 'ags_text', 'state id',
           'state name', 'region id', 'region/city', 'district id', 'DG',
           'district name', 'qkm/plz', 'einwohner/ plz', 'gdp_habitant', 
           'residencial_building', 'floor_area', 'new_apartment_permit',
          'house_hold_size', 'floor_use', 'house_hold_size_1', 'floor_use_residential', 'tax_payers', 'income_total', 'house_hold_size_6_p',
          'house_hold_type_1', 'house_hold_type_2', 'house_hold_type_3', 'house_hold_type_4',  'floor_use_residential', 'wage_n_income', 'gpd_employee']



distric_columns = [c for c in master_data.columns if c not in other]
print("distric specific variables")
print(sorted(distric_columns))

features = [*zip_code_columns, *distric_columns]
df = x_copy.copy()
df = df.replace(r'[-|#|x]', np.nan, regex=True)
df[features] = df[features].astype(float)
df.head()
print(people)
print(floor_area)
print(floor_use)
print(residentital_b_type)
print(new_apartment_permint)
zip_code_columns = [*amenity, *tags]
distric_columns = [c for c in master_data.columns if c not in other]


features2 = ['distance', 'einwohner']
features3 = []
name = 'einwohner'
df_habitant = df.copy()

factor = df_habitant['einwohner'] / df_habitant['habitants']

factor_2 = germany_df3['einwohner'].astype(float) / germany_df3['habitants'].astype(float)
df_habitant['density'] = df_habitant['einwohner'] / df_habitant['qkm']

for name in people:
    df_habitant[name] = df_habitant[name].astype(float) 
    
    df_habitant[name + '_hab'] = df_habitant[name] * factor
    germany_df3[name + '_hab'] = germany_df3[name] * factor_2
    
for name in [*amenity, *tags]:
    
    df_habitant[name] = df_habitant[name].astype(float) 
    df_habitant[name + '_hab'] = df_habitant[name] * factor
    germany_df3[name + '_hab'] = germany_df3[name] * factor_2
    
    features2.append(name + '_hab')
  

df_habitant['pop_density'] = df_habitant['einwohner'] / df_habitant['qkm/plz']
germany_df3['pop_density'] = germany_df3['einwohner'] / germany_df3['qkm/plz']



for name in distric_columns:
    if name not in [*floor_area, *floor_use, 'habitants', *people]:
        df_habitant[name] = df_habitant[name].astype(float) 
        df_habitant[name + '_hab'] = df_habitant[name] * factor
        germany_df3[name + '_hab'] = germany_df3[name] * factor_2
        if name not in [ *residentital_b_type]:
            features2.append(name + '_hab')

name = 'qkm/plz'
factor = df_habitant['qkm/plz'] / df_habitant['qkm']
factor_2 = germany_df3['qkm/plz'] / germany_df3['qkm']
for name in [*floor_area, *floor_use]:
    df_habitant[name] = df_habitant[name].astype(float) 
    df_habitant[name + '_qkm_plz'] = df_habitant[name] * factor
    germany_df3[name + '_qkm_plz'] = germany_df3[name] * factor_2
    
    features3.append(name + '_qkm_plz')


        
# interactions on an observations level
df_habitant['young'] = df_habitant['age_3_hab'] + df_habitant['age_3_6_hab'] + df_habitant['age_6_15_hab'] + df_habitant['age_15-18_hab'] + df_habitant['age_18_25_hab']
df_habitant['middle_age'] = df_habitant['age_25_30_hab'] + df_habitant['age_30_40_hab'] + df_habitant['age_40_50_hab']
df_habitant['old_age'] = df_habitant['age_50_60_hab'] + df_habitant['age_60_75_hab'] + df_habitant['age_75_x_hab']
df_habitant['young_ratio']= df_habitant['young'] / ( df_habitant['young']  + df_habitant['middle_age'] + df_habitant['old_age'])
df_habitant['middle_ratio']= df_habitant['middle_age'] / ( df_habitant['young']  + df_habitant['middle_age'] + df_habitant['old_age'])
df_habitant['old_age_ratio'] = df_habitant['old_age'] / ( df_habitant['young']  + df_habitant['middle_age'] + df_habitant['old_age'])
df_habitant['floor_area_per_veg'] =  df_habitant['floor_area_vegetation'] / ( df_habitant['floor_area_settlement'] + df_habitant['floor_area_traffic'] + df_habitant['floor_area_vegetation'] )                                                                           
df_habitant['salary_per_employed'] = df_habitant['income_total'].astype(float)  / df_habitant['employed'].astype(float)  
df_habitant['residencial_b_1_p_ratio'] = df_habitant['residencial_b_1_p_hab'] / ( df_habitant['residencial_b_1_p_hab'] +  df_habitant['residencial_b_2_p_hab'] +  df_habitant['residencial_b_3_p_hab']) 
df_habitant['residencial_b_2_p_ratio'] = df_habitant['residencial_b_2_p_hab'] / ( df_habitant['residencial_b_1_p_hab'] +  df_habitant['residencial_b_2_p_hab'] +  df_habitant['residencial_b_3_p_hab']) 
df_habitant['residencial_b_3_p_ratio'] = df_habitant['residencial_b_3_p_hab'] / ( df_habitant['residencial_b_1_p_hab'] +  df_habitant['residencial_b_2_p_hab'] +  df_habitant['residencial_b_3_p_hab']) 

# iteractions zip code level
germany_df3['young'] = germany_df3['age_3_hab'] + germany_df3['age_3_6_hab'] + germany_df3['age_6_15_hab'] + germany_df3['age_15-18_hab'] + germany_df3['age_18_25_hab']
germany_df3['middle_age'] = germany_df3['age_25_30_hab'] + germany_df3['age_30_40_hab'] + germany_df3['age_40_50_hab']
germany_df3['old_age'] = germany_df3['age_50_60_hab'] + germany_df3['age_60_75_hab'] + germany_df3['age_75_x_hab']
germany_df3['young_ratio']= germany_df3['young'] / ( germany_df3['young']  + germany_df3['middle_age'] + germany_df3['old_age'])
germany_df3['middle_ratio']= germany_df3['middle_age'] / ( germany_df3['young']  + germany_df3['middle_age'] + germany_df3['old_age'])
germany_df3['old_age_ratio'] = germany_df3['old_age'] / ( germany_df3['young']  + germany_df3['middle_age'] + germany_df3['old_age'])
germany_df3['floor_area_per_veg'] =  germany_df3['floor_area_vegetation'] / ( germany_df3['floor_area_settlement'] + germany_df3['floor_area_traffic'] + germany_df3['floor_area_vegetation'] )                                                                           
germany_df3['salary_per_employed'] = germany_df3['income_total'].astype(float)  / germany_df3['employed'].astype(float)  
germany_df3['residencial_b_1_p_ratio'] = germany_df3['residencial_b_1_p_hab'] / ( germany_df3['residencial_b_1_p_hab'] +  germany_df3['residencial_b_2_p_hab'] +  germany_df3['residencial_b_3_p_hab']) 
germany_df3['residencial_b_2_p_ratio'] = germany_df3['residencial_b_2_p_hab'] / ( germany_df3['residencial_b_1_p_hab'] +  germany_df3['residencial_b_2_p_hab'] +  germany_df3['residencial_b_3_p_hab']) 
germany_df3['residencial_b_3_p_ratio'] = germany_df3['residencial_b_3_p_hab'] / ( germany_df3['residencial_b_1_p_hab'] +  germany_df3['residencial_b_2_p_hab'] +  germany_df3['residencial_b_3_p_hab']) 

features2.append('young_ratio')
features2.append('middle_ratio')
features2.append('old_age_ratio')
features2.append('floor_area_per_veg')
features2.append('salary_per_employed')

features2.append('residencial_b_1_p_ratio')
features2.append('residencial_b_2_p_ratio')
features2.append('residencial_b_3_p_ratio')

features2.append('pop_density')

for col in features2:
    
    if col not in ['east', 'north_city', 'east_city', 'habitants', 'qkm', *people]:
        df_habitant.loc[df_habitant[col] < 0, col] = 0.0001
        df_habitant.loc[df_habitant[col] == 0, col] = 0.0001
        df_habitant[col + '_log'] = np.log(df_habitant[col])
        germany_df3[col + '_log'] = np.log(germany_df3[col])
        
        features3.append(col + '_log')
        
        #df_habitant[col + '_p1'] = np.power(df_habitant[col],1)
        #features3.append(col + '_p1')
        #df_habitant[col + '_p2'] = np.power(df_habitant[col],2)
        #features3.append(col + '_p2')
        #df_habitant[col + '_p3'] = np.power(df_habitant[col],3)
        #features3.append(col + '_p3')  

features2 = sorted(list(set([*features3, 'east', 'north_city', 'east_city', 'einwohner', 'qkm/plz', ])))
print(features2)
print(len(features2))


print(germany_df3.columns)
print(df_habitant.shape)
df_train, df_test, y_train, y_test = train_test_split(df_habitant, df_habitant['log_price'], test_size=0.33, random_state=42)
print(df_train.shape)
df_train.head()

#df_train = fill_missing(df_train) # average
#df_filled[features] = np.round(imp.transform(df[features].values))

X = df_train.copy()
y = df_train['log_price']

from sklearn.impute import SimpleImputer # fill missing values
imp_mean = SimpleImputer(strategy='mean')

imp_mean.fit(X[features2])

imputed_train_df = X.copy()
X[features2] = imp_mean.transform(X[features2])

for name in features2:
    if X[name].isna().values.any():
        print("there is na", name)
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = StandardScaler()

features2 = sorted(list(set(features2)))
print(features2)
                 
for name in features2:
    if X[name].isna().values.any():
        print("there is na", name)
X[features2] = scaler.fit_transform(X[features2])  # scaling of the features

X = sm.add_constant(X[features2])


print("there is nan", y.isna().values.any())

X.head()
reg = sm.OLS(y, X).fit() 
results = reg
reg.summary()
params = pd.Series(reg.params)
print(params.sort_values().head(10))
print(params.sort_values(ascending=False).head(10))
influence = reg.get_influence()
resid_student = influence.resid_studentized
(cooks, p) = influence.cooks_distance
(dffits, p) = influence.dffits
leverage = influence.hat_matrix_diag

print ('\n')
print ('Leverage v.s. Studentized Residuals')
sns.regplot(leverage, reg.resid_pearson,  fit_reg=False)
print(resid_student.size)
res = pd.concat([pd.Series(cooks, name = "cooks"), pd.Series(dffits, name = "dffits"), pd.Series(leverage, name = "leverage"), pd.Series(resid_student, name = "resid_student")], axis = 1)
res_orig = res.copy()
res_copy = res.copy()
X_copy = X.copy()
y_copy = y.copy()
params = pd.Series(reg.params)
print(params.sort_values().head(12))
print(params.sort_values(ascending=False).head(12))
r_sort = res.sort_values(by = 'resid_student')
print ('-'*30 + ' top 5 most negative residuals ' + '-'*30)
print (r_sort.head())
print ('\n')

print ('-'*30 + ' top 5 most positive residuals ' + '-'*30)
print (r_sort.tail())
print(res.shape, df_train.shape)
df_res = pd.concat([res, df_train], axis=1)
df_res.resid_student = np.abs(df_res.resid_student)
avg_res = df_res.groupby(['plz']).mean().reset_index(); 
germany_dfx = pd.merge(
    left=z, 
    right=avg_res, 
    on='plz',
    how='left'
)

germany_dfx = germany_dfx.sort_values(by='resid_student', ascending=False)
plot_country(germany_dfx, 'resid_student')
print(germany_dfx.columns)
germany_dfx = germany_dfx.drop_duplicates(subset = ["plz"])
germany_dfx[['ort', 'plz', 'resid_student', 'bundesland']].head(20)
limit = 2 #resid student

#X = X[features2]
#y = X['log_price']

for name in features2:
    if X[name].isna().values.any():
        print("there is na", name)

y1 = y[np.asarray(res['resid_student']) < limit]
X1 = X[np.asarray(res['resid_student']) < limit]

res1 = res[res['resid_student'] < limit]

y1 = y1[np.asarray(res1['resid_student']) > -limit]
X1 = X1[np.asarray(res1['resid_student']) > -limit]

res1 = res1[res1['resid_student'] > -limit]



limit_lev = 0.05

y1 = y1[np.asarray(res1['leverage']) < limit_lev ]
X1 = X1[np.asarray(res1['leverage']) < limit_lev ]


y1 = y1[X1['new_apartment_permit_6_hab_log'] > -5]
X1 = X1[X1['new_apartment_permit_6_hab_log'] > -5]


X1 = X1[y1.values > 12]
y1 = y1[y1.values > 12]

res1 = res1[res1['leverage'] < limit_lev ]

sns.regplot(res1['leverage'], res1['resid_student'],  fit_reg=False)



from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate, KFold
from itertools import combinations
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
# Data preparation, as we have done before.
label_column = 'log_price'



# We use the linear model for our example.
kfold = KFold(n_splits = 5, shuffle = True, random_state = 0)
model = make_pipeline(
    StandardScaler(),
    LinearRegression())

# Gets the MSE for a model with zero features, just predicting the mean.
def get_base_mse(y):
    predictions = [np.mean(y)] * len(y)
    return mean_squared_error(predictions, y)

# Gets the MSE of our model, for a given dataset, estimated with 5-fold cross-validation.
def k_fold_mse(X, y):
    scores = cross_validate(model, X, y, scoring = 'neg_mean_squared_error', cv = kfold)
    result = np.mean(scores['test_score']) * -1
    return result

def forward_stepwise_selection(X, y, d):
    # Start with the MSE of the zero-features model.
    feature_columns = d.columns
    print(feature_columns)
    current_mse = get_base_mse(y)
    
    # This array contains the indices of the columns (features)
    # currently giving the best ecountered model. At the beginning,
    # this is an empty numpy array.
    current_features = np.array([], dtype = int)
    
    # This array contains the indices of all the columns (features)
    # in our dataset. In other words, it is a numpy array containing
    # [0, 1, ..., p-1] where p is the number of features.
    all_features = np.arange(len(feature_columns))
    
    # In the extreme case, when adding a feature always improves the
    # model, we terminate when we added all features.
    # In that case, current_features == all_features.
    while not len(current_features) > 15: #np.array_equal(current_features, all_features):
        # This variable will contain the index of the *new* feature
        # we want to add to the model. If no improving feature is found,
        # then this variable will keep value None.
        selected_feature = None
        
        # For features not yet in the model...
        for feature in (set(all_features) - set(current_features)):
            # Build a new set of features, adding the new one to the
            # ones already in the model.
            new_features = np.append(current_features, feature)
            
            # Estimate the mse of the new model.
            mse = k_fold_mse(X[:,new_features], y)
            
            # If it's better than the current best, update the best
            # current MSE and mark this feature as the selected new
            # feature.
            if mse < current_mse:
                current_mse = mse
                selected_feature = feature
                
                
                
        # If we found an improving feature...
        if selected_feature is not None:
            #... add it to the current features.
            current_features = np.append(current_features, selected_feature)

        else:
            # Otherwise, terminate.
            break
            
            print("current_features")
            for idx in sorted(current_features):
                print(f"\t{d.columns[idx]}")
    
    return current_features, current_mse
features_idx, mse = forward_stepwise_selection(X1[features2].values, y1.values, X1[features2])
print(f"MSE of the selected model: {mse:.3f}")
print(features_idx)
print(X1.columns[features_idx])
selected_features = X1.columns[features_idx]

X1[selected_features] = scaler.fit_transform(X1[selected_features])

reg = sm.OLS(y1, sm.add_constant(X1[selected_features])).fit()
print("there is nan", y1.isna().values.any())
print("there is null", y1.isnull().values.any())

results = reg
reg.summary()
from statsmodels.compat import lzip
name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(reg.resid)
lzip(name, test)
#sns.kdeplot(np.array(reg.resid), bw=10)
sns.distplot(np.array(reg.resid), hist=True)
import scipy.stats as scipystats
import pylab
scipystats.probplot(reg.resid, dist="norm", plot=pylab)
pylab.show()
# https://songhuiming.github.io/pages/2016/12/31/linear-regression-in-python-chapter-2/

features_idx, mse = forward_stepwise_selection(X1.values, y1.values, X1)
print(f"MSE of the selected model: {mse:.3f}")

print(features_idx)
print(X1.columns[features_idx])
selected_features = X.columns[features_idx]


reg = sm.OLS(y1, sm.add_constant(X1[selected_features])).fit()
results = reg
reg.summary()
params = pd.Series(reg.params)
print(params.sort_values().head(15))
print(params.sort_values(ascending=False).head(15))
X1[selected_features].describe()
np.linalg.cond(reg.model.exog) # Condition number
#sns.pairplot( pd.concat([y1, X1[selected_features]], axis=1), kind="reg", corner=True)
corrMatrix = pd.concat([y1, X1[selected_features]], axis=1).corr()
plt.figure(figsize=(15,10))
#print(corrMatrix.sort_values('log_price', ascending=True)['log_price'].head(30))
#print(corrMatrix.sort_values('log_price', ascending=False)['log_price'].head(30))
sns.heatmap(corrMatrix, annot=True)
from statsmodels.stats.outliers_influence import variance_inflation_factor
selected_features = list(selected_features)
#selected_features.remove('tax_payers_hab_log')
X_train = np.asarray(X1[selected_features])
vif = pd.DataFrame()
vif["features"] = X1[selected_features].columns
vif["VIF Factor"] = [variance_inflation_factor(X_train, i) for i in range(X_train.shape[1])]

#first use variables realted to pop
vif.sort_values(["VIF Factor"], ascending=False).head(30)
name = ['Lagrange multiplier statistic', 'p-value',
        'f-value', 'f p-value']
test = sms.het_breuschpagan(reg.resid, reg.model.exog)
lzip(name, test)
resid = reg.resid
plt.scatter(reg.predict(), resid ** 2)
# plot of residuals in ascending order of price 

from statsmodels.sandbox.regression.predstd import wls_prediction_std
nsample = 50
sig = 0.5
x = np.linspace(0, 20, X1.shape[0])
beta = [0.5, 0.5, -0.02, 5.]

y_true = y1.copy()

ind = np.argsort(y_true, axis=0)

prstd, iv_l, iv_u = wls_prediction_std(reg)


fig, ax = plt.subplots(figsize=(8,6))

#ax.plot(x, y, 'o', label="data")
ax.plot(x, reg.resid.iloc[ind], 'r.', label="Residuals")
ax.set_ylim([-3, 3])
#ax.plot(x, y_true.iloc[ind], 'b-', label="True")
ax = sns.regplot(x=x, y=reg.resid.iloc[ind], scatter=False)


selected_features_no_log = [w.replace('_log', '') for w in selected_features]
selected_features_no_log

#df_train.dropna(inplace=True, subset='log_price')
X_no_log = df_train[selected_features_no_log].copy()
y_log = df_train['log_price']
X_no_log = X_no_log[y_log.notna()]
y_log = y_log[y_log.notna()]
print(X_no_log[selected_features_no_log].shape)
imp_mean = SimpleImputer(strategy='mean')
imp_mean.fit(X_no_log[selected_features_no_log])
X_no_log[selected_features_no_log] = imp_mean.transform(X_no_log[selected_features_no_log])
X_no_log[selected_features_no_log] = scaler.fit_transform(X_no_log[selected_features_no_log])
X_no_log.head()
reg_no_log = sm.OLS(y_log, sm.add_constant(X_no_log[selected_features_no_log])).fit()



nsample = 50
sig = 0.5
x = np.linspace(0, 20, X_no_log.shape[0])
beta = [0.5, 0.5, -0.02, 5.]

y_true = y_log.copy()
ind = np.argsort(y_true, axis=0)
print(ind.shape)
print(y_true.shape)
print(X_no_log.shape)
print(x.shape)
reg_no_log.summary()

#sns.pairplot( pd.concat([y1, X1[selected_features]], axis=1), kind="reg", corner=True)

fig, ax = plt.subplots(figsize=(8,6))
ax.plot(x, reg_no_log.resid.iloc[ind], 'r.', label="Residuals")
ax.set_ylim([-3, 3])
ax = sns.regplot(x=x, y=reg_no_log.resid.iloc[ind], scatter=False)
# plot residuals in a model where log transformation is not applied

plt.scatter(reg_no_log.predict(), reg_no_log.resid ** 2)
names = selected_features
from pylab import *
i = 1
j = 1
fig = plt.figure(figsize=(30, 20))

X_plot = X1.copy()
y_plot = y1.copy()
print(X_plot.shape)
print(y_plot.shape)
for name in names[:]:

    reg1 = sm.OLS(y_plot, sm.add_constant(X_plot[name])).fit()
    ax1 = fig.add_subplot( 4 ,8,i)
    ax1.scatter(X_plot[name], y_plot)
    ax1.plot(X_plot[name], reg1.params[0] + reg1.params[1] * X_plot[name], '-', color='r')
    #ax1.ylabel('Log of Price')
    #ax1.xlabel(name)
    ax1.set_xlabel(name)
    ax1.set_ylabel('Log of Price')
    ax1.set_xlim([-7, 7])
    ax1.set_ylim([10, 15])
    i += 1
    
plt.show()
#results = reg
#reg.summary()
from statsmodels.sandbox.regression.predstd import wls_prediction_std
nsample = 50
sig = 0.5
x = np.linspace(0, 20, X1.shape[0])
beta = [0.5, 0.5, -0.02, 5.]

y_true = y1.copy()

ind = np.argsort(y_true, axis=0)

prstd, iv_l, iv_u = wls_prediction_std(reg)


fig, ax = plt.subplots(figsize=(8,6))

#ax.plot(x, y, 'o', label="data")
ax.plot(x, reg.predict(sm.add_constant(X1[selected_features].iloc[ind])), 'r--.', label="OLS")
ax.plot(x, y_true.iloc[ind], 'b-', label="True")

#ax.plot(x, iv_u, 'r--')
#ax.plot(x, iv_l, 'r--')
ax.legend(loc='best');
from statsmodels.sandbox.regression.predstd import wls_prediction_std
nsample = 50
sig = 0.5


X2 = df_test[selected_features]

imp_mean = SimpleImputer(strategy='mean')
imp_mean.fit(X2[selected_features])
X2[selected_features] = imp_mean.transform(X2[selected_features])

X2[selected_features] = scaler.fit_transform(X2[selected_features])

y_true = df_test['log_price'].copy().astype(float)

X2 = X2[y_true.notna()]
y_true = y_true[y_true.notna()]

X2 = X2[y_true.values > 12]
y_true = y_true[y_true.values > 12]
y_pred = reg.predict(sm.add_constant(X2[selected_features]))
x = np.linspace(0, 20, X2.shape[0])
beta = [0.5, 0.5, -0.02, 5.]

ind = np.argsort(y_true, axis=0)
prstd, iv_l, iv_u = wls_prediction_std(reg)
fig, ax = plt.subplots(figsize=(8,6))
#ax.plot(x, y, 'o', label="data")
ax.plot(x, y_pred.iloc[ind], 'r--.', label="OLS")
ax.plot(x, y_true.iloc[ind], 'b-', label="True")

#ax.plot(x, iv_u, 'r--')
#ax.plot(x, iv_l, 'r--')
ax.legend(loc='best');
from statsmodels.tools import eval_measures
print("MSE", eval_measures.mse(y_true, y_pred))
print("RMSE", eval_measures.rmse(y_true, y_pred))
print(df_train[selected_features].shape)
print(res.shape)
mask1 = res.copy()
mask1['resid_student'] = np.abs(mask1['resid_student'])
print(mask1.shape)
df_train2 = df_train[ np.asarray(mask1['resid_student'] < limit)]
mask1 = mask1[mask1['resid_student'] < limit]
df_train2 = df_train2[ np.asarray(mask1['leverage'] < limit_lev )]
print("remove resid student outlatyer", df_train2.shape)
print("df_text_shape", df_test.shape)
df_test.head()
df_both = pd.concat([df_train, df_test])
print("df_both", df_both.shape)
df_both = df_both[df_both['log_price'].notna()]
print("df_both", df_both.shape)
imp_mean = SimpleImputer(strategy='mean')
X_final = df_both[selected_features].copy()

imp_mean.fit(X_final[selected_features])

X_final[selected_features] = imp_mean.transform(X_final[selected_features])
X_final[selected_features] = imp_mean.transform(X_final[selected_features])

scaler = StandardScaler()

X_final[selected_features] = scaler.fit_transform(X_final[selected_features])
print(X_final.shape)
X_final.head()
reg_final = sm.OLS(df_both['log_price'], sm.add_constant(X_final[selected_features[:]])).fit()
reg_final.summary()
sns.pairplot( pd.concat([df_both['log_price'], X_final[selected_features[11:16]]], axis=1), kind="reg", corner=True)


results = pd.DataFrame(reg_final.summary().tables[1])
results = results.rename(columns=results.iloc[0])
results = results.drop(results.index[0])
results.head(20)
print(germany_df3.shape)
germany_df3 = germany_df3.replace([np.inf, -np.inf], np.nan)

for col in germany_df3[selected_features].columns:
    print(col, ' ', germany_df3[col].isna().values.sum())
    print(col, ' ', germany_df3[col].notnull().values.sum())
    print(col, 'fini ', np.isfinite(germany_df3[col]).all())
    

germany_dfinal = germany_df3[selected_features].copy()
print(germany_dfinal.shape)

germany_dfinal = germany_dfinal.dropna(subset=[*selected_features], how="all")
germany_dfinal = germany_dfinal[np.isfinite(germany_dfinal)]
print(germany_dfinal.shape)
scaler = StandardScaler()

germany_dfinal[selected_features] = scaler.fit_transform(germany_dfinal[selected_features])

    
prediction = reg_final.predict(sm.add_constant((germany_dfinal[selected_features])))

print(prediction.shape)

germany_df3['pred'] = prediction
germany_dfinal[selected_features].describe()
plot_country(germany_df3, 'pred')
plot_city(germany_df3, "Aachen", 'pred')
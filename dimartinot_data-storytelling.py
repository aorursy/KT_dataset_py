!pip install reverse_geocoder
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from tqdm import tqdm

import math

import geopandas as gpd

import folium
ds = pd.read_csv("../input/meteorite-landings/meteorite-landings.csv")
len(ds['id'].unique())
ds.head()
ds_name = ds["name"]
print("There are {} unique names out of {} total with {} NaN values".format(len(ds_name.unique()), len(ds_name), len(ds_name[ds_name.isna()])))
ds_name_with_numeric = ds_name[ds_name.str.contains("[0-9]", regex=True)]

print("There exist {} meteorites with numeric names. It represents {}% of our dataset.\nWe can display the following ones as example:".format(len(ds_name_with_numeric), 100*round(len(ds_name_with_numeric)/len(ds_name), 2)))

ds_name_with_numeric.head()
def remove_non_alpha(string):

    return "".join(char for char in string if char.isalpha())



ds_name_fixed = ds_name.apply(remove_non_alpha)

#To make sure we don't have problem with casing, we put all strings to lower

ds_name_fixed = ds_name_fixed.str.lower()
print("Once deprived from non-alphabetical characteres, there are {} unique names out of {} total".format(len(ds_name_fixed.unique()), len(ds_name_fixed)))
# Setting the size of 

plt.rcParams['figure.figsize'] = [15, 10]



twenty_most_common_names = ds_name_fixed.value_counts()[:20]

common_names_barplot = sns.barplot(twenty_most_common_names.keys(), twenty_most_common_names);

common_names_barplot.set_xticklabels(common_names_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')

common_names_barplot.set_title("Barplot of the 20 most common meteorite names (after processing)")

common_names_barplot;
ds_geo = ds["GeoLocation"]
print("There are {} unique location out of {} total as well as {} NaN values".format(len(ds_geo.unique()), len(ds_geo), len(ds_geo[ds_geo.isna()])))
twenty_most_common_locations = ds_geo.value_counts()[:19]

#Adding the NaN counts

twenty_most_common_locations = twenty_most_common_locations.append(pd.Series(len(ds_geo[ds_geo.isna()]), index=["No value"]))

twenty_most_common_locations = twenty_most_common_locations.sort_values(ascending=False)



common_location_barplot = sns.barplot(twenty_most_common_locations.keys(), twenty_most_common_locations);

common_location_barplot.set_xticklabels(common_location_barplot.get_xticklabels(), rotation=45, horizontalalignment='right')

common_location_barplot.set_title("Barplot of the 20 most common location of meteorites")

common_location_barplot;
meteorite_landing =[-71.5, 35.666670]



m = folium.Map(

    meteorite_landing, 

    zoom_start=5,

    tiles="Stamen Terrain"

)



folium.Marker(location=meteorite_landing).add_to(m)



m
gdf = gpd.GeoDataFrame(

    ds, geometry=gpd.points_from_xy(ds.reclong, ds.reclat))
import folium

from folium.plugins import Draw
m2 = folium.Map(

    #layers=(basemap_to_tiles(basemaps.NASAGIBS.ModisTerraTrueColorCR, "2017-04-08"), ),

    zoom_start=2, 

    no_touch=True)



for key in twenty_most_common_locations.keys():

    try:

        pos = eval(key)

        #print("Position: {}".format(pos))

        folium.Marker(pos).add_to(m2)

    except:

        print("Invalid position")



m2
ds_nametype = ds["nametype"]

ds_nametype_val_counts = ds_nametype.value_counts()

print("{} of 'Valid' values out of {} total".format(ds_nametype_val_counts["Valid"], len(ds_nametype)))

nametype_barplot = sns.barplot(ds_nametype_val_counts.keys(), ds_nametype_val_counts)

nametype_barplot.set_title("Barlot of the two meteorite nametypes")

nametype_barplot;
ds_recclass = ds['recclass']

ds_recclass.head()
main_meteorite_classes = {

    "stony": ["CI", "CM", "CO", "CV", "CK", "CR", "CH", "CB", "H", "L", "LL", "E", "OC", "LLL", "HL", "EH", "EL", "R", "K", "ACAPULCOITE", "LODRANITE", "WINONAITE", "HOWARDITE", "EUCRITE", "DIOGENITE", "ANGRITE", "AUBRITE", "UREILITE", "BRACHINITE", "LUNAR", "SHERGOTTITE", "NAKHLITE", "CHASSIGNITE", "MARTIAN", "ACHONDRITE", "CHONDRITE", "RELICTOC"],

    "stony_iron": ["PALLASITE", "MESOSIDERITE"],

    "iron": ["IC", "IIAB", "IIC", "IID", "IIF", "IIG", "IIIAB", "IIIE", "IIIF", "IVA", "IVB", "IAB", "UDEI", "PITTS", "sLL", "sLM", "sLH", "sHL", "sHH", "IIE"]

}



def get_category(recclass):

    #to make sure the matching works correctly, we remove any non alpha character

    corrected_recclass = "".join(char for char in recclass if char.isalpha())

    

    #some recclass also contains Iron in their name, if it is so, we directly categorise them.

    if "iron" in corrected_recclass.lower():

        return "iron"

    

    #we loop over the meteorite classes to know if, by any chance, the meteorite is indeed in this class

    for key in main_meteorite_classes.keys():

        

        # we check if the exact recclass of the meteorite is in one of the three arrays. If yes, e return the according key (stony, stony_iron or iron)

        if (corrected_recclass.upper() in main_meteorite_classes[key]):

            return key

        

        # some names are composed of the class and the main category or other strings. Therefore, we loop other each key's array and 

        #check if any name of size more than 3 is inside the recclass of the meteorite (ex: "Pallasite" is in "PallasitePMG" )

        for name in main_meteorite_classes[key]:

            if (len(name) >= 3 and name in corrected_recclass.upper()):

                return key

    

    #print("Unclassified: {}".format(corrected_recclass))

    return "unclassified"
ds_recclass_as_df = ds_recclass.to_frame()

ds_recclass_as_df["main_category"] = ds_recclass_as_df["recclass"].apply(get_category)

ds_main_category = ds_recclass_as_df["main_category"]
counted_categories = ds_main_category.value_counts()



main_category_barplot = sns.barplot(counted_categories.keys(), counted_categories)

main_category_barplot.set_xticklabels(main_category_barplot.get_xticklabels())

main_category_barplot.set_title("Barplot of the 3 main recclass types (+ unclassified recclass)")

main_category_barplot;
ds_unclassified_recclass = ds_recclass_as_df[ds_recclass_as_df["main_category"] == "unclassified"]



twenty_most_common_unclassified_recclass = ds_unclassified_recclass["recclass"].value_counts()[:20]



twenty_most_common_unclassified_recclass_as_df = twenty_most_common_unclassified_recclass.to_frame()

twenty_most_common_unclassified_recclass_as_df["recclass_count"] = twenty_most_common_unclassified_recclass_as_df["recclass"]

twenty_most_common_unclassified_recclass_as_df["recclass"] = twenty_most_common_unclassified_recclass_as_df.index
f, ax = plt.subplots(figsize=(15, 15))



sns.set_color_codes("pastel")

common_unclassified_recclass_barplot = sns.barplot(

    x="recclass_count", y="recclass", label="Recorded Class", data=twenty_most_common_unclassified_recclass_as_df, color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)



common_unclassified_recclass_barplot.set_title("Horizontal barplot of the most common unclassified recclass")

common_unclassified_recclass_barplot;
ds_mass = ds['mass']

print("{} out of {} total values are NaN".format(len(ds_mass[ds_mass.isnull()]), len(ds_mass)))
ds_mass_corrected = ds_mass.dropna()

print("Series still contains NaN values? {} (should be False)".format(ds_mass_corrected.isnull().values.any()))
ds_mass_corrected.describe()
median = ds_mass_corrected.median()



ds_mass_corrected_under = ds_mass_corrected[ds_mass_corrected < median]

ds_mass_corrected_over = ds_mass_corrected[ds_mass_corrected >= median]



fig, (ax_under, ax_over) = plt.subplots(1, 2)



sns.distplot(ds_mass_corrected_under, ax=ax_under).set_title("Histogram of the mass of the 50% less heavy meteorites")

sns.distplot(ds_mass_corrected_over, ax=ax_over).set_title("Histogram of the mass of the 50% heaviest meteorites")

fig;
from scipy import stats



stats.zscore(ds_mass_corrected_over)

d = ds_mass_corrected_over[(np.abs(stats.zscore(ds_mass_corrected_over)) < 0.03)]

mass_corrected_plot = sns.distplot(d)

mass_corrected_plot.set_title("Histogram of the corrected mass distribution (without outliers)")

mass_corrected_plot;
ds_fall = ds["fall"]



fell_found_count = ds_fall.value_counts()



fell_found_barplot = sns.barplot(fell_found_count.keys(), fell_found_count)



print("There are {} observed falls for {} total. It is roughly 1 meteorite observed for every {} found."

      .format(fell_found_count["Fell"], len(ds), round(len(ds)/fell_found_count["Fell"]))

     )



fell_found_barplot.set_title("Barplot of the fell/found distribution of meteorites")

fell_found_barplot;
ds_year = ds["year"]

print("{} out of {} total values are NaN".format(len(ds_year[ds_year.isnull()]), len(ds_year)))
ds_year_corrected = ds_year.dropna()

print("Min year: {}\nMax year: {}\nMed year: {}".format(ds_year_corrected.min(), ds_year_corrected.max(), ds_year_corrected.median()))
hist_year = sns.distplot(ds_year_corrected)

hist_year.set_title("Histogram of the year distribution of the found meteorites")

hist_year;
future_discoveries = ds_year_corrected[ds_year_corrected > 2019]

future_discoveries
future_discoveries_rows = ds[ds_year.notnull().any() and ds_year > 2019]

future_discoveries_rows
mystery_meteorite_landing = [30.900000, 46.016670]



m = folium.Map(

    location = mystery_meteorite_landing,

    zoom_start=12,

    tiles='Stamen Terrain'

)





folium.Marker(mystery_meteorite_landing, popup='<i>"Ur"</i> meteorite').add_to(m)



m
# set the path to the world shapefile and load in a shapefile

path = "../input/shapefiles-geodata/countries.shp"



world_map = gpd.read_file(path)

# check data type so we can see that this is not a normal dataframe, but a GEOdataframe

world_map.head()
world_map.plot();
import reverse_geocoder as rg
first_meteorite = ds.iloc[0]

(lat, lon) = (first_meteorite[7], first_meteorite[8])
results = rg.search((lat, lon)) 



print(results[0])
def retrieve_country_code(lat, lon):

    #Some coordinates are set to 0 due to misisng data, we will then insert None instead

    if (lat != 0 and lon != 0):

        #print(r'Treating ({},{})'.format(lat, lon))

        lat_str, lon_str = str(lat), str(lon)

        coord = lat_str+", "+lon_str

        

        coordinates = (lat, lon)

        try:

            results = rg.search(coordinates)

            location = results[0]['cc']

        except IndexError:

            return None

        

        if location == None:

            return None

        else:

            return location

    else:

        #print(r'Treating malicious coordinates')

        return None
iso_col = []



for row in tqdm(ds.itertuples(index = True)):

    iso_col.append(retrieve_country_code(getattr(row,'reclat'), getattr(row,'reclong')))



ds['ISO2'] = pd.Series(iso_col)
ds.head()
def count_iso_fell(iso_to_count):

    return len(ds[(ds["ISO2"] == iso_to_count) & (ds["fall"] == "Fell")])



def count_iso_found(iso_to_count):

    return len(ds[(ds["ISO2"] == iso_to_count) & (ds["fall"] == "Found")])
world_map['fell'] = world_map['ISO2'].apply(count_iso_fell)

world_map['found'] = world_map['ISO2'].apply(count_iso_found)
fig, ax = plt.subplots(1, 1)

ax.set_title("Colormaps of the 'fell' category of meteorites")

world_map.plot(column='fell', ax=ax, legend=True, cmap='OrRd');

prebuilt_world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

prebuilt_world.head()
# we initialize the population density column to make sure that all country have at least a value (defaulting to 0)

world_map["pop_dens"] = np.zeros(world_map.shape[0]).astype(np.float16)
# to help with merging, we rename the iso column and create a copy of the original dataset, with the renamed column.

copy_prebuilt = prebuilt_world.rename(columns={"iso_a3":"ISO3"})
#this holds the result of the merge

result = pd.merge(world_map, copy_prebuilt[["ISO3","pop_est"]], how='outer', on=['ISO3'])
for iso in result["ISO3"].unique():

    samples = result[result["ISO3"] == iso]

    if (len(samples) > 1):

        total_sqkm = samples["SQKM"].sum()

        result.loc[result["ISO3"] == iso, "SQKM"] = total_sqkm
# implements the formula to calculate a population density and pply it

def get_dens(country_pop, size):

    if (np.isnan(country_pop) == False and np.isnan(size) == False and country_pop > 0 and size > 0):

        return math.ceil(country_pop/size)

    else:

        return 0

    

result["pop_dens"] = result.apply(lambda row: get_dens(row["pop_est"],row["SQKM"]), axis=1)
def divide_fallen_by_popdens(fallen, pop_dens):

    if (np.isnan(pop_dens) == False and np.isnan(fallen) == False and pop_dens != 0):

        return fallen/pop_dens

    else:

        return 0



result["fallen_by_pop_dens"] = result.apply(lambda row: divide_fallen_by_popdens(row["fell"], row["pop_dens"]), axis=1)
f, (ax1, ax2) = plt.subplots(1, 2)

ax1.set_title("Colormaps of the number of observed falls")

ax2.set_title("Colormaps of population density")

world_map.plot(column='fell', ax=ax1, cmap='OrRd');

result.plot(column='pop_dens', ax=ax2, cmap='OrRd');
fig_b, ax_b = plt.subplots(1, 1)

ax_b.set_title("Colormaps of the 'found' category of meteorites")

world_map.plot(column='found', ax=ax_b, legend=True, cmap="OrRd");
ds_relict = ds[ds["nametype"] == "Relict"]
def count_iso_relict(iso_to_count):

    return len(ds_relict[(ds_relict["ISO2"] == iso_to_count)])



world_map["relict_count"] = world_map["ISO2"].apply(count_iso_relict)
world_map.plot(column="relict_count", legend=True, cmap="OrRd")
non_null_relict = world_map[world_map["relict_count"] > 0]
f, ax = plt.subplots()



sns.set_color_codes("pastel")

non_null_relict_barplot = sns.barplot(

    x="relict_count", y="COUNTRY", label="Relict count", data=non_null_relict, color="b")



# Add a legend and informative axis label

ax.legend(ncol=2, loc="lower right", frameon=True)



non_null_relict_barplot.set_title("Horizontal barplot of the countries with relict meteorites")

non_null_relict_barplot;
sweden_relict = ds_relict[ds_relict["ISO2"]=="SE"]



m3 = folium.Map(

    #layers=(basemap_to_tiles(basemaps.NASAGIBS.ModisTerraTrueColorCR, "2017-04-08"), ),

    location=[62.6431513,16.2680932],

    zoom_start=4

)



for pos_as_str in sweden_relict["GeoLocation"].unique():

    try:

        pos = eval(pos_as_str)

        #print(r"Position: {}".format(pos))

        folium.Marker(list(pos), popup=pos_as_str).add_to(m3)

    except:

        print("Invalid position")

        

print("All meteorite unique positions: {}".format(sweden_relict["GeoLocation"].unique()))

m3
print("Does our dataset contain such meteorite ? {}".format("Österplana 065" in ds["name"]))
ds_uncl = ds[ds["recclass"] == "Stone-uncl"]
m4 = folium.Map(

    zoom_start=1, 

    tiles='Stamen Terrain')



for pos_as_str in ds_uncl["GeoLocation"].unique():

    try:

        pos = eval(pos_as_str)

        #print(r"Position: {}".format(pos))

        folium.Marker(list(pos), title=pos_as_str).add_to(m4)

    except:

        print("Invalid position")

        

m4
sns.scatterplot(x="year", y="mass", data=ds_uncl, hue="fall")
oldest_stone = ds_uncl[ds_uncl["year"] < 1500]

oldest_stone
oldest_stone = oldest_stone[oldest_stone["name"] == "Rivolta de Bassi"]



pos = eval(oldest_stone["GeoLocation"].values[0])



m_old = folium.Map(

    #layers=(basemap_to_tiles(basemaps.NASAGIBS.ModisTerraTrueColorCR, "2017-04-08"), ),

    location=list(pos),

    zoom_start=10

)



folium.Marker(list(pos), popup=str(pos)).add_to(m_old)



m_old
ds_uncl_it = ds_uncl[ds_uncl["ISO2"] == "IT"]



m_it = folium.Map(

    #layers=(basemap_to_tiles(basemaps.NASAGIBS.ModisTerraTrueColorCR, "2017-04-08"), ),

    location=list(eval(oldest_stone["GeoLocation"].values[0])),

    zoom_start=5, 

    tiles="Stamen Terrain")



for pos_as_str in ds_uncl_it["GeoLocation"].unique():

    try:

        pos = eval(pos_as_str)

        #print(r"Position: {}".format(pos))

        folium.Marker(list(pos), popup=pos_as_str).add_to(m_it)

    except:

        print("Invalid position")

        

m_it
heaviest_stone_uncl = ds_uncl[ds_uncl["mass"]>170000]

heaviest_stone_uncl
heaviest_stone_uncl_pos = heaviest_stone_uncl["GeoLocation"].values[0]



m_mn = folium.Map(

    location=list(eval(heaviest_stone_uncl_pos)),

    zoom_start=9,

    tiles='Stamen Terrain'

)



folium.Marker(list(eval(heaviest_stone_uncl_pos)), popup=heaviest_stone_uncl_pos).add_to(m_mn)



m_mn
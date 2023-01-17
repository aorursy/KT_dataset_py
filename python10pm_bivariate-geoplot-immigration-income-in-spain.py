!pip install spectra
import numpy as np

import pandas as pd

import geopandas as gpd



import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.colors import rgb2hex

import matplotlib.patheffects as path_effects

%matplotlib inline



import mapclassify

import spectra # color calculation and manipulation

import shapely



import os

for dirname, _, filenames in os.walk('/kaggle/input/'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import warnings

warnings.filterwarnings("ignore")
# MAIN GLOBAL VARIABLES FOR THIS KERNEL



# path to our data that we will use in this kernel

PATH_INCOME = "../input/average-income-per-city/renta_media_municipios_espaa_2017.xlsx"

PATH_IMMIGRATION_BCN = "../input/inmigration-per-city/00008002bd.csv"

PATH_IMMIGRATION_MAD = "../input/inmigration-per-city/00028002bd.csv"

PATH_GEOJSON = "../input/spanish-cities-geojson/espana-municipios.geojson"



# simplify the topology of the regions from the geojson file

# this will help render faster the folium map

SIMPLIFY = 0



# how many colors do you wan to generate

NR_COLORS = 4



# filter the province you want to visualize

PROVINCE = "Barcelona"
# function to extract the zip code

def extract_zip_code(city, position):

    '''

    This function takes 2 mandatory arguments:

        --> 1. city: a string that must contain the zip code and separated by a semicolon (-)

        --> 2. position: the position to extract when you apply str.split("-")

    '''

    if "-" in city:

        zip_code = city.split("-")[position]

        return zip_code

    else:

        return '00000'

    

def generate_bins(series, NR_COLORS):

    

    '''

    Binazires the values you pass into the nr of categories you choose.

    This will help map values to colors.

    '''

    bins_ = mapclassify.FisherJenks(series, k = NR_COLORS)

    

    return bins_



def plot_bivariate_map(gdf_, migrant_rate_bin, income_rate_bin, bivariate_palette, PROVINCE, NR_COLORS):

    

    '''

    Wrap up function to make bivariate plots.

    '''

    # get the bounds of our dataframe

    bounds = gdf_.total_bounds

    aspect = (bounds[2] - bounds[0]) / (bounds[3] - bounds[1])

    box = shapely.geometry.box(*bounds)



    # prepare the figure to plot on

    fig, ax = plt.subplots(1, 1, figsize = (20, 20 / aspect))

    ax.set_xlim(bounds[0], bounds[2])

    ax.set_ylim(bounds[1], bounds[3])



    # iterate over all posible combinations and add them to the map

    for i in range(NR_COLORS):

        for j in range(NR_COLORS):

            # get the area where the coditions are met

            # basically we start with the lowest bin possible both for immigration and income (0, 0)

            areas = set(np.where(migrant_rate_bin.yb == i)[0]) & set(np.where(income_rate_bin.yb == j)[0])

            # using our dict, get the specific color and plot it on the map

            gdf_.iloc[list(areas)].plot(color = bivariate_palette[(i, j)], ax = ax, edgecolor = 'none')



    # no additional info on the map

    ax.set_axis_off()



    # adding borders

    gdf_.boundary.plot(ax = ax, edgecolor = "black", alpha = 0.2)



    # add the city name

    for idx, row in gdf_.set_index('City_name').iterrows():

        # add the city name only for one third of the cities, otherwise the map will be too poluted

        random_number = np.random.random()

        if random_number < 0.3:

            centroid = row.geometry.intersection(box).representative_point()

            t = ax.text(centroid.x, centroid.y, idx, horizontalalignment = 'center', fontsize = 10, fontweight = 'bold', color = 'white')

            t.set_path_effects([path_effects.Stroke(linewidth = 2, foreground = 'black'), path_effects.Normal()])



    # add the legend

    a = fig.add_axes([0.95, .425, .15, .15], facecolor = 'y')

    a.imshow(cmap_xy, origin = 'lower')



    a.set_xlabel('Average Income (lower to higher)\n$\\rightarrow$', fontsize = 'medium')

    a.set_ylabel('% of Immigrants (lover to higher)\n$\\rightarrow$', fontsize = 'medium')



    a.set_xticks([])

    a.set_yticks([])

    a.set_title('Legend', loc = 'left')

    sns.despine(ax = a)



    # set a title 

    ax.set_title('Average Income (binnazired) and the % of immigrants in the cities of {} region)'.format(PROVINCE), loc = 'left', fontweight = 'bold', fontsize = 14);
# list with english columns

lc_income = [

    "City",

    "Entitles",

    "Nr_declarations",

    "Population",

    "Average_income_over_nation_level",

    "Average_income_over_ccaa_leval",

    "Average_income",

    "Average_net_income"

]



# import the df and change the columns name

df_income = pd.read_excel(PATH_INCOME)

df_income.columns = lc_income



# extract the zip code and pass an additional argument when using apply

df_income["Zip_code"] = df_income["City"].apply(extract_zip_code, args = [1])



print("We have a total of {} unique zip codes in our income dataframe.".format(len(df_income["Zip_code"].unique())))
# let's take a quick view of our dataframe after all the manipulations

df_income.head()
# list with english columns

lc_immigration = [

    "Gender",

    "City",

    "Local_alien",

    "Age",

    "Total"

]



# import the data for cities from Madrid and Barcelona province

df_mad = pd.read_csv(PATH_IMMIGRATION_MAD, encoding = 'latin-1', delimiter = "\t")

df_bcn = pd.read_csv(PATH_IMMIGRATION_BCN, encoding = 'latin-1', delimiter = "\t")



# change the columns name

df_mad.columns = lc_immigration

df_bcn.columns = lc_immigration



# create a new dataframe by appending the 2 datasets

df_immigration = df_mad.append(df_bcn)



# extract the zip code using the previous function and drop the age columns

df_immigration["Zip_code"] = df_immigration["City"].apply(extract_zip_code, args = [0])

df_immigration.drop("Age", inplace = True, axis = 1)



# convert the "Total" column to numeric

df_immigration["Total"] = df_immigration["Total"].apply(lambda x: x.replace(".", ""))

df_immigration["Total"] = df_immigration["Total"].astype(int)



# here we change the structure of our dataframe using a pivot table

df_immigration = df_immigration.pivot_table(values = "Total", index = ["City", "Zip_code", "Gender"], columns = ["Local_alien"]).reset_index()



# calculate the percentage of immigrants over total population

df_immigration["Immigrants_over_total"] = df_immigration["extranjeros"]/df_immigration["españoles"]



print("We have a total of {} unique zip codes in our immigration dataframe.".format(len(df_immigration["Zip_code"].unique())))
# let's take a quick view of our dataframe after all the manipulations

df_immigration.head()
# columns from the original geojson, in spanish, that we are interested in

columns_to_import_geojson = [

'pais',

'communidad_autonoma',

'provincia',

'municipio',

'codigo_postal',

'geometry'

]



# rename the columns to english

english_column_name_geojson = [

'Country',

'Autonomous_community',

'Province',

'City_name',

'Zip_code',

'geometry'

]



# import geojson using geopandas and rename the columns to english

gdf = gpd.read_file(PATH_GEOJSON)

gdf = gdf[columns_to_import_geojson]

gdf.columns = english_column_name_geojson



# eliminate all nulls from the zip_code column

gdf = gdf[gdf["Zip_code"].notnull()]



# simplify the topology of the regions

gdf["geometry"] = gdf["geometry"].simplify(SIMPLIFY, preserve_topology = True)



print("We have a total of {} unique cities in our geojson.".format(len(gdf["City_name"].unique())))
# this df is crucial, since it contains all the geometry/shape of our regions. This will allow us to plot them on a map.

gdf.head()
# let's find the intersection of 3 list, this way we will only leave the common cities that contain the same zip_code

zip_code_income = sorted(list(df_income["Zip_code"].unique()))

zip_code_immigration = sorted(list(df_immigration["Zip_code"].unique()))

zip_code_gdf = sorted(list(gdf["Zip_code"].unique()))



# intersection operation in python using set

# using this code, we can filter the common elements of 3 lists or sets.

l_zip = [zip_code_income, zip_code_immigration, zip_code_gdf]

common_zip_code = list(set(l_zip[0]).intersection(*l_zip))
print("We have a total of {} cities that are common in the income, immigration and geojson dataframes.".format(len(common_zip_code)))
# select only unique entries for the df_immigration for the common cities

df_immigration = df_immigration[(df_immigration["Zip_code"].isin(common_zip_code)) & (df_immigration["Gender"] == "Ambos sexos")]



# filter the common cities

df_income = df_income[(df_income["Zip_code"].isin(common_zip_code))]

gdf = gdf[gdf["Zip_code"].isin(common_zip_code)]
# the selected columns to create a unique dataframe

lc_def = [

    "Province",

    "City_name",

    "Zip_code",

    "Immigrants_over_total",

    "Average_net_income",

    "geometry"

]



# geopandas can work simultaneously with pandas

# so we can easily merge our 3 data sources into 1 dataframe



# NOTICE: that we have filtered previously our dataframes to the common cities.

gdf = gdf.merge(df_immigration, on = "Zip_code")

gdf = gdf.merge(df_income, on = "Zip_code")

gdf = gdf[lc_def]
gdf.head()
del df_immigration, df_income
# We do the following operation (NR_COLORS * 2) - 1

# because we want the withe color to be common to both axis

full_palette = sns.color_palette('PiYG_r', n_colors = (NR_COLORS*2)-1)

sns.palplot(full_palette)
# Now let's separate x and y from our color palette

# x axis

# here is how we can acces the colors

cmap_x = full_palette[NR_COLORS - 1:]

sns.palplot(cmap_x)
# y axis

# here is how we can acces the colors

cmap_y = list(reversed(full_palette[:NR_COLORS]))

sns.palplot(cmap_y)
# we will only visualize the data for Madrid

# otherwise we won't be able to fit all the data in one plot

gdf_ = gdf[gdf["Province"] == PROVINCE]
migrant_rate_bin = generate_bins(gdf_["Immigrants_over_total"], NR_COLORS = NR_COLORS)



# instanciante the plot

fig = plt.figure(figsize = (10, 10))

ax = fig.add_subplot()



# iterate over all colors

for color in range(NR_COLORS):

    # get the areas based on the corresponding bin

    # .yb creates an array of all the areas from Madrid

    areas = np.where(migrant_rate_bin.yb == color)[0]

    # geopandas will plot automatically based on the geometry column

    gdf_.iloc[areas].plot(color = rgb2hex(cmap_x[color]), ax = ax)

    

# get rid of additional information

ax.set_axis_off()



# set a title 

ax.set_title('% of immigrants of the total population of the city from {} region'.format(PROVINCE), loc = 'left', fontweight = 'bold', fontsize = 14);
income_rate_bin = generate_bins(gdf_["Average_net_income"], NR_COLORS = NR_COLORS)



# instanciante the plot

fig = plt.figure(figsize = (10, 10))

ax = fig.add_subplot()



# iterate over all colors

for color in range(NR_COLORS):

    # get the areas based on the corresponding bin

    # .yb creates an array of all the areas from Madrid

    areas = np.where(income_rate_bin.yb == color)[0]

    # geopandas will plot automatically based on the geometry column

    gdf_.iloc[areas].plot(color = rgb2hex(cmap_y[color]), ax = ax)

    

# get rid of additional information

ax.set_axis_off()



# set a title 

ax.set_title('Average income in the city (binnarized over all the cities from the {} region)'.format(PROVINCE), loc = 'left', fontweight = 'bold', fontsize = 14);
# create to store the colors

cmap_xy = []



# the dictionary is useful to acces the colors in HEX by a tuple of (0, 0), (0, 1) when we iterate

bivariate_palette = {}



# color brewing

for i in range(NR_COLORS):

    for j in range(NR_COLORS):

        

        x = spectra.rgb(*cmap_x[i][0:NR_COLORS - 1])

        y = spectra.rgb(*cmap_y[j][0:NR_COLORS - 1])

        

        if i == j and i == 0:

            cmap_xy.append(x.darken(1.5).rgb)

            

        elif i == 0:

            cmap_xy.append(y.rgb)

            

        elif j == 0:

            cmap_xy.append(x.rgb)

            

        else: 

            blended = x.blend(y, ratio=0.5)

            

            if i == j:

                blended = blended.saturate(7.5 * (i + 1))

            else:

                blended = blended.saturate(4.5 * (i + 1))

                

            cmap_xy.append(blended.rgb)

            

        # depending on the NR_COLORS you selected, you might get an error of RGBA must be less than 1

        try:

            bivariate_palette[(i, j)] = rgb2hex(cmap_xy[-1])

            

        except:

            print(i, j)

            print(cmap_xy[-1])

        

# reshape the array to plot it

cmap_xy = np.array(cmap_xy).reshape(NR_COLORS, NR_COLORS, 3)
plt.imshow(cmap_xy)
plot_bivariate_map(gdf_ = gdf_, 

                   migrant_rate_bin = migrant_rate_bin, 

                   income_rate_bin = income_rate_bin, 

                   bivariate_palette = bivariate_palette, 

                   PROVINCE = PROVINCE, 

                   NR_COLORS = NR_COLORS)
# swap the province and plot Madrid Region

PROVINCE = "Madrid"

gdf_ = gdf[gdf["Province"] == PROVINCE]



# We must calculate the new bins both for immigration and income, because the previous were for Barcelona Region.

migrant_rate_bin = generate_bins(gdf_["Immigrants_over_total"], NR_COLORS = NR_COLORS)

income_rate_bin = generate_bins(gdf_["Average_net_income"], NR_COLORS = NR_COLORS)



plot_bivariate_map(gdf_ = gdf_, 

                   migrant_rate_bin = migrant_rate_bin, 

                   income_rate_bin = income_rate_bin, 

                   bivariate_palette = bivariate_palette, 

                   PROVINCE = PROVINCE, 

                   NR_COLORS = NR_COLORS);
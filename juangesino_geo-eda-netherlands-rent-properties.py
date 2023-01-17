from IPython.display import display, Markdown, Latex

import json

import numpy as np

import pandas as pd

import re

import matplotlib.pyplot as plt

import seaborn as sns



import descartes

import geopandas as gpd

from shapely.geometry import Point, Polygon

import contextily as ctx



# Viz config

plt.style.use('seaborn')

colors = ["#4B88A2", "#BC2C1A", "#DACC3E", "#EA8C55", "#7FB7BE"]

palette = sns.color_palette(colors)

sns.set_palette(palette)

alpha = 0.7
raw_data_file = "/kaggle/input/netherlands-rent-properties/properties.json"



def load_raw_data(filepath):

    raw_data = []

    for line in open(filepath, 'r'):

        raw_data.append(json.loads(line))

    

    df = pd.DataFrame(raw_data)

    

    return df

    

df = load_raw_data(raw_data_file)



Markdown(f"Successfully imported DataFrame with shape: {df.shape}.")
# Functions from: https://www.kaggle.com/juangesino/starter-netherlands-rent-properties



# Define all columns that need to be flatten and the property to extract

flatten_mapper = {

    "_id": "$oid",

    "crawledAt": "$date",

    "firstSeenAt": "$date",

    "lastSeenAt": "$date",

    "detailsCrawledAt": "$date",

}





# Function to do all the work of flattening the columns using the mapper

def flatten_columns(df, mapper):

    

    # Iterate all columns from the mapper

    for column in flatten_mapper:

        prop = flatten_mapper[column]

        raw_column_name = f"{column}_raw"

        

        # Check if the raw column is already there

        if raw_column_name in df.columns:

            # Drop the generated one

            df.drop(columns=[column], inplace=True)

            

            # Rename the raw back to the original

            df.rename(columns={ raw_column_name: column }, inplace=True)        

    

        # To avoid conflicts if re-run, we will rename the columns we will change

        df.rename(columns={

            column: raw_column_name,

        }, inplace=True)



        # Get the value inside the dictionary

        df[column] = df[raw_column_name].apply(lambda obj: obj[prop])

        

    return df





def rename_columns(df):

    # Store a dictionary to be able to rename later

    rename_mapper = {}

    

    # snake_case REGEX pattern

    pattern = re.compile(r'(?<!^)(?=[A-Z])')

    

    # Iterate the DF's columns

    for column in df.columns:

        rename_mapper[column] = pattern.sub('_', column).lower()

        

    # Rename the columns using the mapper

    df.rename(columns=rename_mapper, inplace=True)

    

    return df





def parse_types(df):

    

    df["crawled_at"] = pd.to_datetime(df["crawled_at"])

    df["first_seen_at"] = pd.to_datetime(df["first_seen_at"])

    df["last_seen_at"] = pd.to_datetime(df["last_seen_at"])

    df["details_crawled_at"] = pd.to_datetime(df["details_crawled_at"])

    df["latitude"] = pd.to_numeric(df["latitude"])

    df["longitude"] = pd.to_numeric(df["longitude"])

    

    return df





def add_features(df):

    

    df["rent_per_area"] = df["rent"] / df["area_sqm"]

    

    return df
df = (df

      .pipe(flatten_columns, mapper=flatten_mapper)

      .pipe(rename_columns)

      .pipe(parse_types)

      .pipe(add_features)

     )
geometry = [Point(xy) for xy in zip(df['longitude'],df['latitude'])]

crs = { 'init': 'epsg:4326' }

gdf = gpd.GeoDataFrame(

    df,

    crs = crs,

    geometry = geometry

)
netherlands_map = "/kaggle/input/netherlands-rent-properties/2019_ggd_regios_kustlijn.gpkg"



nl_map = gpd.read_file(netherlands_map)

fig, ax = plt.subplots(figsize=(10, 10))

nl_map.to_crs(epsg=3857).plot(ax=ax, alpha=0.1, color='grey', edgecolor='black')

gdf.to_crs(epsg=3857).plot(ax=ax, markersize=10, color=colors[0], marker='o')

ax.set_title('Properties in the Netherlands')

ax.set_axis_off()
fig, ax = plt.subplots(figsize=(10,10))

nl_map.to_crs(epsg=3857).plot(ax=ax, alpha=0.1, color='grey', edgecolor='black')

gdf.sort_values('rent_per_area').to_crs(epsg=3857).plot(ax=ax, markersize=10, marker='o', column='rent_per_area', cmap='inferno_r')

ax.set_title('Rental prices Netherlands')

ax.set_axis_off()

# Create legend scale

cax = fig.add_axes([1, 0.5, 0.01, 0.2])

sm = plt.cm.ScalarMappable(cmap='inferno_r', norm=plt.Normalize(vmin=gdf['rent_per_area'].min(), vmax=gdf['rent_per_area'].max()))

sm._A = []

cbar = fig.colorbar(sm, cax=cax,)

cbar.set_label('\nPrice per $m^2$ $(€/m^2)$')

cbar.ax.tick_params(labelsize=10)
gdf = gdf[np.abs(gdf['area_sqm']-gdf['area_sqm'].mean()) <= (3*gdf['area_sqm'].std())]

gdf = gdf[np.abs(gdf['rent']-gdf['rent'].mean()) <= (3*gdf['rent'].std())]
fig, ax = plt.subplots(figsize=(10,10))

nl_map.to_crs(epsg=3857).plot(ax=ax, alpha=0.1, color='grey', edgecolor='black')

gdf.sort_values('rent_per_area').to_crs(epsg=3857).plot(ax=ax, markersize=10, marker='o', column='rent_per_area', cmap='inferno_r')

ax.set_title('Rental prices Netherlands')

ax.set_axis_off()

# Create legend scale

cax = fig.add_axes([1, 0.5, 0.01, 0.2])

sm = plt.cm.ScalarMappable(cmap='inferno_r', norm=plt.Normalize(vmin=gdf['rent_per_area'].min(), vmax=gdf['rent_per_area'].max()))

sm._A = []

cbar = fig.colorbar(sm, cax=cax,)

cbar.set_label('\nPrice per $m^2$ $(€/m^2)$')

cbar.ax.tick_params(labelsize=10)
# Extremly slow and inefficient. This can probably be improved by using Pandas' `.apply`

for index, row in nl_map.to_crs(epsg=3857).iterrows():

    area = row.geometry

    intersect = gdf[gdf.to_crs(epsg=3857).within(area)]

    nl_map.loc[index, 'properties'] = len(intersect)

    nl_map.loc[index, 'mean_rent'] = intersect['rent'].mean()

    nl_map.loc[index, 'mean_rent_per_sqm'] = intersect['rent_per_area'].mean()

nl_map = nl_map.fillna(0)
f,ax = plt.subplots(1, 3, figsize=(15,15), subplot_kw=dict(aspect='equal'))



nl_map.to_crs(epsg=3857).plot(ax=ax[0], column='mean_rent', cmap='inferno_r', alpha=0.8, edgecolor='black')

ax[0].set_title("Average Rent (€)")

ax[0].set_axis_off()



nl_map.to_crs(epsg=3857).plot(ax=ax[1], column='mean_rent_per_sqm', cmap='inferno_r', alpha=0.8, edgecolor='black')

ax[1].set_title("Average Rent (€/m2)")

ax[1].set_axis_off()



nl_map.to_crs(epsg=3857).plot(ax=ax[2], column='properties', cmap='inferno_r', alpha=0.8, edgecolor='black')

ax[2].set_title("Total Properties")

ax[2].set_axis_off()


# Create a map.

def print_map(centerX, centerY, title=None, 

              df=None, radius=6000, zoom=14, 

              height=8, width=8, markersize=10, 

              legend=False, ax=None, axis=False, 

              column=None, color="blue", tiles=False, 

              cmap="inferno_r", marker='o', epsg=3857, 

              legend_label=''):

    """Print a map in a figure

    

    Prints a map in a figure/axis



    Parameters

    ----------

    centerx : float

        The X coordinate for the center of the map.

    centery : float

        The Y coordinate for the center of the map.

    title : str, optional

        A string to be used as a title for the map.

    df : pandas.DataFame, optional

        Pandas DataFrame to be used as data for plotting.

    radius : int, optional

        Coordinate radius around the center to be show

        in the map. Notice this measure will be in whichever

        coordinate system your map is.

    zoom : int

        A level of zoom to be used when getting the tiles.

    height : int, optional

        If no set axis are provided, this will be used

        to create the figure and axis.

    width : int, optional

        If no set axis are provided, this will be used

        to create the figure and axis.

    markersize : int, optional

        When displaying data, this will be used for the

        size of the data points.

    legend : bool, optional

        Choose whether to show a legend or not.

    ax : `.axes.Axes` object, optional

        Single `~matplotlib.axes.Axes` object to be

        used when plotting the map.

    axis : bool, optional

        Choose if showing the axis values.

    column : str, optional

        A sequence of column identifiers to plot.

    color : str, optional

        `~matplotlib.colors.Colormap` instance, or 

        the name of a registered colormap.

    tile : bool, optional

        Choose if showing tiles map as background.

    cmap : str, optional

        `~matplotlib.colors.Colormap` instance, or 

        the name of a registered colormap.

    marker : str, optional

        `~matplotlib.markers`, or the name of a registered maker.

        https://matplotlib.org/_modules/matplotlib/markers.html#MarkerStyle

    epsg : str, optional

        Coordinate system EPSG indentifier.

    legend_label : str, optional

        Text to be displayed as the title of the legend.

    """

    

    if ax is None:

        fig, ax = plt.subplots(figsize=(height,width))

    

    if df is not None:

        if column is None:

            df.to_crs(epsg=epsg).plot(

                ax=ax, markersize=markersize, 

                marker=marker, color=color)

        else:

            df.to_crs(epsg=epsg).plot(

                ax=ax, markersize=markersize,

                marker=marker, column=column, cmap=cmap)

    

    ax.set_xlim((centerX-radius), (centerX+radius))

    ax.set_ylim((centerY-radius), (centerY+radius))

    

    if not axis:

        ax.set_axis_off()

    

    # Add a color bar legend.

    if legend and column and df is not None:

        cax = fig.add_axes([1, 0.5, 0.01, 0.2])

        sm = plt.cm.ScalarMappable(

            cmap=cmap, 

            norm=plt.Normalize(

                vmin=df[column].min(), 

                vmax=df[column].max()))

        sm._A = []

        cbar = fig.colorbar(sm, cax=cax,)

        cbar.set_label(legend_label)

        cbar.ax.tick_params(labelsize=10)

    

    if title is not None:

        ax.set_title(title)

    

    if tiles:

        tile_url = 'http://services.arcgisonline.com/arcgis/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'

        add_basemap(ax, zoom, url=tile_url)



        

def add_basemap(ax, zoom, url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png'):

    """Add tiles basemap.

    

    Adds a background map using tiles.



    Parameters

    ----------

    ax : `.axes.Axes` object or array of Axes objects.

        *ax* can be either a single `~matplotlib.axes.Axes` object or an

        array of Axes objects if more than one subplot was created.  The

        dimensions of the resulting array can be controlled with the squeeze

        keyword, see above.

    zoom : int

        A level of zoom to be used when getting the tiles.

    url :  str, optional

        The tile url to be used to get the tiles.

    """

    xmin, xmax, ymin, ymax = ax.axis()

    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)

    ax.imshow(basemap, extent=extent, interpolation='bilinear')

    ax.axis((xmin, xmax, ymin, ymax))
f, ax = plt.subplots(figsize=(15,15), subplot_kw=dict(aspect='equal'))



print_map(545234.877387, 6867523.024796, ax=ax, df=gdf, column='rent_per_area', title="Amsterdam", tiles=True)



cax = f.add_axes([1, 0.5, 0.01, 0.2])

sm = plt.cm.ScalarMappable(cmap='inferno_r', norm=plt.Normalize(vmin=df['rent_per_area'].min(), vmax=df['rent_per_area'].max()))

sm._A = []

cbar = f.colorbar(sm, cax=cax,)

cbar.set_label('\nPrice per $m^2$ $(€/m^2)$')

cbar.ax.tick_params(labelsize=10)
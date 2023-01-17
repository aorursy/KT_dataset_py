# conda create -n msc-ds-core-pds-assignment-1 python=3.6
# source activate msc-ds-core-pds-assignment-1
# pip install -r requirements.txt
from __future__ import print_function
import sys
import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# We need this for palette management
from matplotlib.colors import ListedColormap
import seaborn as sns
from sodapy import Socrata

from IPython.display import display, HTML
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

# Matplotlib config
%matplotlib inline

# Seaborn config
sns.set(style = "darkgrid")

# Set a random seed so that possible random operations performed by us can be replicated by others.
np.random.seed(19730618)
# Global functions

# Helper function that will redirect printing to stderr.
# Used when printing for exception handling etc.
def print_stderr(*args, **kwargs):
    print(*args, file = sys.stderr, **kwargs)

# Returns the ceiling of a number, rounded according to the 'to' parameter
# i.e. to thousands, to hundred-thousands etc.
def ceiling(num, to = 100000):
    y, z = divmod(num, to)
    
    return int(to * (y + 1))

def normalize(df_column):
    return (df_column - df_column.min()) / (df_column.max() - df_column.min())
crimes_csv = "../input/city_of_chicago_crimes_2001_to_present.csv"

crimes_df = pd.read_csv(crimes_csv, delimiter = ",", header = 0)
crimes_df.head(5)
crimes_df.columns
crimes_df = crimes_df.drop(columns = [
    "ID", 
    "Case Number", 
    "Block", 
    "IUCR", 
    "Description", 
    "Location Description", 
    "Arrest",
    "Domestic",
    "FBI Code", 
    "Year", 
    "Updated On",
    "Location"
])
# Replace any spaces with underscores
crimes_df.columns = crimes_df.columns.str.replace(" ", "_")

# Make sure we strip any invisible white spaces
crimes_df.columns = crimes_df.columns.str.strip()

# Set all columns names to lower case
crimes_df.columns = crimes_df.columns.str.lower()
crimes_df.columns
crimes_df.isnull().any()
crimes_df["beat"] = crimes_df["beat"].fillna(value = 0.0)
crimes_df["district"] = crimes_df["district"].fillna(value = 0.0)
crimes_df["ward"] = crimes_df["ward"].fillna(value = 0.0)
crimes_df["community_area"] = crimes_df["community_area"].fillna(value = 0.0)
crimes_df["x_coordinate"] = crimes_df["x_coordinate"].fillna(value = 0.0)
crimes_df["y_coordinate"] = crimes_df["y_coordinate"].fillna(value = 0.0)
crimes_df["latitude"] = crimes_df["latitude"].fillna(value = 0.0)
crimes_df["longitude"] = crimes_df["longitude"].fillna(value = 0.0)
crimes_df.isnull().any()
for column in crimes_df.columns:
    print("Column name: ", column, " / Data type: ", type(crimes_df[column][0]))
# Including the date format dramatically increases performance.
crimes_df["date"] = pd.to_datetime(crimes_df["date"], format = "%m/%d/%Y %I:%M:%S %p", utc = True)
crimes_df["beat"] = crimes_df["beat"].astype(np.int64)
crimes_df["district"] = crimes_df["district"].astype(np.int64)
crimes_df["ward"] = crimes_df["ward"].astype(np.int64)
crimes_df["community_area"] = crimes_df["community_area"].astype(np.int64)
crimes_df = crimes_df.set_index("date")
crimes_df = crimes_df.sort_index()
crimes_df.index
crimes_df = crimes_df.loc['2001-01-01':'2017-12-31']
# Convert primary_type to a categorical variable
crimes_df["primary_type"] = crimes_df["primary_type"].astype("category")
###
# Group our data
##
# Group our data by year
crimes_gb_y = crimes_df.groupby(by = [crimes_df.index.year], axis = 0)

# Count the number of incidents of our grouped set
crimes_gb_y_pt_count = crimes_gb_y["primary_type"].count()

# Get the max crime count so that we can normalize the plotting axis
crimes_gb_y_pt_max = crimes_gb_y_pt_count.max()

###
# Plot our data
##
plot_main_title_font = {
    "family": "sans serif",
    "color":  "black",
    "weight": "bold",
    "size": 18,
}

plot_axis_title_font = {
    "family": "sans serif",
    "color":  "black",
    "weight": "bold",
    "size": 14,
}

ax = crimes_gb_y_pt_count.plot(
    kind = "line", 
    marker = ".",
    stacked = False, 
    figsize = (16, 10), 
    use_index = True, 
    legend = False
)

# Normalize the y-axis values
ax.set_ylim(0, ceiling(crimes_gb_y_pt_max, to = 1000000))

# Set graph title
ax.set_title(label = "Yearly count of crimes", fontdict = plot_main_title_font, pad = 20)

# Set axis proper labels
ax.set_xlabel(xlabel = "Year", fontdict = plot_axis_title_font, labelpad = 20)
ax.set_ylabel(ylabel = "Incident frequency", fontdict = plot_axis_title_font, labelpad = 20)

# Set x-axis ticks to be the index of our series object
ax.set_xticks(ticks = crimes_gb_y_pt_count.index);

# Auto-format year labels
ax.get_figure().autofmt_xdate();
###
# Compute some statistics
##
# Convert the series to a dataframe
crimes_gb_y_pt_count_df = crimes_gb_y_pt_count.to_frame()

# Compute the per year incident count difference
crimes_gb_y_pt_count_df["diff_from_previous_year"] = crimes_gb_y_pt_count_df["primary_type"].diff(periods = 1)

# Compute the per year incident difference as a percentage
crimes_gb_y_pt_count_df["diff_from_previous_year_perc"] = crimes_gb_y_pt_count_df["primary_type"].pct_change(periods = 1)

crimes_gb_y_pt_count_df
num_primary_types = 10
num_months = 12

###
# Group our data
##
# Group our data by month
crimes_gb_mpt = crimes_df.groupby(by = [crimes_df.index.month, "primary_type"], axis = 0)

# Count the number of incidents of our grouped set
crimes_gb_mpt_pt_count = crimes_gb_mpt["primary_type"].count()

# We may chose to display only some of the crime class, for readability
if num_primary_types != None:
    crimes_gb_mpt_pt_count = crimes_gb_mpt_pt_count.nlargest(num_primary_types * num_months)

crimes_gb_mpt_pt_count_un = crimes_gb_mpt_pt_count.unstack()

###
# Plot our data
##
months = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December",
}

plot_main_title_font = {
    "family": "sans serif",
    "color":  "black",
    "weight": "bold",
    "size": 18,
}

plot_axis_title_font = {
    "family": "sans serif",
    "color":  "black",
    "weight": "bold",
    "size": 14,
}

# Since the catecorical variable 'primary_type' has many levels,  
# we are required to create a color map such that different levels
# can be as distinguishable as possible...
cat_cmap = ListedColormap(sns.color_palette("BrBG", 10).as_hex())

ax = crimes_gb_mpt_pt_count_un.plot(
    kind = "bar", 
    stacked = True, 
    figsize = (14, 10), 
    use_index = False, 
    legend = True,
    cmap = cat_cmap
)

# Normalize the y-axis values
ax.set_ylim(0, ceiling(crimes_gb_mpt_pt_count.max(), to = 1000000))

# Set graph title
ax.set_title(label = "Aggregated monthly count of crimes", fontdict = plot_main_title_font, pad = 20)

# Set axis proper labels
ax.set_xlabel(xlabel = "Month", fontdict = plot_axis_title_font, labelpad = 20)
ax.set_ylabel(ylabel = "Aggregated incident frequency", fontdict = plot_axis_title_font, labelpad = 20)

# Set x-axis ticks and corresponding labels to be the index of our series object
ax.set_xticks(ticks = crimes_gb_mpt_pt_count.index.levels[0] - 1);
ax.set_xticklabels(labels = [months[x] for x in crimes_gb_mpt_pt_count.index.levels[0]])

# Auto-format year labels
ax.get_figure().autofmt_xdate();

ax.legend(bbox_to_anchor = (1, 1.01));
# We need this for palette management
from matplotlib.colors import ListedColormap

num_primary_types = 10
num_hours = 24

###
# Group our data
##
# Group our data by hour
crimes_gb_hpt = crimes_df.groupby(by = [crimes_df.index.hour, "primary_type"], axis = 0)

# Count the number of incidents of our grouped set
crimes_gb_hpt_pt_count = crimes_gb_hpt["primary_type"].count()

# We may chose to display only some of the crime class, for readability
if num_primary_types != None:
    crimes_gb_hpt_pt_count = crimes_gb_hpt_pt_count.nlargest(num_primary_types * num_hours)
    
crimes_gb_hpt_pt_count_un = crimes_gb_hpt_pt_count.unstack()

###
# Plot our data
##
plot_main_title_font = {
    "family": "sans serif",
    "color":  "black",
    "weight": "bold",
    "size": 18,
}

plot_axis_title_font = {
    "family": "sans serif",
    "color":  "black",
    "weight": "bold",
    "size": 14,
}

# Since the catecorical variable 'primary_type' has many levels,  
# we are required to create a color map such that different levels
# can be as distinguishable as possible...
cat_cmap = ListedColormap(sns.color_palette("BrBG", 10).as_hex())

ax = crimes_gb_hpt_pt_count_un.plot(
    kind = "bar", 
    stacked = True, 
    figsize = (14, 10), 
    use_index = False, 
    legend = True,
    cmap = cat_cmap
)

# Normalize the y-axis values
ax.set_ylim(0, ceiling(crimes_gb_hpt_pt_count.max(), to = 1000000))

# Set graph title
ax.set_title(label = "Aggregated hourly count of crimes", fontdict = plot_main_title_font, pad = 20)

# Set axis proper labels
ax.set_xlabel(xlabel = "Hour", fontdict = plot_axis_title_font, labelpad = 20)
ax.set_ylabel(ylabel = "Aggregated incident frequency", fontdict = plot_axis_title_font, labelpad = 20)

# Rotate x-axis labels so as to appear vertical
for tick in ax.get_xticklabels():
    tick.set_rotation(0)

ax.legend(bbox_to_anchor = (1, 1.01));
fig, (ax1, ax2) = plt.subplots(nrows = 2, ncols = 1, sharex = False, sharey = False, squeeze = True)

###
# Plot 1
##
sub_ax_1 = crimes_gb_mpt_pt_count_un.plot(
    kind = "bar", 
    stacked = True, 
    figsize = (20, 20), 
    use_index = False, 
    legend = True,
    cmap = cat_cmap,
    ax = ax1
)

# Normalize the y-axis values
sub_ax_1.set_ylim(0, ceiling(crimes_gb_mpt_pt_count.max(), to = 1000000))

# Set graph title
sub_ax_1.set_title(label = "Aggregated monthly count of crimes", fontdict = plot_main_title_font, pad = 20)

# Set axis proper labels
sub_ax_1.set_xlabel(xlabel = "Month", fontdict = plot_axis_title_font, labelpad = 20)
sub_ax_1.set_ylabel(ylabel = "Aggregated incident frequency", fontdict = plot_axis_title_font, labelpad = 20)

# Set x-axis ticks and corresponding labels to be the index of our series object
sub_ax_1.set_xticks(ticks = crimes_gb_mpt_pt_count.index.levels[0] - 1);
sub_ax_1.set_xticklabels(labels = [months[x] for x in crimes_gb_mpt_pt_count.index.levels[0]])

# Auto-format year labels
sub_ax_1.get_figure().autofmt_xdate();

sub_ax_1.legend(bbox_to_anchor = (1, 1.01));


###
# Plot 2
##
sub_ax_2 = crimes_gb_hpt_pt_count_un.plot(
    kind = "bar", 
    stacked = True, 
    figsize = (20, 20), 
    use_index = False, 
    legend = True,
    cmap = cat_cmap,
    ax = ax2
);

# Normalize the y-axis values
sub_ax_2.set_ylim(0, ceiling(crimes_gb_hpt_pt_count.max(), to = 1000000))

# Set graph title
sub_ax_2.set_title(label = "Aggregated hourly count of crimes", fontdict = plot_main_title_font, pad = 20)

# Set axis proper labels
sub_ax_2.set_xlabel(xlabel = "Hour", fontdict = plot_axis_title_font, labelpad = 20)
sub_ax_2.set_ylabel(ylabel = "Aggregated incident frequency", fontdict = plot_axis_title_font, labelpad = 20)

# Rotate x-axis labels so as to appear vertical
for tick in ax.get_xticklabels():
    tick.set_rotation(0)

sub_ax_2.legend(bbox_to_anchor = (1, 1.01))

fig.subplots_adjust(hspace = 0.5)
fig.savefig(fname = "crime_trends.pdf", dpi = 150, papertype = "a4", format = "pdf", facecolor = "w", edgecolor = "b", orientation = "landscape", pad_inches = 3);
###
# Group our data
##
# Group our data by month
crimes_gb_mpt = crimes_df.groupby(by = ["primary_type", crimes_df.index.month], axis = 0)

# Count the number of incidents of our grouped set
crimes_gb_mpt_count = crimes_gb_mpt["primary_type"].count()

crimes_gb_mpt_count_un = crimes_gb_mpt_count.unstack()

###
# Plot our data
##
plot_main_title_font = {
    "family": "sans serif",
    "color":  "black",
    "weight": "bold",
    "size": 18,
}

plot_axis_title_font = {
    "family": "sans serif",
    "color":  "black",
    "weight": "bold",
    "size": 14,
}

fig, ax = plt.subplots(figsize = (12, 16))
sns.heatmap(data = crimes_gb_mpt_count_un, annot = True, fmt = ".0f", linewidths = .0, center = crimes_gb_mpt_count.max() / 2, cmap = "YlGnBu", ax = ax)

# Set graph title
ax.set_title(label = "Heatmap of crimes per month / incident class", fontdict = plot_main_title_font, pad = 20)

# Set axis proper labels
ax.set_xlabel(xlabel = "Month", fontdict = plot_axis_title_font, labelpad = 20)
ax.set_ylabel("Incident class", fontdict = plot_axis_title_font, labelpad = 20);
import matplotlib
import matplotlib.cm as cm

# Required for palette management
from matplotlib.colors import ListedColormap

import folium

# Required for creating the heatmap of crimes
from folium import plugins

def create_heatmap(df, group_by = "district", point_radius = 10, heatmap_radius = 30):
    # Work on our dataset
    # Work on a copy of the original dataframe
    df_c = df.copy(deep = True)
    
    # Exclude 0s
    df_c = df_c[(df_c["latitude"] != 0) & (df_c["longitude"] != 0)]
    
    # Group by the supplied field, perform aggregations
    # and get back our summarized dataframe
    df_c_gb_df = df_c.groupby(by = [group_by], axis = 0).agg({"latitude": ["mean"], "longitude": ["mean"], "primary_type": ["count"]})
    
    # Create an array required for the heatmap plugin
    df_c_gb_df_lat_long_arr = df_c_gb_df[["latitude", "longitude"]].values
    
    # Find the map location start positions
    lat_start = df_c_gb_df["latitude"]["mean"].median()
    long_start = df_c_gb_df["longitude"]["mean"].median()
    
    # Create the map var
    folium_map = folium.Map(
        tiles = "Stamen Toner",
        location = [lat_start, long_start],
        zoom_start = 10
    )
    
    # Create a descriptive string from the group_by argument
    # Will be used for map tooltip display purposes
    geo_aggr_level_name = group_by.replace("_", " ").strip().capitalize()
    
    # Create points on the map
    for i in range(0, len(df_c_gb_df.index)):
        
        lat = df_c_gb_df.loc[df_c_gb_df.index[i], ["latitude", "mean"]][0]
        long = df_c_gb_df.loc[df_c_gb_df.index[i], ["longitude", "mean"]][0]
        geo_area_id = df_c_gb_df.index[i]
        crime_count = df_c_gb_df.loc[df_c_gb_df.index[i], ["primary_type", "count"]][0]
        
        circle_marker = folium.CircleMarker(
            location = [lat, long],
            radius = point_radius,
            tooltip = \
                geo_aggr_level_name + " " + "<strong>" + str(geo_area_id) + "</strong>" + \
                "<hr/>" + \
                "Crime count " + "<strong>" + str(int(crime_count)) + "</strong>",
            color = "#FFFFFF",
            fill = True,
            fill_color = "#000000"
        )
        
        circle_marker.add_to(folium_map);
    
    # Add the heatmap
    folium_map.add_child(plugins.HeatMap(df_c_gb_df_lat_long_arr, radius = heatmap_radius))
    
    return folium_map
# Animated
from datetime import datetime, timedelta

import matplotlib
import matplotlib.cm as cm

# Required for palette management
from matplotlib.colors import ListedColormap

import folium

# Required for creating the heatmap of crimes
from folium import plugins

def create_animated_heatmap(df, group_by = "district", heatmap_radius = 30):
    # Work on our dataset
    # Work on a copy of the original dataframe
    df_c = df.copy(deep = True)
    
    # Exclude 0s
    df_c = df_c[(df_c["latitude"] != 0) & (df_c["longitude"] != 0)]
    
    # Group by the supplied field, perform aggregations
    # and get back our summarized dataframe
    df_c_gb_df = df_c.groupby(by = [df_c.index.year, group_by], axis = 0).agg({"latitude": ["mean"], "longitude": ["mean"], "primary_type": ["count"]})
    
    # Flatten the grouped dataset
    df_map = df_c_gb_df.reset_index()
    
    # Pick the required columns for creating the animated heatmap
    df_map = df_map[["latitude", "longitude", "primary_type", "date"]]
    
    # Create an array of data points required by the animated heatmap plugin
    df_map_data_gb_d = df_map.groupby(by = "date")
    df_map_data_arr = [item.tolist() for item in [df_map_data_gb_d.get_group(g)[["latitude", "longitude", "primary_type"]].values for g in df_map_data_gb_d.groups]]
    
    # Create an array of time intervals required by the animated heatmap plugin
    df_map_time_arr = np.unique(df_map[["date"]].values)
    
    # Flatten and format date from %Y to %Y-%m-%d
    df_map_time_arr = [str(year) + "-01-01" for year in df_map_time_arr]
    
    # Find the map location start positions
    lat_start = df_c_gb_df["latitude"]["mean"].median()
    long_start = df_c_gb_df["longitude"]["mean"].median()
    
    # Create the map var
    folium_map = folium.Map(
        tiles = "Stamen Toner",
        location = [lat_start, long_start],
        zoom_start = 10
    )
    
    heat_map_with_time = plugins.HeatMapWithTime( 
        data = df_map_data_arr,
        index = df_map_time_arr,
        radius = heatmap_radius,
        display_index = True,
        auto_play = True
    )

    heat_map_with_time.add_to(folium_map)
    
    return folium_map
create_heatmap(crimes_df, group_by = "beat", point_radius = 2, heatmap_radius = 30)
create_heatmap(crimes_df, group_by = "ward", point_radius = 2, heatmap_radius = 30)
create_heatmap(crimes_df, group_by = "district", point_radius = 2, heatmap_radius = 30)
create_heatmap(crimes_df, group_by = "community_area", point_radius = 2, heatmap_radius = 30)
create_animated_heatmap(crimes_df, group_by = "district", heatmap_radius = 30)
def pareto(df, group_by = "district"):
    # Work on our dataset
    # Work on a copy of the original dataframe
    df_c = crimes_df.copy(deep = True)
    
    # Group by the provided geographical level,
    # counting crimes instances
    df_c_gb_pt_df = df_c.groupby(by = [group_by], axis = 0).agg({"primary_type": ["count"]})
    
    # Sort by crimes count, descending
    df_c_gb_pt_df = df_c_gb_pt_df.sort_values(by = ("primary_type", "count"), ascending = False)
    
    # Calculate the percentage of crimes each geographic region
    # contributes to the whole
    df_c_gb_pt_df["percentage"] = df_c_gb_pt_df[("primary_type", "count")] / df_c_gb_pt_df[("primary_type", "count")].sum()
    
    # Calculate the running sum of percentages
    df_c_gb_pt_df["percentage_cumsum"] = df_c_gb_pt_df["percentage"].cumsum()
    
    # Return the percentage of observations up to and including contribution
    # to 80% of cases, divided by the total number of observations.
    return df_c_gb_pt_df[df_c_gb_pt_df["percentage_cumsum"] <= .8].iloc[:, 0].count() / df_c_gb_pt_df.iloc[:, 0].count()
print("Pareto ratio: ", pareto(crimes_df, group_by = "district"))
# https://dev.socrata.com/foundry/data.cityofchicago.org/6zsd-86xi
# Create a client to the City of Chicago data source

# Some variables required for accessing the City of Chicago API
domain = "data.cityofchicago.org"
# Ommited for privacy
app_token = "<your_app_token>"
dataset_identifier = "6zsd-86xi"

starting_date = "2001-01-01T00:00:00"

# Dataset throttle control
data_limit = 1000
data_offset = 0

temp_data = []

# The City of Chicago API feeds request utilising a paging mechanism.
# Therefore, we need to provide a way to page our requests to the API by utilising
# 'limit' and 'offset' variables in our query.
# Since there is no clear way for the API to indicate the end-of-data-stream location,
# we need a way to break from the while loop. This will happen when the length of the
# data returned from the API call is 0.

# Initialize the length variable to -1.
data_len = -1

with Socrata(domain=domain, app_token=app_token) as client:
    # Make sure we have the correct data set
    metadata = client.get_metadata(dataset_identifier = dataset_identifier)
    print(metadata["name"])
    print(metadata["description"])
    
    # Extract the dataset column names from the dataset metadata
    data_columns = [x["name"] for x in metadata["columns"]]
    
    while(data_len != 0):
        try:
            # Retrieve paged data, starting from 2010-01-01T00:00:00
            data = client.get(dataset_identifier = dataset_identifier, where = "date >= '" + starting_date + "'", order = "date ASC", content_type = "json", limit = data_limit, offset = data_offset)
            # Append the retrieved data to a temporary array
            temp_data += data
            # Set the offset to the next 'page' of data
            data_offset += 1000
            # Evaluate the data array length so that we know when to break from the while loop
            data_len = len(data)
            print("Data offset: ", data_offset)
        except Exception as e:
            print_stderr(e)
            break
    
    # Convert the entire array of data to a Pandas dataframe
    crimes_df = pd.DataFrame.from_records(data = temp_data)
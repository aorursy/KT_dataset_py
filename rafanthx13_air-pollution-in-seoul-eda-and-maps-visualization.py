# TODO News ideias
# + Evoluçâo do PM.2.5 pois é o que tem mais mediçôes negativas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import json
import datetime

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models.tools import HoverTool
from bokeh.models import GeoJSONDataSource
from bokeh.layouts import row
output_notebook()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Import DataFrame
file_path = '/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv'
df = pd.read_csv(file_path)
print("DataSet = {:,d} rows and {} columns".format(df.shape[0], df.shape[1]))

print("\nAll Columns:\n=>", df.columns.tolist())

quantitative = [f for f in df.columns if df.dtypes[f] != 'object']
qualitative = [f for f in df.columns if df.dtypes[f] == 'object']

print("\nStrings Variables:\n=>", qualitative,
      "\n\nNumerics Variables:\n=>", quantitative)

df.head()
# Information on the dangerousness of the concentration of each gas

df_gas_info = pd.read_csv('/kaggle/input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')
df_gas_info
# Create dict to pollutants level

list_cols = list(df_gas_info.columns)[3:]

level_dangerous = {}
for i in range(len(df_gas_info)):
    name = df_gas_info["Item name"][i]
    level_dangerous[name] = df_gas_info[list_cols].loc[i].tolist()
    
level_dangerous
# Import geo_json with geopandas

import geopandas as gpd

seoul_geojson = gpd.read_file('../input/seoul-map-geojson/seoul_municipalities_geo_simple.json')
seoul_geojson = seoul_geojson.drop(['code', 'base_year'], axis = 1)
seoul_geojson.head()
df['Address'].value_counts().head(3)
df['Address'] = df['Address'].map(lambda street: street.split(',')[2].strip())
df['Address'].value_counts().head(3)
df = df.drop(['Latitude', 'Longitude', 'Station code'], axis = 1).rename( columns = {'Address': 'District', 'Measurement date': 'Date'})
df['Date'] = pd.to_datetime(df['Date'])
df.info()
# Final dataSet

df.head()
import missingno as msno
sns.heatmap(df.isnull(), cbar=False)
def generate_GeoJSONSource_to_districts(my_df, column_value):
    """
    Generate GeoJSONDataSource. This is necessary to each part of GeoPlot except calculate low and high of colors
        By default must be 'District' in all df in.
    """
    with open('../input/seoul-map-geojson/seoul_municipalities_geo_simple.json') as json_file:
        data = json.load(json_file)
    if(len(my_df) != 25):
        raise Exception('df with len != 25')
    if('District' not in list(my_df.columns) ):
        raise Exception('df not contains "District"')
    for i in range(25):
        city = data['features'][i]['properties']['name_eng']
        index = my_df.query('District == "' + city +'"').index[0]
        data['features'][i]['properties'][column_value] = my_df[column_value][index]
        data['features'][i]['properties']['District'] = my_df['District'][index]
    geo_source = GeoJSONDataSource( geojson = json.dumps(data, separators=(',', ':')) )
    return geo_source  
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, HoverTool, ColorBar
from bokeh.palettes import magma,viridis,cividis, inferno

def eda_seoul_districts_geo_plot(geosource, df_in, title, column, state_column, low = -1, high = -1, palette = -1):
    """
    Generate Bokeh Plot to Brazil States:
        geosource: GeoJSONDataSource of Bokeh
        df_in: DataSet before transformed in GeoJSONDataSource
        title: title of plot
        column: column of df_in to be placed values in geoplot
        state_column: indicate column with names of States
        low = (optional) min value of range of color spectre
        high = (optional) max values of range of color spectre
        palette: (optional) can be magma, viridis, civis, inferno e etc.. (with number os colors)
            Example: cividis(8) (8 colors to classify), cividis(256)  (256, more colors to clasify)
    """
    if high == -1:
        high = max(df_in[column])
    if low == -1:
        low = min(df_in[column])
    if palette == -1:
        palette = inferno(32)
        
    palette = palette[::-1]
    color_mapper = LinearColorMapper(palette = palette, low = low, high = high)
    
    hover = HoverTool(tooltips = [ ('District','@{'+state_column+'}'), (column, '@{'+column+'}{%.6f}')],
                  formatters={'@{'+column+'}' : 'printf'})

    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8, width = 300, height = 20, 
                         border_line_color=None, location = (0,0),  orientation = 'horizontal')

    p = figure(title = title, plot_height = 430, plot_width = 330, tools = [hover])

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.xaxis.visible = False
    p.yaxis.visible = False

    p.patches('xs','ys', source = geosource, line_color = 'black', line_width = 0.25,
              fill_alpha = 1, fill_color = {'field' : str(column), 'transform' : color_mapper})

    p.add_layout(color_bar, 'below')
    return p   
from bokeh.palettes import Turbo256 
from bokeh.models import ColumnDataSource
from bokeh.transform import factor_cmap
from bokeh.palettes import magma,viridis,cividis, inferno

def eda_bokeh_horiz_bar_ranked(df, column_target, title = '', int_top = 3, second_target = 'state'):
    """
    Generate Bokeh Plot ranking top fists and last value:
        df: data_frame
        column_targe: a column of df inputed
        title: title of plot
        int_top: number of the tops
        column: column of df_in to be placed values in geoplot
        second_targe = 'state'
    """
    ranked = df.sort_values(by=column_target).reset_index(drop = True)
    top_int = int_top
    top = ranked[:top_int].append(ranked[-top_int:])
    top.index = top.index + 1
    source = ColumnDataSource(data=top)
    list_second_target = source.data[second_target].tolist()
    index_label = list_second_target[::-1] # reverse order label

    p = figure(plot_width=500, plot_height=400, y_range=index_label, 
                toolbar_location=None, title=title)   

    p.hbar(y=second_target, right=column_target, source=source, height=0.85, line_color="#000000",
          fill_color=factor_cmap(second_target, palette=inferno(16)[::-1], factors=list_second_target))
    p.x_range.start = 0  # start value of the x-axis

    p.xaxis.axis_label = "value of '" + column_target + "'"

    hover = HoverTool()  # initiate hover tool
    hover.tooltips = [("Value","@{" + column_target + "}{%.6f}" ),   
                       ("Ranking","@index°")]
    hover.formatters={'@{'+column_target+'}' : 'printf'}

    hover.mode = 'hline' # set the mode of the hover tool
    p.add_tools(hover)   # add the hover tooltip to the plot

    return p # show in notebook
def eda_foward_2_plots(my_df, primary_column, target_column, first_title, second_title, int_top = 8, location_column = 'District'):
    """
    Execute and show all together:
    @ primary_columns must to be a float to join to make a GeoSource
    generate_GeoJSONSource_to_districts()
    eda_seoul_districts_geo_plot()
    eda_bokeh_horiz_bar_ranked()
    """
    my_df = my_df.rename({primary_column: target_column}, axis = 1)

    geo_source = generate_GeoJSONSource_to_districts(my_df, target_column)

    geo = eda_seoul_districts_geo_plot(geo_source, my_df, first_title,
                                       target_column, location_column, palette = inferno(32))

    rank = eda_bokeh_horiz_bar_ranked(my_df, target_column, second_title,
                                      int_top = int_top, second_target = location_column)

    show( row( geo, rank ))
def eda_categ_feat_desc_plot(series_categorical, title = ""):
    """Generate 2 plots: barplot with quantity and pieplot with percentage. 
       @series_categorical: categorical series
       @title: optional
    """
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    
    fig, ax = plt.subplots(figsize = (12,4), ncols=2, nrows=1) # figsize = (width, height)
    if(title != ""):
        fig.suptitle(title, fontsize=18)
        fig.subplots_adjust(top=0.8)

    s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])
    for index, row in val_concat.iterrows():
        s.text(row.name, row['quantity'], row['quantity'], color='black', ha="center")

    s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),
                             labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],
                             title="Percentage Plot")

    ax[1].set_ylabel('')
    ax[0].set_title('Quantity Plot')

    plt.show()
def measurement_evaluator(value, column):
    if(pd.isnull(value) or value < 0):
        return 'Error'
    elif(value <= level_dangerous[column][0]):
        return 'Good'
    elif(value <= level_dangerous[column][1]):
        return 'Normal'
    elif(value <= level_dangerous[column][2]):
        return 'Bad'
    else:
        return 'Very bad'
    return value
def generate_level_danger_gas_series(my_df, column):
    series = my_df[column].map(lambda x: measurement_evaluator(x, column))
    return series
# Generate many boxplots to each pollutant data

fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(ncols=3, nrows=2, figsize=(15, 7), sharex=False)

map_feat_ax = {'SO2': ax1, 'NO2': ax2, 'O3': ax3, 'CO': ax4, 'PM10': ax5, 'PM2.5': ax6}

for key, value in map_feat_ax.items():
    sns.boxplot(x=df[key], ax=value)
    
fig.suptitle('Distribution to each polluant', fontsize=18)
    
plt.show()
# Generate DataFrame with each 'describe' to each pollutant data

gas_list = list(map_feat_ax.keys())

list_describes = []
for f in gas_list:
    list_describes.append(df[f].describe())

df_describe_gas = pd.concat(list_describes, axis = 1)
df_describe_gas   
# Before

df.shape
condicional = df[gas_list] > 0.0

df = df[condicional.all(axis=1)]
df.shape
# Before

df.shape
# Remove rows where one of gas data has z_score bigger than threshold

# link to talbe of z score associated to percentage of distriution: associated
## https://www.math.arizona.edu/~rsims/ma464/standardnormaltable.pdf
## Examples
# 3 = .99865 = 99,85%
# 2 = .97725 = 97,72%

from scipy import stats

z = np.abs(stats.zscore(df[gas_list]))

threshold = 2

df = df[(z < 2).all(axis=1)]
df.shape
# Finally, we have ...

f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(ncols=3, nrows=2, figsize=(15, 7), sharex=False)

map_feat_ax = {'SO2': ax1, 'NO2': ax2, 'O3': ax3, 'CO': ax4, 'PM10': ax5, 'PM2.5': ax6}

for key, value in map_feat_ax.items():
    sns.boxplot(x=df[key], ax=value)
    
fig.suptitle('Distribution of polluants after remove outiliers', fontsize=18)
    
plt.show()
# Show describe() to each pollutant

gas_list = list(map_feat_ax.keys())

list_describes = []
for f in gas_list:
    list_describes.append(df[f].describe())

df_describe_gas1 = pd.concat(list_describes, axis = 1)
df_describe_gas1  
# Generate columns to year, mont and weekday

df['Year']  = pd.DatetimeIndex(df['Date']).year
df['Month']  = pd.DatetimeIndex(df['Date']).month
df['Weekday'] = pd.DatetimeIndex(df['Date']).strftime("%A")
df['Semester'] = ((pd.DatetimeIndex(df['Date']).month.astype(int) - 1) // 6) + 1

# show new columns
df.head(3)
fig, ax = plt.subplots(figsize=(15,5))

df.groupby(['Year','Semester', 'District']).mean()['SO2'].unstack().plot(ax=ax)
ax.set_ylabel("SO2")
ax.set_title("S02 Evolution")
ax.set_title("Evolution of SO2")
plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=2)
plt.grid(True)
district_list = df['District'].unique().tolist()

turbo_pallete = Turbo256[0:256:int(256/len(district_list) )][::-1]
gas_type = 'SO2'
df_SO2_semester = df.groupby(['Year','Semester', 'District']).mean()[gas_type].reset_index()

avg_semester = {}
for d in district_list:
    avg_semester[d] = np.array(df_SO2_semester.query('District == "' + d + '"')[gas_type])
from bokeh.palettes import Turbo256
from bokeh.models import Legend

x_axis = np.array([2017,2017.5,2018,2018.5,2019,2019.5])

fig = figure(title="Interactive overall evolution of the SO in Bokeh", plot_width=1000, plot_height=700, x_axis_type="linear")

count = 0
for d in district_list:
    line = fig.line(x_axis, avg_semester[d], legend_label=d, color=turbo_pallete[count] ,line_width=3)
    fig.circle(x_axis, avg_semester[d], legend_label=d, color=turbo_pallete[count], fill_color='white', size=7)
    count += 1
# plot title
fig.legend.title = 'Gas'
# Relocate Legend
fig.legend.location = 'bottom_left'
# Click to hide/show lines
fig.legend.click_policy = 'hide'
# Add Hover
fig.add_tools(HoverTool(tooltips=[('SO2', '@y{%.5f}')], formatters={'@y' : 'printf'} ))

show(fig)
primary_column = 'SO2'
target_column = 'total_average_SO2'

df1 = df.groupby(['District']).mean()[primary_column].reset_index()

eda_foward_2_plots(df1, primary_column, target_column, "SO total average per district", "The first and last 8 on average for SO")
primary_column = 'CO'
target_column = 'total_average_CO'

df1 = df.groupby(['District']).mean()[primary_column].reset_index()

eda_foward_2_plots(df1, primary_column, target_column, "CO total average per district", "The first and last 8 on average for CO")
primary_column = 'O3'
target_column = 'total_average_O3'

df1 = df.groupby(['District']).mean()[primary_column].reset_index()

eda_foward_2_plots(df1, primary_column, target_column,
                   "O3 total average per district", "The first and last 8 on average for O3")
primary_column = 'PM10'
target_column = 'total_average_PM10'

df1 = df.groupby(['District']).mean()[primary_column].reset_index()

eda_foward_2_plots(df1, primary_column, target_column,
                   "PM10 total average per district", "The first and last 8 on average for PM10")
primary_column = 'PM2.5'
target_column = 'total_average_PM2.5'

df1 = df.groupby(['District']).mean()[primary_column].reset_index()

eda_foward_2_plots(df1, primary_column, target_column,
                   "PM2.5 total average per district", "The first and last 8 on average for PM2.5")
primary_column = 'NO2'
target_column = 'total_average_NO2'

df1 = df.groupby(['District']).mean()[primary_column].reset_index()

eda_foward_2_plots(df1, primary_column, target_column,
                   "NO2 total average per district", "The first and last 8 on average for NO2")
# Create `df_measures`

# Filter only measures columns
l_cols = df.columns.tolist()[1:8]
df_measures = df[l_cols]

# Generate level dangerous to each gas
l_level = [x+'_Level' for x in df_measures.columns.tolist()[1:] ]
for l in l_level:
    df_measures[l] = generate_level_danger_gas_series(df_measures, l[:-6])
df_measures = df_measures.reset_index().drop('index', axis = 1)

# output
df_measures.head()
eda_categ_feat_desc_plot(df_measures['SO2_Level'], 'SO2 level in all gas measurements')
eda_categ_feat_desc_plot(df_measures['NO2_Level'], "NO2 level in all gas measurements")
eda_categ_feat_desc_plot(df_measures['O3_Level'], 'O3 level in all gas measurements')
eda_categ_feat_desc_plot(df_measures['PM10_Level'], 'PM10 level in all gas measurements')
eda_categ_feat_desc_plot(df_measures['PM2.5_Level'], "PM2.5 level in all gas measurements")
# Getting Data Frame with rows where exist 'Bad' or 'Very Bad' on measures gas level

gas_level_list = list(df_measures.columns)[7:]

cond = df_measures.isin(['Bad', 'Very Bad'])
cond = cond[ cond[gas_level_list] == True].dropna(how="all")
list_remove = list(cond.index)
df_bad = df_measures.iloc[list_remove]
df_bad.head()
my_df_bad = df_bad['District'].value_counts().reset_index().rename(columns={'index': 'District', 'District': 'count_bad_measures'})
my_df_bad['count_bad_measures'] = my_df_bad['count_bad_measures'].astype(float)

primary_column = 'count_bad_measures'
target_column = 'count_bad_measures'

eda_foward_2_plots(my_df_bad, primary_column, target_column,
                   "Counting 'Bad' and 'Very Bad' values for all pollutants", "The first and last 8 Counting 'Bad' and 'Very Bad' ")
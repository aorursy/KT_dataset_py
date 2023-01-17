import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import random



import plotly.express as px 

from plotly.offline import init_notebook_mode, iplot, plot

import plotly as py

init_notebook_mode(connected=True)

import plotly.graph_objs as go



import warnings

warnings.filterwarnings("ignore")



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Configs

pd.options.display.float_format = '{:,.4f}'.format

sns.set(style="whitegrid")

plt.style.use('seaborn')

seed = 42

np.random.seed(seed)

random.seed(seed)
file_path = '/kaggle/input/co2-ghg-emissionsdata/co2_emission.csv'

df = pd.read_csv(file_path)

print("DataSet = {:,d} rows and {} columns".format(df.shape[0], df.shape[1]))



print("\nAll Columns:\n=>", df.columns.tolist())



quantitative = [f for f in df.columns if df.dtypes[f] != 'object']

qualitative = [f for f in df.columns if df.dtypes[f] == 'object']



print("\nStrings Variables:\n=>", qualitative,

      "\n\nNumerics Variables:\n=>", quantitative)



df.head()
def eda_categ_feat_desc_plot(series_categorical, title = "", fix_labels=False):

    series_name = series_categorical.name

    val_counts = series_categorical.value_counts()

    val_counts.name = 'quantity'

    val_percentage = series_categorical.value_counts(normalize=True)

    val_percentage.name = "percentage"

    val_concat = pd.concat([val_counts, val_percentage], axis = 1)

    val_concat.reset_index(level=0, inplace=True)

    val_concat = val_concat.rename( columns = {'index': series_name} )

    

    fig, ax = plt.subplots(figsize = (12,4), ncols=2, nrows=1)

    if(title != ""):

        fig.suptitle(title, fontsize=18)

        fig.subplots_adjust(top=0.8)



    s = sns.barplot(x=series_name, y='quantity', data=val_concat, ax=ax[0])

    if(fix_labels):

        val_concat = val_concat.sort_values(series_name).reset_index()

    

    for index, row in val_concat.iterrows():

        s.text(row.name, row['quantity'], '{:,d}'.format(row['quantity']), color='black', ha="center")



    s2 = val_concat.plot.pie(y='percentage', autopct=lambda value: '{:.2f}%'.format(value),

                             labels=val_concat[series_name].tolist(), legend=None, ax=ax[1],

                             title="Percentage Plot")



    ax[1].set_ylabel('')

    ax[0].set_title('Quantity Plot')



    plt.show()
import squarify 

import matplotlib



def tree_map_cat_feat(dfr, column, title='', threshold=1, figsize=(18, 6), alpha=.7):

    plt.figure(figsize=figsize)

    df_series = dfr[column].value_counts()

    df_mins = df_series[ df_series <= threshold ].sum()

    df_series = df_series[ df_series > threshold ]

    df_series['Others'] = df_mins

    percentages = df_series / df_series.sum()

    alist, mini, maxi = [], min(df_series), max(df_series)

    for i in range(len(df_series)):

        alist.append( df_series.index[i] + '\n{:.2%}'.format(percentages[i]) )

    cmap = matplotlib.cm.viridis

    norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)

    colors = [cmap(norm(i)) for i in df_series]

    squarify.plot(sizes=df_series.values, label=alist, color=colors, alpha=alpha)

    plt.axis('off')

    plt.title(title)

    plt.show()
def eda_numerical_feat(series, title="", with_label=True, number_format="", show_describe=False, size_labels=10):

    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)

    if(show_describe):

        print(series.describe())

    if(title != ""):

        f.suptitle(title, fontsize=18)

    sns.distplot(series, ax=ax1)

    sns.boxplot(series, ax=ax2)

    if(with_label):

        describe = series.describe()

        labels = { 'min': describe.loc['min'], 'max': describe.loc['max'], 

              'Q1': describe.loc['25%'], 'Q2': describe.loc['50%'],

              'Q3': describe.loc['75%']}

        if(number_format != ""):

            for k, v in labels.items():

                ax2.text(v, 0.3, k + "\n" + number_format.format(v), ha='center', va='center', fontweight='bold',

                         size=size_labels, color='white', bbox=dict(facecolor='#445A64'))

        else:

            for k, v in labels.items():

                ax2.text(v, 0.3, k + "\n" + str(v), ha='center', va='center', fontweight='bold',

                     size=size_labels, color='white', bbox=dict(facecolor='#445A64'))

    plt.show()
def eda_cat_top_slice_count(s, start=1, end=None, rotate=0):

    # @rotate: 45/80; 

    column, start, threshold = s.name, start - 1, 30

    s = df[column].value_counts()

    lenght = len(s)

    if(end is None):

        end = lenght if lenght <= threshold else threshold

    s = s.reset_index()[start:end]

    s = s.rename(columns = {column: 'count'}).rename(columns = {'index': column,})

    fig, ax = plt.subplots(figsize = (12,4))

    barplot = sns.barplot(x=s[column], y=s['count'], ax=ax)

    # sort by name

    s = s.sort_values(column).reset_index()

    for index, row in s.iterrows():

        barplot.text(row.name, row['count'], '{:,d}'.format(row['count']), color='black', ha="center")

    ax.set_title('Quantity Plot to {}. Top {}°-{}°'.format(column, start+1, end))

    plt.xticks(rotation=rotate)

    plt.show()
from bokeh.io import show

from bokeh.plotting import figure

from bokeh.models import LinearColorMapper, HoverTool, ColorBar

from bokeh.palettes import magma,viridis,cividis, inferno

from bokeh.models import WheelZoomTool, BoxZoomTool, ResetTool



def eda_us_states_geo_plot(geosource, df_in, title, column, state_column, low = -1, high = -1, palette = -1, plot_width=500):

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

        palette = inferno(24)

        

    palette = palette[::-1]

    color_mapper = LinearColorMapper(palette = palette, low = low, high = high)

    

    hover = HoverTool(tooltips = [ ('State','@{'+'name'+'}'), (column, '@{'+column+'}{%.2f}')],

                  formatters={'@{'+column+'}' : 'printf'})



    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8, width = 450, height = 20, 

                         border_line_color=None, location = (0,0),  orientation = 'horizontal')



    p = figure(title = title, plot_height = 400, plot_width = plot_width, tools = [hover])



    p.xgrid.grid_line_color = None

    p.ygrid.grid_line_color = None

    p.xaxis.visible = False

    p.yaxis.visible = False



    p.patches('xs','ys', source = geosource, line_color = 'black', line_width = 0.25,

              fill_alpha = 1, fill_color = {'field' : str(column), 'transform' : color_mapper})



    p.add_layout(color_bar, 'below')

    p.add_tools(WheelZoomTool())

    p.add_tools(ResetTool())

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

    ranked = df.sort_values(by=column_target, ascending=False).reset_index(drop = True)

    ranked = ranked.dropna()

    top_int = int_top

    # top = ranked[:top_int].append(ranked[-top_int:]) # Bottom an Top

    top = ranked[:top_int+top_int]# only Top

    top.index = top.index + 1

    source = ColumnDataSource(data=top)

    list_second_target = source.data[second_target].tolist()

    index_label = list_second_target[::-1] # reverse order label



    p = figure(plot_width=500, plot_height=400, y_range=index_label, 

                toolbar_location=None, title=title)   



    p.hbar(y=second_target, right=column_target, source=source, height=0.85, line_color="#000000",

          fill_color=factor_cmap(second_target, palette=inferno(24), factors=list_second_target))

    p.x_range.start = 0  # start value of the x-axis



    p.xaxis.axis_label = "value of '" + column_target + "'"



    hover = HoverTool()  # initiate hover tool

    hover.tooltips = [("Value","@{" + column_target + "}{%.2f}" ),("Ranking","@index°")]

    hover.formatters={'@{'+column_target+'}' : 'printf'}



    hover.mode = 'hline' # set the mode of the hover tool

    p.add_tools(hover)   # add the hover tooltip to the plot



    return p # show in notebook



def eda_geplot_state_rank_plot(my_df, primary_column, target_column, first_title, second_title, int_top = 10, location_column = 'state', ):

    """

    Execute and show all together:

    @ primary_columns must to be a float to join to make a GeoSource

    generate_GeoJSONSource_to_districts()

    eda_seoul_districts_geo_plot()

    eda_bokeh_horiz_bar_ranked()

    """

    my_df = my_df.rename({primary_column: target_column}, axis = 1)



    geo_source = generate_GeoJSONSource(my_df)



    geo = eda_us_states_geo_plot(geo_source, my_df, first_title,

                                       target_column, location_column, palette = inferno(32))





    # rank 8 bottom and Up

    rank = eda_bokeh_horiz_bar_ranked(my_df, target_column, second_title,

                                      int_top = int_top, second_target = 'entity')



    show( row( geo, rank ))
# new functions to geojson of each continent



def create_geojson_to_one_map(my_df, my_geojson):

    geo_source_merged_re = my_geojson.merge(my_df, left_on = 'name', right_on = 'entity', how='left')

    eight_columns = geo_source_merged_re.columns[7]

    return GeoJSONDataSource( geojson = geo_source_merged_re.to_json()), geo_source_merged_re[['entity', 'year', eight_columns]]



def geoplot_to_map(my_df, my_geojson, primary_column, target_column, first_title, second_title, int_top = 10, location_column = 'state', plot_width=500):

    my_df = my_df.rename({primary_column: target_column}, axis = 1)



    geo_source, my_df = create_geojson_to_one_map(my_df, my_geojson)



    geo = eda_us_states_geo_plot(geo_source, my_df, first_title,target_column,

                                 location_column, palette = inferno(32), plot_width = plot_width)



    rank = eda_bokeh_horiz_bar_ranked(my_df, target_column, second_title,int_top = int_top,

                                      second_target = 'entity')



    show( row( geo, rank ))

    
def filter_by_merging_geojson(my_df, my_geojson):

    return my_geojson.merge(my_df, left_on = 'name', right_on = 'entity', how='left')['entity'].dropna().unique().tolist()
sns.heatmap(df.isnull(), cbar=False, yticklabels=False)
df.duplicated().sum() # no duplicate rows
df.columns = ['entity', 'code', 'year', 'CO2'] # lower and simplify column names

df = df.drop(['code'], axis=1) # Remove code
# simplify and replace continents names

continent_replace = {'Americas (other)': 'Americas', 'Asia and Pacific (other)': 'Asia and Pacific',

                     'EU-28':'European Union', 'Europe (other)': 'Europe'}

continents_list = ['Americas', 'Middle East', 'Asia and Pacific', 'Europe', 'Africa']

df['entity'] = df['entity'].replace(continent_replace)



# Replacement of some country names to match GeoJSON names

contries_replace = {'Democratic Republic of Republic of the Congo': 'Democratic Republic of the Congo',

                    'Republic of the Congo': 'Republic of Congo', 'Cote d\'Ivoire':'Ivory Coast',

                    'Faeroe Islands': 'Faroe Islands', 'Guinea-Bissau': 'Guinea Bissau'}

df['entity'] = df['entity'].replace(contries_replace)
entities = df['entity'].unique().tolist()

print('There are {} diferents entities:\n'.format(len(entities)))

print(entities)
tree_map_cat_feat(df, 'entity', 'count by entity', 50, figsize=(20, 10))
# Year

years = np.sort(df['year'].unique())



is_full = True

for i in range(1751,2017):

    if(i not in years):

        print('not found', i)

        is_full = False

        

if(is_full):

    print('Tem ao menos um dado para cada ano entre 1751 e 2017')

eda_cat_top_slice_count(df['year'], start=1, end=35, rotate=60)
from bokeh.io import output_notebook, show

from bokeh.plotting import figure

from bokeh.models.tools import HoverTool

from bokeh.models import GeoJSONDataSource

from bokeh.layouts import row

output_notebook()



import geopandas as gpd



# import geojson

geojson = gpd.read_file('../input/world-map-eckert3/world-eckert3.geo.json')



# delete useless columns

list_to_delete = ['id', 'hc-group', 'hc-middle-x', 'hc-middle-y', 'hc-key', 'hc-a2',

                  'labelrank', 'woe-id', 'labelrank', 'iso-a3', 'iso-a2', 'woe-id']



# replace some name to match with df

replace_dict = {'United States of America': 'United States', 'United Republic of Tanzania': 'Tanzania',

                'Republic of Serbia': 'Serbia', 'The Bahamas':'Bahamas'}



geojson = geojson.drop(list_to_delete, axis = 1).dropna().replace(replace_dict)



def generate_GeoJSONSource(my_df):

    global geojson

    geo_source_merged = geojson.merge(my_df, left_on = 'name', right_on = 'entity')

    return GeoJSONDataSource( geojson = geo_source_merged.to_json())



# show

geojson.head(3)
print(list(np.sort(geojson.name.unique()))) # Countries in GeoJSON
df1 = df.groupby(['entity']).count()['year'].reset_index()

geo_source_merged = geojson.merge(df1, left_on = 'name', right_on = 'entity', how='right')

geo_source_merged['name'] = geo_source_merged['name'].fillna('mising')

remove_list = geo_source_merged.query('name == "mising"')['entity'].tolist()

print('Countries that did not match df["entity"], that is, are in DF but not in GeoJSON: \n')

print(remove_list)
primary_column = 'CO2'

target_column = 'sum_co2'



df1 = df.groupby(['entity']).sum()[primary_column].reset_index()

df1 = df1.drop(df1[df1['entity'].isin(remove_list)].index, axis=0) # remove_list: removes mismatched data



eda_geplot_state_rank_plot(df1, primary_column, target_column,

                           "Sum of Emission Of CO2", "The Top 20 Countries on Sum of Emission of CO2")
primary_column = 'CO2'

target_column = 'CO2/2017'



df1 = df.query('year == 2017')

df1 = df1.drop(df1[df1['entity'].isin(remove_list)].index, axis=0)  # remove_list: removes mismatched data



eda_geplot_state_rank_plot(df1, primary_column, target_column,

                           "Emission in last Year 2017", "The Top 20 Countries on emission of CO2 on 2017")
# import europe_geojson

europe_geojson = gpd.read_file('../input/global-map-geojson/europe.geo.json')

europe_geojson = europe_geojson.drop(list_to_delete + ['country-abbrev'], axis = 1).dropna().replace(replace_dict)



primary_column = 'CO2'

target_column = 'CO2/2017'



df1 = df.query('year == 2017')

df1 = df1.drop(df1[df1['entity'].isin(remove_list)].index, axis=0)  # remove_list: removes mismatched data



geoplot_to_map(df1, europe_geojson, primary_column, target_column,

               "Emission of CO2 in Europe at 2017", "The Top 20 Countries on Emission of CO2 in Europe on 2017", plot_width=400)
countries = filter_by_merging_geojson(df, europe_geojson)



df1 = df[ df['entity'].isin(countries) ]



fig = px.line(df1, x="year", y="CO2", color='entity')

fig.update_layout(title='Evolution of CO2 emissions in Europe')

fig.show()
# import africa geojson

africa_geojson = gpd.read_file('../input/global-map-geojson/africa.geo.json')

africa_geojson = africa_geojson.drop(list_to_delete + ['country-abbrev'], axis = 1).dropna().replace(replace_dict)



primary_column = 'CO2'

target_column = 'CO2/2017'



df1 = df.query('year == 2017')

df1 = df1.drop(df1[df1['entity'].isin(remove_list + ['France'])].index, axis=0) # remove_list: removes mismatched data



geoplot_to_map(df1, africa_geojson, primary_column, target_column,

               "Emission of CO2 in Africa at 2017", "The Top 20 Countries on Emission of CO2 in Africa on 2017", plot_width=400)
countries = filter_by_merging_geojson(df, africa_geojson)



df1 = df[ df['entity'].isin(countries) ]

df1 = df1.drop( df1[df1['entity'].isin(['France'])].index, axis=0)





fig = px.line(df1, x="year", y="CO2", color='entity')

fig.update_layout(title='Evolution of CO2 emissions in Africa')

fig.show()
# import south_america geojson

south_america_geojson = gpd.read_file('../input/global-map-geojson/south-america.geo.json')

south_america_geojson = south_america_geojson.drop(list_to_delete + ['country-abbrev'], axis = 1).dropna().replace(replace_dict)



primary_column = 'CO2'

target_column = 'CO2_2017'



df1 = df.query('year == 2017')

df1 = df1.drop(df1[df1['entity'].isin(remove_list + ['France', 'United Kingdom'])].index, axis=0) # remove_list: removes mismatched data



geoplot_to_map(df1, south_america_geojson, primary_column, target_column,

               "Emission of CO2 in South America at 2017", "The Top 20 Countries on Emission of CO2 in South America on 2017", plot_width=300)
countries = filter_by_merging_geojson(df, south_america_geojson)



df1 = df[ df['entity'].isin(countries) ]

df1 = df1.drop( df1[df1['entity'].isin(['United Kingdom'])].index, axis=0)



fig = px.line(df1, x="year", y="CO2", color='entity')

fig.update_layout(title='Evolution of CO2 emissions in South America')

fig.show()
# import geojson

asia_geojson = gpd.read_file('../input/global-map-geojson/asia.geo.json')

asia_geojson = asia_geojson.drop(list_to_delete + ['country-abbrev'], axis = 1).dropna().replace(replace_dict)



primary_column = 'CO2'

target_column = 'CO2/2017'



df1 = df.query('year == 2017')

df1 = df1.drop(df1[df1['entity'].isin(remove_list + ['France', 'United Kingdom'])].index, axis=0) 



geoplot_to_map(df1, asia_geojson, primary_column, target_column,

               "Emission of CO2 in Asia at 2017", "The Top 20 Countries on Emission of CO2 in Asia on 2017", plot_width=450)
countries = filter_by_merging_geojson(df, asia_geojson)



df1 = df[ df['entity'].isin(countries) ]



fig = px.line(df1, x="year", y="CO2", color='entity')

fig.update_layout(title='Evolution of CO2 emissions in Asia')

fig.show()
# import geojson

north_america_geojson = gpd.read_file('../input/global-map-geojson/north-america.geo.json')

north_america_geojson = north_america_geojson.drop(list_to_delete + ['country-abbrev'], axis = 1).dropna().replace(replace_dict)



primary_column = 'CO2'

target_column = 'CO2/2017'



df1 = df.query('year == 2017')

df1 = df1.drop(df1[df1['entity'].isin(remove_list + ['France', 'United Kingdom'])].index, axis=0) 



geoplot_to_map(df1, north_america_geojson, primary_column, target_column,

               "Emission of CO2 in North America at 2017", "The Top 20 Countries on Emission of CO2 in North America on 2017", plot_width=400)
countries = filter_by_merging_geojson(df, north_america_geojson)



df1 = df[ df['entity'].isin(countries) ]



fig = px.line(df1, x="year", y="CO2", color='entity')

fig.update_layout(title='Evolution of CO2 emissions in North America')

fig.show()
df1 = df[ df['entity'].isin(continents_list + ['World'] ) ] 

fig = px.line(df1, x="year", y="CO2", color='entity')

fig.update_layout(title='Evolution Emission Of CO2 in World and Continents')

fig.show()
import plotly.express as px



countries = ['United States', 'China', 'Russia', 'United Kingdom', 'Germany', 'France', 'Japan', 'Canada', 'Brazil',

             'South Africa', 'India', 'Mexico', 'Australia', 'Iran', 'Saudi Arabia']

df1 = df[ df['entity'].isin(countries) ]



fig = px.line(df1, x="year", y="CO2", color='entity')

fig.update_layout(title='Evolution of CO2 emissions in Big Contries')

fig.show()
df_countries = pd.read_csv('../input/countries-of-the-world/countries of the world.csv')

df_countries['Country'] = df_countries['Country'].apply(lambda x: x.strip())

dict_replace_countries = { 'Bahamas, The': 'Bahamas', 'Congo, Dem. Rep.': 'Democratic Republic of the Congo',

                          'Congo, Repub. of the': 'Republic of Congo', 'Korea, South':'South Korea',

                          'Korea, North':'North Korea', 'Central African Rep.': 'Central African Republic',

                          "Cote d'Ivoire": 'Ivory Coast', 'Guinea-Bissau': 'Guinea Bissau', 'Gambia, The': 'Gambia',

                          'Western Sahara': 'Western Sahara'}



df_countries['Country'] = df_countries['Country'].replace(dict_replace_countries)



df_countries['Agriculture'] = df_countries['Agriculture'].fillna(0).astype('object').apply(lambda x: x if x == 0 else x.replace(',','.')).astype('float64')

df_countries['Industry'] = df_countries['Industry'].fillna(0).astype('object').apply(lambda x: x if x == 0 else x.replace(',','.')).astype('float64')

df_countries['Service'] = df_countries['Service'].fillna(0).astype('object').apply(lambda x: x if x == 0 else x.replace(',','.')).astype('float64')



df_countries.head(1)
dfc = df.merge(df_countries, left_on='entity', right_on='Country')



dfc['CO2/population'] = dfc['CO2']/dfc['Population']

dfc['CO2/area'] = dfc['CO2']/dfc['Area (sq. mi.)']

dfc['CO2/GDP'] = dfc['CO2']/dfc['GDP ($ per capita)']
primary_column = 'CO2/GDP'

target_column = 'CO2/GDP at 2017'



df1 = dfc.query('year == 2017')

df1 = df1.drop(df1[df1['entity'].isin(remove_list)].index, axis=0)  # remove_list: removes mismatched data



eda_geplot_state_rank_plot(df1, primary_column, target_column,

                           "Emission Of CO2/GPD on 2017", "The Top 20 Countries on Emission Of CO2/GPD on 2017")
primary_column = 'CO2/area'

target_column = 'CO2/area at 2017'



df1 = dfc.query('year == 2017')

df1 = df1.drop(df1[df1['entity'].isin(remove_list)].index, axis=0)  # remove_list: removes mismatched data



eda_geplot_state_rank_plot(df1, primary_column, target_column,

                           "Emission of CO2/Area on 2017", "The Top 20 Countries on CO2/Area on 2017")
primary_column = 'CO2/population'

target_column = 'CO2/population at 2017'



df1 = dfc.query('year == 2017')

df1 = df1.drop(df1[df1['entity'].isin(remove_list)].index, axis=0)  # remove_list: removes mismatched data



eda_geplot_state_rank_plot(df1, primary_column, target_column,

                           "Emission of CO2/Population 2017", "The Top 20 Countries on CO2/Population at 2017")
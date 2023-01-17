import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

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
# Plots Format
pd.options.display.float_format = '{:.4f}'.format
sns.set(style="whitegrid")
# Import CSV
df = pd.read_csv("/kaggle/input/us-police-shootings/shootings.csv")
df.head()
df.info()
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
def eda_categ_feat_desc_df(series_categorical):
    """Generate DataFrame with quantity and percentage of categorical series
    @series_categorical = categorical series
    """
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    return val_concat
def eda_numerical_feat(series, title=""):
    """
    Generate series.describe(), bosplot and displot to a series
    """
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5), sharex=False)
    print(series.describe())
    if(title != ""):
        f.suptitle(title, fontsize=18)
    sns.distplot(series, ax=ax1)
    sns.boxplot(series, ax=ax2)
    plt.show()
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, HoverTool, ColorBar
from bokeh.palettes import magma,viridis,cividis, inferno

def eda_us_states_geo_plot(geosource, df_in, title, column, state_column, low = -1, high = -1, palette = -1):
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
    
    hover = HoverTool(tooltips = [ ('State','@{'+'name'+'}'), (column, '@{'+column+'}{%.2f}')],
                  formatters={'@{'+column+'}' : 'printf'})

    color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8, width = 450, height = 20, 
                         border_line_color=None, location = (0,0),  orientation = 'horizontal')

    p = figure(title = title, plot_height = 400, plot_width = 500, tools = [hover])

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
    hover.tooltips = [("Value","@{" + column_target + "}{%.2f}" ),   
                       ("Ranking","@index°")]
    hover.formatters={'@{'+column_target+'}' : 'printf'}

    hover.mode = 'hline' # set the mode of the hover tool
    p.add_tools(hover)   # add the hover tooltip to the plot

    return p # show in notebook

def eda_geplot_state_rank_plot(my_df, primary_column, target_column, first_title, second_title, int_top = 8, location_column = 'state'):
    """
    Execute and show all together:
    @ primary_columns must to be a float to join to make a GeoSource
    generate_GeoJSONSource_to_districts()
    eda_seoul_districts_geo_plot()
    eda_bokeh_horiz_bar_ranked()
    """
    my_df = my_df.rename({primary_column: target_column}, axis = 1)

    geo_source = generate_GeoJSONSource_to_states(my_df)

    geo = eda_us_states_geo_plot(geo_source, my_df, first_title,
                                       target_column, location_column, palette = inferno(32))
    
    my_df['name_state'] = my_df['state'].map(lambda x: mapping_abrev_to_name[x])

    rank = eda_bokeh_horiz_bar_ranked(my_df, target_column, second_title,
                                      int_top = int_top, second_target = 'name_state')

    show( row( geo, rank ))
def eda_horiz_plot(df, x, y, title, figsize = (8,5), palette="Blues_d", formating="int"):
    """Using Seaborn, plot horizonal Bar with labels
    !!! Is recomend sort_values(by, ascending) before passing dataframe
    !!! pass few values, not much than 20 is recommended
    """
    f, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=x, y=y, data=df, palette=palette)
    ax.set_title(title)
    for p in ax.patches:
        width = p.get_width()
        if(formating == "int"):
            text = int(width)
        else:
            text = '{.2f}'.format(width)
        ax.text(width + 1, p.get_y() + p.get_height() / 2, text, ha = 'left', va = 'center')
    plt.show()
def eda_numerical_feat(series, title="", with_label=True, number_format=""):
    """ Generate series.describe(), bosplot and displot to a series
    @with_label: show labels in boxplot
    @number_format: 
        integer: 
            '{:d}'.format(42) => '42'
            '{:,d}'.format(12855787591251) => '12,855,787,591,251'
        float:
            '{:.0f}'.format(91.00000) => '91' # no decimal places
            '{:.2f}'.format(42.7668)  => '42.77' # two decimal places and round
            '{:,.4f}'.format(1285591251.78) => '1,285,591,251.7800'
            '{:.2%}'.format(0.09) => '9.00%' # Percentage Format
        string:
            ab = '$ {:,.4f}'.format(651.78) => '$ 651.7800'
    def swap(string, v1, v2):
        return string.replace(v1, "!").replace(v2, v1).replace('!',v2)
    # Using
        swap(ab, ',', '.')
    """
    f, (ax1, ax2) = plt.subplots(ncols=2, figsize=(18, 5), sharex=False)
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
                         size=8, color='white', bbox=dict(facecolor='#445A64'))
        else:
            for k, v in labels.items():
                ax2.text(v, 0.3, k + "\n" + str(v), ha='center', va='center', fontweight='bold',
                     size=8, color='white', bbox=dict(facecolor='#445A64'))
    plt.show()
    
# eda_numerical_feat(df['age'], "Distribution of 'age'", number_format="{:.0f}")
# Check number of rows with null values

df.isnull().sum().max()
df['date'] = pd.to_datetime(df['date'])
df['date'].dtypes
print(list(df.columns))
print("fist record:", df['date'].min(), "| Last Record:",  df['date'].max())
df['date'].describe()
eda_categ_feat_desc_plot(df['manner_of_death'], "Frequency of values for 'manner_of_death'")
eda_categ_feat_desc_df(df['armed'])
eda_numerical_feat(df['age'], "Distribution of 'age'", number_format="{:.0f}")
eda_categ_feat_desc_plot(df['gender'], "Frequency by 'gender'")
eda_categ_feat_desc_plot(df['race'], "Frequency per 'race'")
eda_horiz_plot(eda_categ_feat_desc_df(df['city']).head(10),
               'quantity', 'city', 'Ranking of the 10 cities with the most deaths in USA')
eda_horiz_plot(eda_categ_feat_desc_df(df['state']).head(10),
               'quantity', 'state', 'Ranking of the 10 states with the most deaths in USA')
eda_categ_feat_desc_plot(df['signs_of_mental_illness'], "Count values to 'signs_of_mental_illness'")
eda_categ_feat_desc_plot(df['threat_level'], "Count values to 'threat_level'")
eda_categ_feat_desc_plot(df['flee'], "Count values to 'flee'")
eda_categ_feat_desc_plot(df['body_camera'], "Count values to 'body_camera'")
eda_categ_feat_desc_df(df['arms_category'])
import geopandas as gpd

# import geojson
us_geojson_re = gpd.read_file('../input/us-state-better-view/us-all.geo.json')

# delete useless columns
list_to_delete = ['id', 'hc-group', 'hc-middle-x', 'hc-middle-y', 'hc-key', 'hc-a2',
       'labelrank', 'hasc', 'woe-id', 'state-fips', 'fips', 'country' , 'longitude',
       'woe-name', 'latitude', 'woe-label', 'type' ]
us_geojson_re = us_geojson_re.drop(list_to_delete, axis = 1)

# remove useless trace to separate Alaska and Havai
us_geojson_re = us_geojson_re.dropna()

us_geojson_re.head(1)
def generate_GeoJSONSource_to_states(my_df):
    """Function to generetae GeoJSONSource to generate plots"""
    global us_geojson
    geo_source = us_geojson_re.merge(my_df, left_on = 'postal-code', right_on = 'state')
    return GeoJSONDataSource( geojson = geo_source.to_json())
mapping_us_states = {
 'Alabama' : 'AL',
 'Alaska' : 'AK',
 'Arizona' : 'AZ',
 'Arkansas' : 'AR',
 'California' : 'CA',
 'Colorado' : 'CO',
 'Connecticut' : 'CT',
 'Delaware' : 'DE',
 'Florida' : 'FL',
 'Georgia' : 'GA',
 'Hawaii' : 'HI',
 'Idaho' : 'ID',
 'Illinois' : 'IL',
 'Indiana' : 'IN',
 'Iowa' : 'IA',
 'Kansas' : 'KS',
 'Kentucky' : 'KY',
 'Louisiana' : 'LA',
 'Maine' : 'ME',
 'Maryland' : 'MD',
 'Massachusetts' : 'MA',
 'Michigan' : 'MI',
 'Minnesota' : 'MN',
 'Mississippi' : 'MS',
 'Missouri' : 'MO',
 'Montana' : 'MT',
 'Nebraska' : 'NE',
 'Nevada' : 'NV',
 'New Hampshire' : 'NH',
 'New Jersey' : 'NJ',
 'New Mexico' : 'NM',
 'New York' : 'NY',
 'North Carolina' : 'NC',
 'North Dakota' : 'ND',
 'Ohio' : 'OH',
 'Oklahoma' : 'OK',
 'Oregon' : 'OR',
 'Pennsylvania' : 'PA',
 'Rhode Island' : 'RI',
 'South Carolina' : 'SC',
 'South Dakota' : 'SD',
 'Tennessee' : 'TN',
 'Texas' : 'TX',
 'Utah' : 'UT',
 'Vermont' : 'VT',
 'Virginia' : 'VA',
 'Washington' : 'WA',
 'West Virginia' : 'WV',
 'Wisconsin' : 'WI',
 'Wyoming' : 'WY',
 'District of Columbia': 'DC'
}

# reverse dict: key => values to values => key
mapping_abrev_to_name = {v: k for k, v in mapping_us_states.items()}
df['state_name'] = df['state'].map(lambda state: mapping_abrev_to_name[state])
df['city_state'] = df['city'] + ', ' + df['state']
df['semester'] = ((pd.DatetimeIndex(df['date']).month.astype(int) - 1) // 6) + 1
df['year'] = pd.DatetimeIndex(df['date']).year
df.head(3)
df_teen = df.query('age < 18')
df_teen.shape
eda_horiz_plot(eda_categ_feat_desc_df(df_teen['arms_category']).head(10),
               'quantity', 'arms_category', 'Ranking of the 10 arms category used by for minors')
eda_categ_feat_desc_plot(df_teen['signs_of_mental_illness'], "Count values to 'signs_of_mental_illness' for minors")
eda_categ_feat_desc_plot(df_teen['race'], "Count values to 'race' for minors")
primary_column = 'date'
target_column = 'count_occurrences'

df1 = df.groupby(['state']).count()[primary_column].reset_index()

eda_geplot_state_rank_plot(df1, primary_column, target_column,
                           "Total occurrences by state", "The first and last 8 on occurrences count")
top_number = 10

df_california = df.query("state == 'CA'").groupby(['city']).count()['date'].sort_values(ascending = False).reset_index().rename({'date': 'count'}, axis = 1)

list_cities_CA = list(df_california.head(top_number)['city']) # Guard 'top_number' cities

eda_horiz_plot(df_california.head(top_number), 'count', 'city', 'Ranking of the 10 cities with the most deaths in CA')
print("Count Cities with deaths in CA:", len(df_california), ' cities \n')

eda_numerical_feat(df_california['count'], "Distribution of ocorrencies in California cities", with_label=False)
# Plot evolution of bigges 'top_int' city in california

df_california_re = df.query("state == 'CA'")
df_california_re = df.query("city in " + str(list_cities_CA))

fig, ax = plt.subplots(figsize=(15,5))

df_CA = df_california_re.groupby(['year','semester','city']).count()['id'].unstack()
df_CA.plot(ax=ax)

ax.set_ylabel("Count")
ax.set_title("Evolution of the death count over the years in Top 10 Cities of CA")

x_axis = list(df_CA.index)
ax.set_xticks(range(0,len(x_axis)))
ax.set_xticklabels(x_axis)

plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=2)
plt.grid(True)
plt.show()
top_number = 10

df_tops = df.groupby(['city_state']).count()['date'].sort_values(ascending = False).reset_index().rename({'date': 'count'}, axis = 1)

list_cities_tops = list(df_tops.head(top_number)['city_state']) # guard 'top_number' biggers count number

eda_horiz_plot(df_tops.head(top_number), 'count', 'city_state', 'the top 10 cities with the most deaths in USA')
df_california_re = df.query("city_state in " + str(list_cities_tops))

fig, ax = plt.subplots(figsize=(18,6))

df_CA = df_california_re.groupby(['year','semester','city_state']).count()['id'].unstack()
df_CA.plot(ax=ax)
x_axis = list(df_CA.index)

ax.set_ylabel("Count")
ax.set_title("Evolution Top 10 Biggest Ocorrences cities in USA")

# Set more labels (sem isso so vai mostrar os semestre 1 e nao o 2)
ax.set_xticks(range(0,len(x_axis)))
ax.set_xticklabels(x_axis)

plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=2)
plt.show()
# generate df to plot

df['year'] = pd.DatetimeIndex(df['date']).year
df_re = df.groupby(['year','state_name']).count()['id'].unstack() #unstack is important to make easy make line plot
df_re = df_re.fillna(0) # case dont exist data in this time
# df_re
# get top and bototm 10 in ranking

int_ranking = 5 
df_top_count = df.groupby(['state_name']).count()['id'].reset_index().sort_values(by="id", ascending=False)
top_10_state = list(df_top_count['state_name'][:int_ranking])
bottom_10_state = list(df_top_count['state_name'][-int_ranking:])
print(top_10_state, "\n", bottom_10_state)
from bokeh.palettes import Turbo256
from bokeh.models import Legend

# define pallets of colors
turbo_pallete = Turbo256[0:256:int(256/ (len(top_10_state) + len(bottom_10_state)) )][::-1]

# define x axis
x_axis = np.array([2015,2016,2017,2018,2019,2020])

fig = figure(title="Evolution of the first and last 10 state per total count occorrencies",
             plot_width=800, plot_height=450, x_axis_type="linear")

# is necessary generate a list to put state name in each circle plot with "soruce_re"
def g_list(desc, length):
    l = []
    for i in range(length):
        l.append(desc)
    return l

# Plot Lines
count = 0

for d in top_10_state:
    source_re = ColumnDataSource(dict(x=x_axis, y=np.array(df_re[d]), name=g_list(d, len(x_axis))))
    line = fig.line('x', 'y', legend_label=d, color=turbo_pallete[count] ,line_width=3, source = source_re)
    fig.circle('x', 'y', legend_label=d, color=turbo_pallete[count], fill_color='white', size=7, source = source_re)
    count += 1
    
for d in bottom_10_state:
    source_re = ColumnDataSource(dict(x=x_axis, y=np.array(df_re[d]), name=g_list(d, len(x_axis))))
    line = fig.line('x', 'y', legend_label=d, color=turbo_pallete[count] ,line_width=3, source = source_re)
    fig.circle('x', 'y', legend_label=d, color=turbo_pallete[count], fill_color='white', size=7, source = source_re)
    count += 1

# plot title
fig.legend.title = 'State'
# Relocate Legend
fig.legend.location = 'bottom_left'
# Click to hide/show lines
fig.legend.click_policy = 'hide'
# Add Hover
fig.add_tools(HoverTool(tooltips=[('State', '@name'),('Year', '@x'),('Count', '@y{%.1f}')], formatters={'@y' : 'printf'} ))

show(fig)
primary_column = 'race'
target_column = 'number_of_occurrences'

df_plot = df.query("race == 'Black'").groupby(['state']).count()[primary_column].reset_index()

eda_geplot_state_rank_plot(df_plot, primary_column, target_column,
                           "Total number of Black deaths by state", "The first and last 8 on total death count to black race")
primary_column = 'race'
target_column = 'number_of_occurrences'

df_plot = df.query("race == 'Hispanic'").groupby(['state']).count()[primary_column].reset_index()

eda_geplot_state_rank_plot(df_plot, primary_column, target_column,
                           "Total number of Hispanic deaths by state", "The first and last 8 on total death count to hispanic race")
primary_column = 'race'
target_column = 'number_of_occurrences'

df_plot = df.query("race == 'White'").groupby(['state']).count()[primary_column].reset_index()

eda_geplot_state_rank_plot(df_plot, primary_column, target_column,
                           "Total number of White deaths by state", "The first and last 8 on total death count to white race")
primary_column = 'race'
target_column = 'number_of_occurrences'

df_plot = df.query("race == 'Asian'").groupby(['state']).count()[primary_column].reset_index()

eda_geplot_state_rank_plot(df_plot, primary_column, target_column,
                           "total occurrences by state asian", "The first and last 8 on ocorrencies count:asian")
# import census dataset
df_census = pd.read_csv("../input/us-census-demographic-data/acs2017_county_data.csv")
# delete 'Puerto Rico' cuz dont have data in US shooting dataset
list_indexs = df_census[ df_census['State'] == 'Puerto Rico'].index 
df_census = df_census.drop(list_indexs)
# map states to abreviations
df_census['state'] = df_census['State'].map(lambda x: mapping_us_states[x])
df_census.head(1)
print(list(df_census.columns))
# Feature Engineering to census

df_census["BlackPopulation"] = ((df_census["TotalPop"]/100) * df_census["Black"]).astype('int')
df_census["HispanicPopulation"] = ((df_census["TotalPop"]/100) * df_census["Hispanic"]).astype('int')
df_census["WhitePopulation"] = ((df_census["TotalPop"]/100) * df_census["White"]).astype('int')
df_census["AsianPopulation"] = ((df_census["TotalPop"]/100) * df_census["Asian"]).astype('int')
df_census["NativePopulation"] = ((df_census["TotalPop"]/100) * df_census["Native"]).astype('int')
primary_column = 'TotalPop'
target_column = 'population'

total_pop = df_census.groupby(['state']).sum()['TotalPop'].reset_index().sort_values(by="TotalPop", ascending=False)

eda_geplot_state_rank_plot(total_pop, primary_column, target_column,
                           "Population by 2017 census", "The first and last 8 on ocorrencies count: tasered casses")
primary_column = 'BlackPopulation'
target_column = 'population'

df3 = df_census.groupby(['state']).sum()[primary_column].reset_index().sort_values(by=primary_column, ascending=False)

eda_geplot_state_rank_plot(df3, primary_column, target_column,
                           "Total Black per states by the 2017 census", "The first and last 8 on total black population")
primary_column = 'HispanicPopulation'
target_column = 'population'

df3 = df_census.groupby(['state']).sum()[primary_column].reset_index().sort_values(by=primary_column, ascending=False)

eda_geplot_state_rank_plot(df3, primary_column, target_column,
                           "Total Hispanics per states by the 2017 census", "The first and last 8 on total hispanic population")
primary_column = 'WhitePopulation'
target_column = 'population'

df3 = df_census.groupby(['state']).sum()[primary_column].reset_index().sort_values(by=primary_column, ascending=False)

eda_geplot_state_rank_plot(df3, primary_column, target_column,
                           "Total white per states by the 2017 census", "The first and last 8 on total white population")
primary_column = 'AsianPopulation'
target_column = 'population'

df3 = df_census.groupby(['state']).sum()[primary_column].reset_index().sort_values(by=primary_column, ascending=False)

eda_geplot_state_rank_plot(df3, primary_column, target_column,
                           "Total asian per states by the 2017 census", "The first and last 8 on total asian population")
primary_column = 'NativePopulation'
target_column = 'population'

df3 = df_census.groupby(['state']).sum()[primary_column].reset_index().sort_values(by=primary_column, ascending=False)

eda_geplot_state_rank_plot(df3, primary_column, target_column,
                           "Total native per states by the 2017 census", "The first and last 8 on total native population")
primary_column = 'HispanicPopulation'
target_column = 'population'

df3 = df_census.groupby(['state']).sum()[primary_column].reset_index().sort_values(by=primary_column, ascending=False)
df4 = df3.merge(total_pop, left_on = 'state', right_on = 'state')
df4['percentage_hispanic'] = df4['HispanicPopulation'] / df4['TotalPop']
df4 = df4.sort_values(by='percentage_hispanic', ascending=False)

primary_column = 'percentage_hispanic'
eda_geplot_state_rank_plot(df4, primary_column, target_column,
                           "Percentage of Hispanics by state", "The first and last 8 in % Hipanics by state")
primary_column = 'BlackPopulation'
target_column = 'population'

df3 = df_census.groupby(['state']).sum()[primary_column].reset_index().sort_values(by=primary_column, ascending=False)
df4 = df3.merge(total_pop, left_on = 'state', right_on = 'state')
df4['percentage_black'] = df4['BlackPopulation'] / df4['TotalPop']
df4 = df4.sort_values(by='percentage_black', ascending=False)

primary_column = 'percentage_black'
eda_geplot_state_rank_plot(df4, primary_column, target_column,
                           "Percentage of Black by state", "The first and last 8 in % Black by state")
primary_column = 'WhitePopulation'
target_column = 'population'

df3 = df_census.groupby(['state']).sum()[primary_column].reset_index().sort_values(by=primary_column, ascending=False)
df4 = df3.merge(total_pop, left_on = 'state', right_on = 'state')
df4['percentage_White'] = df4['WhitePopulation'] / df4['TotalPop']
df4 = df4.sort_values(by='percentage_White', ascending=False)

primary_column = 'percentage_White'
eda_geplot_state_rank_plot(df4, primary_column, target_column,
                           "Percentage of White by state", "The first and last 8 in % White by state")
primary_column = 'AsianPopulation'
percentage_column = '%_asian_pop'

df3 = df_census.groupby(['state']).sum()[primary_column].reset_index().sort_values(by=primary_column, ascending=False)
df4 = df3.merge(total_pop, left_on = 'state', right_on = 'state')
df4[percentage_column] = df4[primary_column] / df4['TotalPop']
df4 = df4.sort_values(by=percentage_column, ascending=False)

eda_geplot_state_rank_plot(df4, percentage_column, percentage_column,
                           "Percentage of Asian by state", "The first and last 8 in % Asian by state")
primary_column = 'NativePopulation'
percentage_column = '%_native_pop'

df3 = df_census.groupby(['state']).sum()[primary_column].reset_index().sort_values(by=primary_column, ascending=False)
df4 = df3.merge(total_pop, left_on = 'state', right_on = 'state')
df4[percentage_column] = df4[primary_column] / df4['TotalPop']
df4 = df4.sort_values(by=percentage_column, ascending=False)

eda_geplot_state_rank_plot(df4, percentage_column, percentage_column,
                           "Percentage of Native by state", "The first and last 8 in % Native by state")
df_total_population = df_census.groupby(['state']).sum()['TotalPop'].reset_index().sort_values(by='TotalPop', ascending=False)
df_total_population.head()
column_name = 'death_ratio'

df1 = df.groupby(['state']).count()['id'].reset_index().rename({"id": "count"}, axis=1)
df1 = df1.merge(df_total_population, left_on="state", right_on="state")

df1[column_name] = (df1['count'] * 100000) / df1['TotalPop']
df1.head()

eda_geplot_state_rank_plot(df1, column_name, column_name,
                           "death rate per 100 thousand inhabitants", "The first and last 8 on death rate")
column_name = 'black_death_ratio'

df1 = df.query("race == 'Black'").groupby(['state']).count()['id'].reset_index().rename({"id": "count"}, axis=1)
# df1 = df1.merge(df_total_population, left_on="state", right_on="state")
df1 = df1.merge(df_total_population, left_on="state", right_on="state", how="right")
df1 = df1.fillna(0.0)
df1[column_name] = (df1['count'] * 100000) / df1['TotalPop']
df1.head()

eda_geplot_state_rank_plot(df1, column_name, column_name,
                           "black death rate per 100 thousand inhabitants", "The first and last 8 on black death rate")
column_name = 'hispanic_death_ratio'

df1 = df.query("race == 'Hispanic'").groupby(['state']).count()['id'].reset_index().rename({"id": "count"}, axis=1)
# df1 = df1.merge(df_total_population, left_on="state", right_on="state")
df1 = df1.merge(df_total_population, left_on="state", right_on="state", how="right")
df1 = df1.fillna(0.0)
df1[column_name] = (df1['count'] * 100000) / df1['TotalPop']

eda_geplot_state_rank_plot(df1, column_name, column_name,
                           "hispanic death rate per 100 thousand inhabitants", "The first and last 8 on hispanic death rate")
column_name = 'white_death_ratio'

df1 = df.query("race == 'White'").groupby(['state']).count()['id'].reset_index().rename({"id": "count"}, axis=1)
df1 = df1.merge(df_total_population, left_on="state", right_on="state")

df1[column_name] = (df1['count'] * 100000) / df1['TotalPop']

eda_geplot_state_rank_plot(df1, column_name, column_name,
                           "white death rate per 100 thousand inhabitants", "The first and last 8 on white death rate")
column_name = 'black_death_ratio'

df1 = df.query("race == 'Black'").query('year == 2020').groupby(['state']).count()['id'].reset_index().rename({"id": "count"}, axis=1)
df1 = df1.merge(df_total_population, left_on="state", right_on="state", how="right")
df1 = df1.fillna(0.0)
df1[column_name] = (df1['count'] * 100000) / df1['TotalPop']
df1.head()

eda_geplot_state_rank_plot(df1, column_name, column_name,
                           "black death rate per 100 thousand inhabitants in 2020", "The first and last 8 on black death rate in 2020")
df1 = df.query("race == 'Black'").query('state == "MN"')
df1 = df1.sort_values('date', ascending=False)
df1.head(1) # Max date to Black case in Minnesota . Dont have cases REGISTED in 2020
df1 = df.query("race == 'Black'").query('state == "MN"')
df1 = df1.groupby(['year','semester','state']).sum()['id'].reset_index().rename({"id": "count"}, axis=1)
df1 = df1.merge(df_total_population, left_on="state", right_on="state")
df1['black_death_ratio'] = (df1['count'] * 100000) / df1['TotalPop']
df1['year_semester'] = df1['year'].astype(str) + " | " + df1['semester'].astype(str)
df1.head()

fig, ax = plt.subplots(figsize=(15,5))
ax = sns.barplot(x="year_semester", y="black_death_ratio", data=df1)
ax.set_title('Evolution death rate of Minnesota')
df1

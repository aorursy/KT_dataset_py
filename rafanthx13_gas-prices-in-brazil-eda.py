# Import Libs
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

# Bokeh
from bokeh.io import output_notebook
from bokeh.models import GeoJSONDataSource
output_notebook()

# Print DataSet paths
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Configs
pd.options.display.float_format = '{:,.4f}'.format
sns.set(style="whitegrid")
plt.style.use('seaborn')
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
df = pd.read_csv("/kaggle/input/gas-prices-in-brazil/2004-2019.tsv", delimiter = '\t')
def eda_categ_feat_desc_plot(series_categorical, title = "", fix_labels=False):
    """Generate 2 plots: barplot with quantity and pieplot with percentage. 
       @series_categorical: categorical series
       @title: optional
       @fix_labels: The labes plot in barplot in sorted by values, some times its bugs cuz axis ticks is alphabethic
           if this happens, pass True in fix_labels
       @bar_format: pass {:,.0f} to int
    """
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    
    fig, ax = plt.subplots(figsize = (15,4), ncols=2, nrows=1) # figsize = (width, height)
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

def series_remove_outiliers(series):
    # Use IQR Strategy
    # https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/
    # def quantils
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    print('Cut Off: below than', lower, 'and above than', upper)
    outliers = series[ (series > upper) | (series < lower)]
    print('Identified outliers: {:,d}'.format(len(outliers)), 'that are',
          '{:.2%}'.format(len(outliers)/len(series)), 'of total data')
    # remove outliers
    outliers_removed = [x for x in series if x >= lower and x <= upper]
    print('Non-outlier observations: {:,d}'.format(len(outliers_removed)))
    series_no_outiliers = series[ (series <= upper) & (series >= lower) ]
    return series_no_outiliers
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
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, HoverTool, ColorBar
from bokeh.palettes import magma,viridis,cividis, inferno

def eda_brazil_state_geo_plot(geosource, df_in, title, column, state_column, low = -1, high = -1, palette = -1):
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
    
    hover = HoverTool(tooltips = [ ('State','@{'+state_column+'}'), (column, '@{'+column+'}{%.2f}')],
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
    top = ranked[:top_int].append(ranked[-top_int:]).drop(['geometry'], axis = 1)
    top.index = top.index + 1
    source = ColumnDataSource(data=top)
    list_second_target = source.data[second_target].tolist()
    index_label = list_second_target[::-1] # reverse order label

    p = figure(plot_width=500, plot_height=300, y_range=index_label, 
                toolbar_location=None, title=title)   

    
    # turbo_pallete = Turbo256[0:256:int(256/len(list_second_target) - 2)][::-1] # proportional of number of bars
    p.hbar(y=second_target, right=column_target, source=source, height=0.85, line_color="#000000",
          fill_color=factor_cmap(second_target, palette=inferno(16)[::-1], factors=list_second_target))
    p.x_range.start = 0  # start value of the x-axis

    p.xaxis.axis_label = "value of '" + column_target + "'"

    hover = HoverTool()  # initiate hover tool
    hover.tooltips = [("Value","R$ @{" + column_target + "}" ),   
                       ("Ranking","@index°")]

    hover.mode = 'hline' # set the mode of the hover tool
    p.add_tools(hover)   # add the hover tooltip to the plot

    return p # show in notebook

# Example
# show(eda_bokeh_horiz_bar_ranked(df = dfgeo_fuel_last_row['ETANOL HIDRATADO'], column_target = 'avg_price', title = 'AVG Gasolina',
#     int_top = 5, second_target = 'state') )
def eda_categ_feat_desc_plot(series_categorical, title = "", fix_labels=False):
    """Generate 2 plots: barplot with quantity and pieplot with percentage. 
       @series_categorical: categorical series
       @title: optional
       @fix_labels: The labes plot in barplot in sorted by values, some times its bugs cuz axis ticks is alphabethic
           if this happens, pass True in fix_labels
       @bar_format: pass {:,.0f} to int
    """
    series_name = series_categorical.name
    val_counts = series_categorical.value_counts()
    val_counts.name = 'quantity'
    val_percentage = series_categorical.value_counts(normalize=True)
    val_percentage.name = "percentage"
    val_concat = pd.concat([val_counts, val_percentage], axis = 1)
    val_concat.reset_index(level=0, inplace=True)
    val_concat = val_concat.rename( columns = {'index': series_name} )
    
    fig, ax = plt.subplots(figsize = (15,4), ncols=2, nrows=1) # figsize = (width, height)
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

def series_remove_outiliers(series):
    # Use IQR Strategy
    # https://machinelearningmastery.com/how-to-use-statistics-to-identify-outliers-in-data/
    # def quantils
    q25, q75 = series.quantile(0.25), series.quantile(0.75)
    iqr = q75 - q25
    print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
    cut_off = iqr * 1.5
    lower, upper = q25 - cut_off, q75 + cut_off
    # identify outliers
    print('Cut Off: below than', lower, 'and above than', upper)
    outliers = series[ (series > upper) | (series < lower)]
    print('Identified outliers: {:,d}'.format(len(outliers)), 'that are',
          '{:.2%}'.format(len(outliers)/len(series)), 'of total data')
    # remove outliers
    outliers_removed = [x for x in series if x >= lower and x <= upper]
    print('Non-outlier observations: {:,d}'.format(len(outliers_removed)))
    series_no_outiliers = series[ (series <= upper) & (series >= lower) ]
    return series_no_outiliers
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
from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import LinearColorMapper, HoverTool, ColorBar
from bokeh.palettes import magma,viridis,cividis, inferno

def eda_brazil_state_geo_plot(geosource, df_in, title, column, state_column, low = -1, high = -1, palette = -1):
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
    
    hover = HoverTool(tooltips = [ ('State','@{'+state_column+'}'), (column, '@{'+column+'}{%.2f}')],
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
    top = ranked[:top_int].append(ranked[-top_int:]).drop(['geometry'], axis = 1)
    top.index = top.index + 1
    source = ColumnDataSource(data=top)
    list_second_target = source.data[second_target].tolist()
    index_label = list_second_target[::-1] # reverse order label

    p = figure(plot_width=500, plot_height=300, y_range=index_label, 
                toolbar_location=None, title=title)   

    
    # turbo_pallete = Turbo256[0:256:int(256/len(list_second_target) - 2)][::-1] # proportional of number of bars
    p.hbar(y=second_target, right=column_target, source=source, height=0.85, line_color="#000000",
          fill_color=factor_cmap(second_target, palette=inferno(16)[::-1], factors=list_second_target))
    p.x_range.start = 0  # start value of the x-axis

    p.xaxis.axis_label = "value of '" + column_target + "'"

    hover = HoverTool()  # initiate hover tool
    hover.tooltips = [("Value","R$ @{" + column_target + "}" ),   
                       ("Ranking","@index°")]

    hover.mode = 'hline' # set the mode of the hover tool
    p.add_tools(hover)   # add the hover tooltip to the plot

    return p # show in notebook

# Example
# show(eda_bokeh_horiz_bar_ranked(df = dfgeo_fuel_last_row['ETANOL HIDRATADO'], column_target = 'avg_price', title = 'AVG Gasolina',
#     int_top = 5, second_target = 'state') )
def describe_horizontal_serie(serie):
    adec = serie.describe()
    adtype = serie.dtype
    adf = pd.DataFrame(data=adec.values).T
    adf.columns = adec.index
    adf.index = pd.Index([adec.name])
    if(adtype in ['int64']):
        alist = ['min', '25%', '50%', '75%', 'max']
        for c in alist:
            adf[c] = adf[c].astype('int64')
            adf[c] = adf[c].map(lambda x: "{:,d}".format(int(x)))
    adf['count'] = adf['count'].map(lambda x: "{:,d}".format(int(x)))
    return adf
# rename columns
df.rename(
    columns={
        "DATA INICIAL": "start_date",
        "DATA FINAL": "end_date",
        "REGIÃO": "region",
        "ESTADO": "state",
        "PRODUTO": "fuel",
        "NÚMERO DE POSTOS PESQUISADOS": "n_gas_stations",
        "UNIDADE DE MEDIDA": "unit",
        "PREÇO MÉDIO REVENDA": "avg_price",
        "DESVIO PADRÃO REVENDA": "sd_price",
        "PREÇO MÍNIMO REVENDA": "min_price",
        "PREÇO MÁXIMO REVENDA": "max_price",
        "MARGEM MÉDIA REVENDA": "avg_price_margin",
        "ANO": "year",
        "MÊS": "month",
        "COEF DE VARIAÇÃO DISTRIBUIÇÃO": "coef_dist",
        "PREÇO MÁXIMO DISTRIBUIÇÃO": "dist_max_price",
        "PREÇO MÍNIMO DISTRIBUIÇÃO": "dist_min_price",
        "DESVIO PADRÃO DISTRIBUIÇÃO": "dist_sd_price",
        "PREÇO MÉDIO DISTRIBUIÇÃO": "dist_avg_price",
        "COEF DE VARIAÇÃO REVENDA": "coef_price"
    },
    inplace=True
)
# converter several units of measurement for a single
event_dictionary_units = {'R$/l ' : 'R$/l', 'R$/13Kg' : 'R$/l', 'R$/m3' : 'R$/l'}
event_dictionary_conversion = {'R$/l' : 1 , 'R$/13Kg' : 0.00006153, 'R$/m3' : 0.001}

df['conversion'] = df['unit'].map(event_dictionary_conversion) # new column
df['avg_price'] = df.avg_price * df.conversion # convert avg_price to R$/l numeric
df['unit'] = df['unit'].map(event_dictionary_units)
# Convert to Number
numeric_columns = ['sd_price','min_price', 'max_price','avg_price_margin','coef_price', 
                        'dist_avg_price', 'dist_sd_price', 'dist_min_price', 'dist_max_price',
                        'coef_dist']
for col in numeric_columns:
  df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert to Date Coluns to DateTime
for col in ['start_date', 'end_date']:
    df[col] = pd.to_datetime(df[col]) 
# remove useless columns
df = df.drop(["unit", "conversion", 'Unnamed: 0'], axis = 1) 
# final dataset
df.head()
sns.heatmap(df.isnull(), cbar=False)
plt.show()
# replace with median values
df['avg_price_margin'].fillna(df['avg_price_margin'].median(), inplace = True)
df['dist_avg_price'].fillna(df['dist_avg_price'].median(), inplace = True)
df['dist_sd_price'].fillna(df['dist_sd_price'].median(), inplace = True)
df['dist_min_price'].fillna(df['dist_min_price'].median(), inplace = True)
df['dist_max_price'].fillna(df['dist_max_price'].median(), inplace = True)
df['coef_dist'].fillna(df['coef_dist'].median(), inplace = True)
sns.heatmap(df.isnull(), cbar=False)
plt.show()
eda_categ_feat_desc_plot(df['region'], 'Records by Region')
describe_horizontal_serie(df['n_gas_stations'])
f, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(17, 3))

sns.boxplot(x=df['n_gas_stations'], ax=ax1)
sns.boxplot(x=series_remove_outiliers(df.query('state != "SAO PAULO"')['n_gas_stations']), ax=ax2)

ax1.set_title("Dist. of Number of gas Station in All States")
ax2.set_title("Dist. of Number of gas Station in All States (Remove Outiliers)")

plt.show()
eda_categ_feat_desc_plot(df['fuel'], 'Fuel Distribution')
fig, ax = plt.subplots(figsize=(18,6))
ax = sns.countplot(x="year", data=df)
plt.title("Number of records per year", fontsize=24)
plt.xlabel('Year', fontsize=18)
plt.ylabel('Count', fontsize=18)
plt.xticks(rotation=45, fontsize=12, ha='right')
plt.show()
# dict to mapping states
obj_map_states = {
 'ACRE': 'AC',
 'ALAGOAS': 'AL',
 'AMAPA': 'AP',
 'AMAZONAS': 'AM',
 'BAHIA': 'BA',
 'CEARA': 'CE',
 'DISTRITO FEDERAL': 'DF',
 'ESPIRITO SANTO': 'ES',
 'GOIAS': 'GO',
 'MARANHAO': 'MA' ,
 'MATO GROSSO': 'MT' ,
 'MATO GROSSO DO SUL': 'MS',
 'MINAS GERAIS': 'MG',
 'PARA': 'PA',
 'PARAIBA': 'PB',
 'PARANA': 'PR',
 'PERNAMBUCO': 'PE',
 'PIAUI': 'PI',
 'RIO DE JANEIRO': 'RJ',
 'RIO GRANDE DO NORTE': 'RN',
 'RIO GRANDE DO SUL':  'RS',
 'RONDONIA': 'RO',
 'RORAIMA': 'RR',
 'SANTA CATARINA': 'SC',
 'SAO PAULO': 'SP',
 'SERGIPE': 'SE',
 'TOCANTINS': 'TO'
}
obj_map_states = {v: k for k, v in obj_map_states.items()}

mapping_dict_regions = {'3': 'NORTE', '4': 'NORDESTE', '2': 'SUDESTE', '1': 'SUL', '5': 'CENTRO OESTE'}
# import brazil-state.geojson and prepare it

import geopandas as gpd

# get geojson
brazil_geojson = gpd.read_file('../input/brazilstatejsongeogrpah/brazil-states.geojson')
# sort
brazil_geojson = brazil_geojson.sort_values('name')
# delete useless columns
columns_to_drop = ['id', 'codigo_ibg', 'cartodb_id', 'created_at', 'updated_at']
brazil_geojson = brazil_geojson.drop(columns_to_drop, axis = 1)
# Map state name
brazil_geojson['sigla'].tolist()
brazil_geojson['state_sigla'] = brazil_geojson['sigla'].map(lambda v: obj_map_states[v])
# region
brazil_geojson['regiao_id'] = brazil_geojson['regiao_id'].map(mapping_dict_regions)

brazil_geojson.head()
# create dict with last row to each product

dict_fuel_last_row = {}

all_states = df['state'].unique().tolist()
all_fuel = df['fuel'].unique().tolist()
for f in all_fuel:
    # get last row to each fuel f to each one of 27 states
    by_state = []
    for s in all_states:
        try:
            by_state.append( df[ (df['fuel'] == f) & (df['state'] == s) ]['start_date'].idxmax() )
        except:
            # Dont exist GNV in Roraima
            print('Dont exist ...', f,s)
            continue
    dict_fuel_last_row[f] = df.iloc[ by_state ]

# Copy a row of ACRE to RORAIMA

roraima_gnv_row = dict_fuel_last_row['GNV'].loc[ dict_fuel_last_row['GNV']['state'] == 'ACRE']
roraima_gnv_row['state'] = 'RORAIMA'
dict_fuel_last_row['GNV'] = dict_fuel_last_row['GNV'].append(roraima_gnv_row, ignore_index = True)
len(dict_fuel_last_row['GNV'])
# Create df_geo (df join geo) and GeoJSONDataSource to each fuel using last date record

from bokeh.models import GeoJSONDataSource

dfgeo_fuel_last_row = {}
geosource_fuel_last_row = {}
for key, value in dict_fuel_last_row.items():
    dfgeo_fuel_last_row[key] = brazil_geojson.merge(dict_fuel_last_row[key], left_on = 'state_sigla', right_on = 'state').drop(['start_date', 'end_date'], axis = 1)
    geosource_fuel_last_row[key] = GeoJSONDataSource(geojson = dfgeo_fuel_last_row[key].to_json())  
# Generate to one price/dist to each fuel
# type = avg_price, sd_price, min_price, max_price, avg_price_margin, coef_price, dist_avg_price,
#        dist_sd_price, dist_min_price, dist_max_price, coef_dist

type_price = 'avg_price'
geo = {}
rank = {}
for p in geosource_fuel_last_row.keys():
    product = p
    
    geo[p] =  eda_brazil_state_geo_plot( geosource_fuel_last_row[product], dfgeo_fuel_last_row[product], type_price + ' to ' + product, 
                                    type_price, "state", palette = inferno(16) )
    
    rank[p] =   eda_bokeh_horiz_bar_ranked(df = dfgeo_fuel_last_row[product], column_target = type_price,
                                      title = 'Ranking: first and last 5 of ' + product , int_top = 5, second_target = 'state')
# Choose One : ['ETANOL HIDRATADO', 'GASOLINA COMUM', 'GLP', 'GNV', 'ÓLEO DIESEL', 'ÓLEO DIESEL S10']

from bokeh.layouts import row

p = 'ÓLEO DIESEL S10'
show( row( geo[p],  rank[p]))
p = 'GASOLINA COMUM'
show( row( geo[p],  rank[p]))
# Example to generate other price/coef Generate content to 'dist_avg_price'
type_price = 'dist_avg_price'
geo1 = {}
rank1 = {}
for p in geosource_fuel_last_row.keys():
    product = p
    
    geo1[p] =  eda_brazil_state_geo_plot( geosource_fuel_last_row[product], dfgeo_fuel_last_row[product], type_price + ' to ' + product, 
                                    type_price, "state", palette = inferno(16) )
    
    rank1[p] =   eda_bokeh_horiz_bar_ranked(df = dfgeo_fuel_last_row[product], column_target = type_price,
                                      title = 'Ranking: first and last 5 of ' + product , int_top = 5, second_target = 'state')
p = 'GASOLINA COMUM'
show( row( geo1[p],  rank1[p]))
p = 'ÓLEO DIESEL'
show( row( geo1[p],  rank1[p]))
# avg_price, sd_price, min_price, max_price, avg_price_margin, coef_price, 
# dist_avg_price, dist_sd_price, dist_min_price, dist_max_price, coef_dist
type_price = 'coef_dist' 

# 'ETANOL HIDRATADO', 'GASOLINA COMUM', 'GLP', 'GNV', 'ÓLEO DIESEL', 'ÓLEO DIESEL S10']
product = 'GASOLINA COMUM' 

geo2 = eda_brazil_state_geo_plot( geosource_fuel_last_row[product], dfgeo_fuel_last_row[product], type_price + ' to ' + product, 
                                    type_price, "state", palette = inferno(16) )
rank2 = eda_bokeh_horiz_bar_ranked(df = dfgeo_fuel_last_row[product], column_target = type_price,
                                      title = 'Ranking: first and last 5 of ' + product , int_top = 5, second_target = 'state')
    
show( row( geo2, rank2) )
# max of gas_station to each state

df_gas_s = df.groupby(['state']).max()['n_gas_stations']
df_gas_s.sort_values(ascending = False)
# mean of all records to each state

df_gas_s = df.groupby(['state']).mean()['n_gas_stations']
df_gas_s.sort_values(ascending = False)
df_gas_s = df.groupby(['state']).sum()['n_gas_stations']
df_gas_s.sort_values(ascending = False)
df_price = df.query("fuel == 'GASOLINA COMUM'").groupby(['state']).mean()['avg_price'].sort_values(ascending = False)
df_price = df_price.to_frame().reset_index()
df_price['name'] = df_price['state'].apply(lambda name: name.capitalize())
df_price.sort_values('avg_price', ascending=False).head()
df_avg_all_gas = brazil_geojson.merge(df_price, left_on = 'state_sigla', right_on = 'state').drop(['state_sigla'], axis = 1)

geo_source_avg_all_gas = GeoJSONDataSource(geojson = df_avg_all_gas.to_json())

type_price = 'avg_price'
product = 'GASOLINA COMUM'

geo3 = eda_brazil_state_geo_plot(
    geo_source_avg_all_gas, df_avg_all_gas,
    "avg_price of gas ('GASOLINA COMUM') of all time", type_price, "state", palette = inferno(16) )

rank3 = eda_bokeh_horiz_bar_ranked(
    df = df_avg_all_gas, column_target = type_price, 
    title = 'Ranking: first and last 5 of ' + product , int_top = 5, second_target = 'state')
    
show( row( geo3, rank3) )
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('More expensive and cheaper gas')

df.query("fuel == 'GASOLINA COMUM' ").groupby(
    ['year','state']).mean()['avg_price'].unstack().plot(ax=ax1, figsize = (16,5), grid = True, legend = False)
ax1.set_ylabel("avg_price")
ax1.set_title("All States")

df.query("fuel == 'GASOLINA COMUM' & state in ['SAO PAULO', 'ACRE']").groupby(
    ['year','state']).mean()['avg_price'].unstack().plot(ax=ax2, figsize = (16,5), grid = True)
ax2.set_ylabel("avg_price")
ax2.set_title("SAO PAULO x ACRE")

plt.show()
df_re = df.query('fuel == "GASOLINA COMUM" & year == 2019').groupby(['year','region']).mean()['avg_price'].reset_index()

geoplot_region = brazil_geojson.dissolve(by='regiao_id').reset_index()

## Plot region
# geoplot_region['geometry'][1]

drop_columns = ['name', 'sigla', 'year', 'regiao_id']
geoplot_re_final = geoplot_region.merge(df_re, left_on = 'regiao_id', right_on = 'region').drop(drop_columns, axis = 1)
geo_source_region_gas = GeoJSONDataSource(geojson = geoplot_re_final.to_json())

geoplot_re_final.head()
# Plot 'GASOLINA COMUM' avg_price in 2019
show(
    eda_brazil_state_geo_plot( geo_source_region_gas, geoplot_re_final, "avg_price of gas by region in 2019",
                              'avg_price', "region", palette = inferno(16) )
)
# Evolution of gas by region since 2004 to 2019

fig, ax = plt.subplots(figsize=(15,5))
df.query('fuel == "GASOLINA COMUM"').groupby(['year','region']).mean()['avg_price'].unstack().plot(ax=ax)
ax.set_ylabel("avg_price")
ax.set_title("avg_price of gas per region")
plt.grid(True)
# Plot evolution of each region states separated

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, sharey=True, figsize=(15,15) )
fig.suptitle('Gas per Region')

df.query("fuel == 'GASOLINA COMUM' & region == 'NORDESTE' ").groupby(
    ['year','state']).mean()['avg_price'].unstack().plot(ax=ax1, figsize = (15,9), grid = True)
ax1.set_ylabel("avg_price")
ax1.set_title("Gas in NORDESTE")

df.query("fuel == 'GASOLINA COMUM' & region == 'SUL' ").groupby(
    ['year','state']).mean()['avg_price'].unstack().plot(ax=ax2, figsize = (15,9), grid = True)
ax2.set_ylabel("avg_price")
ax2.set_title("Gas in SUL")

df.query("fuel == 'GASOLINA COMUM' & region == 'NORTE' ").groupby(
    ['year','state']).mean()['avg_price'].unstack().plot(ax=ax3, figsize = (15,9), grid = True)
ax3.set_ylabel("avg_price")
ax3.set_title("Gas in NORTE")

df.query("fuel == 'GASOLINA COMUM' & region == 'CENTRO OESTE' ").groupby(
    ['year','state']).mean()['avg_price'].unstack().plot(ax=ax4, figsize = (15,9), grid = True)
ax4.set_ylabel("avg_price")
ax4.set_title("Gas in CENTRO OESTE")

df.query("fuel == 'GASOLINA COMUM' & region == 'SUDESTE' ").groupby(
    ['year','state']).mean()['avg_price'].unstack().plot(ax=ax5, figsize = (15, 27), grid = True)
ax5.set_ylabel("avg_price")
ax5.set_title("Gas in SUDESTE")

plt.show()
# Plot all state avg by year

fig, ax1 = plt.subplots(figsize=(15,7))
df.query("fuel == 'GASOLINA COMUM' ").groupby(
    ['year','state']).mean()['avg_price'].unstack().plot(ax=ax1, figsize = (16,5), grid = True)
ax1.set_ylabel("avg_price")
ax1.set_title("All States")
plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=2)

plt.show()
# Generate dataframe 'df_diff_2017_2018' with the increase of gas price betwen 2017 and 2018

diff = df.query("fuel == 'GASOLINA COMUM' ").groupby(['year','state']).mean()['avg_price'].reset_index()

diff_2017 = diff.query('year == 2017')
diff_2018 = diff.query('year == 2018')


rename_columns = {'avg_price_x': 'avg_price_2018', 'avg_price_y': 'avg_price_2017'}

df_diff_2017_2018 = diff_2018.merge(diff_2017, left_on = 'state', right_on = 'state').rename(
    rename_columns, axis = 1).drop(['year_x','year_y'], axis = 1)

df_diff_2017_2018['increase'] = df_diff_2017_2018['avg_price_2018'] - df_diff_2017_2018['avg_price_2017']
df_diff_2017_2018.sort_values('increase', ascending = False).head()
# Plot gas price difference between 2017 and 2018

df_geo_diff_2017_2018 = brazil_geojson.merge(df_diff_2017_2018, left_on = 'state_sigla', right_on = 'state').drop(['sigla'], axis = 1)
df_geo_diff_2017_2018.head()

geo_source_diff_2017_2018 = GeoJSONDataSource(geojson = df_geo_diff_2017_2018.to_json())

type_price = 'increase'

geo_plot = eda_brazil_state_geo_plot( geo_source_diff_2017_2018, df_geo_diff_2017_2018, "Increase in gasoline between 2017 and 2018", type_price, "state", palette = inferno(16) )

rank_plot = eda_bokeh_horiz_bar_ranked(df = df_geo_diff_2017_2018, column_target = type_price,
                                      title = 'Ranking: first and last 8 of increase in gasoline' , int_top = 8, second_target = 'state')
    
show( row( geo_plot, rank_plot) )
# Generate dataframe 'df_diff_2017_2018' with the increase of gas price betwen 2017 and 2018

diff = df.query("fuel == 'GASOLINA COMUM' ").groupby(['year','state']).mean()['avg_price'].reset_index()

diff_2017 = diff.query('year == 2017')
diff_2018 = diff.query('year == 2018')
diff_2019 = diff.query('year == 2019')


rename_columns = {'avg_price_x': 'avg_price_2018', 'avg_price_y': 'avg_price_2017'}

df_diff_2017_2018 = diff_2018.merge(diff_2017, left_on = 'state', right_on = 'state').rename(rename_columns, axis = 1)

df_diff_2017_2019 = diff_2019.merge(df_diff_2017_2018, left_on = 'state', right_on = 'state')

df_diff_2017_2019

df_diff_2017_2019['increase2'] = df_diff_2017_2019['avg_price'] - df_diff_2017_2019['avg_price_2018']
df_diff_2017_2019.sort_values('increase2', ascending = False).head()
df_geo_diff_2017_2019 = brazil_geojson.merge(df_diff_2017_2019, left_on = 'state_sigla', right_on = 'state').drop(['state_sigla'], axis = 1)

geo_source_diff_2017_2019 = GeoJSONDataSource(geojson = df_geo_diff_2017_2019.to_json())

type_price = 'increase2'

geo_plot = eda_brazil_state_geo_plot( geo_source_diff_2017_2019, df_geo_diff_2017_2019, "Increase in gasoline between 2018 and 2019",
                                     type_price, "state", palette = inferno(32) )

rank_plot = eda_bokeh_horiz_bar_ranked(df = df_geo_diff_2017_2019, column_target = type_price,
                                      title = 'Ranking: first and last 8 of increase in gasoline' , int_top = 8, second_target = 'state')
    
show( row( geo_plot, rank_plot) )
# Plot evolution of gas price by region between 2017 and 2019

fig, ax = plt.subplots(figsize=(15,5))
df.query('fuel == "GASOLINA COMUM" & year in [2017, 2018, 2019]' ).groupby(['year','region']).mean()['avg_price'].unstack().plot(ax=ax)
ax.set_ylabel("avg_price")
ax.set_title("avg_price of gas per region betwen 2017 and 2019")
plt.grid(True)
plt.show()
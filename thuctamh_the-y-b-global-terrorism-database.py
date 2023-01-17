#@title

# !pip install kmapper

# !pip install igraph

# !pip install plotly

# !pip install ipywidgets

# !pip install umap-learn
#@title

import numpy as np

import pandas as pd 

pd.set_option('display.max_columns', 200)

pd.set_option('display.max_rows', 10)



import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



import matplotlib.pyplot as plt

import matplotlib as mpl

import seaborn as sns

import scipy.stats as stats



from IPython.core.display import HTML, display, Javascript



import kmapper as km

import sklearn

from sklearn import ensemble



from html.parser import HTMLParser

import requests

from bs4 import BeautifulSoup



# from IPython.display import display, Javascript, HTML

import json

# # bokeh

import bokeh

import bokeh.io

import bokeh.models

import bokeh.palettes

import bokeh.plotting

from bokeh.plotting import figure

from bokeh.resources import CDN

from bokeh.embed import file_html

# from bokeh.io import gridplot, output_file, show

from bokeh.layouts import gridplot

from bokeh.io import output_file, show



# Display graphics in this notebook

bokeh.io.output_notebook()



# # pd.set_option('display.width', 1000)
#@title

def influence(row):

    """ Used for display information with geomap, it accept one row and return list of tupple"""

    

    color, size = None, None # color and size of dots on the maps, the larger the more important

    

    if row['Killed'] == 0: # event with killed people equals zero

        color, size = 'lightgrey', 5

        

    elif row['Killed'] <= 20 and row['Killed'] > 0: # event with killed people is in range (0, 20)

        color, size = 'lightseagreen', 10

        

    elif row['Killed'] <= 100 and row['Killed'] > 20: # event with killed people is in range (20, 100)

        color, size = 'royalblue', 20

        

    elif row['Killed'] <= 200 and row['Killed'] > 100:# event with killed people is in range (100, 200)

        color, size = 'crimson', 30

        

    elif row['Killed'] > 200: # event with killed people is higher than 200

        color, size = 'red', 60

        

    else: # event with injured people won't be displayed clearly on the map

        color, size = 'orange', 1

        

    return (row['longitude'], row['latitude'], 

            row['Killed'], row['Group'], row['Motive'], 

            color, size,

            row['Year'], row['Month'], row['Day'])
#@title

def count_event_with_year(row, selected_reg):

    """ return information about year and event happended at the certain region """

    

    if row['Region'] == selected_reg:

        return (row['Year'], row['eventid'])
#@title

def test_success_type(row, selected_reg, selected_type):

    """return information about year and event happended at the certain region and attack type"""

    

    if row['Region'] == selected_reg:

        if row['AttackType'] == selected_type:

            return (row['Year'], row['eventid'])
#@title

def make_expression_axes(tooltips, title,

                          xlabel, ylabel):

    """A function to plot the bokeh single mutant comparisons."""

    # Make the hover tool

    hover = bokeh.models.HoverTool(tooltips=tooltips,

                                   names=['circles'])



    # Create figure

    p = bokeh.plotting.figure(title=title, plot_width=650, 

                              plot_height=450)



    p.xgrid.grid_line_color = 'white'

    p.ygrid.grid_line_color = 'white'

    p.xaxis.axis_label = xlabel

    p.yaxis.axis_label = ylabel



    # Add the hover tool

    p.add_tools(hover)

    return p
#@title

def add_points(p, df1, x, y, se_x, color='blue', alpha=0.2, outline=False):

    # Define colors in a dictionary to access them with

    # the key from the pandas groupby funciton.

    df = df1.copy()

    transformed_q = -df[y].apply(np.log10)

    df['transformed_q'] = transformed_q

    #    FEATURE_delta_MEAN_IC50

    df['transform_e'] = list(np.sign(df[se_x])*df[x])

    source1 = bokeh.models.ColumnDataSource(df)



    # Specify data source

    p.circle(x='transform_e', y='transformed_q', size=7,

             alpha=alpha, source=source1,

             color=color, name='circles')

    if outline:

        p.circle(x='transform_e', y='transformed_q', size=7,

                 alpha=1,

                 source=source1, color='black',

                 fill_color=None, name='outlines')



    # prettify

    p.background_fill_color = "#DFDFE5"

    p.background_fill_alpha = 0.5

    

    return p
#@title

def selector(df,psig,fdrsig):

    """A function to separate tfs from everything else"""

    sig_p = (df['ANOVA_FEATURE_pval'] < psig)

    sig_fdr = (df['ANOVA_FEATURE_FDR'] < fdrsig)

    to_plot_yes = df[sig_p & sig_fdr]

    to_plot_not = df[~sig_p & ~sig_fdr]

    return to_plot_not, to_plot_yes
#@title

def replace_drugname(df):

    df = df.replace({'nkill':'Number People Killed',

                'nkillus':'Number US People Killed',

                'success':'Success',

                'propvalue':'Damaged Property Value',

                'nwound':'Number Wounded',

                'nwoundus':'Number US Wounded',

                'nhostkid':'Number Kidnapped Hostage',

                'nhostkidus':'Number Kidnapped US Hostage',

                'ndays':'Days Kidnapped Hostage',

                'nreleased':'Number Hostage Released',

                'ransomamt':'Ransom Amount',

                'ransomamtus':'Ransom Amount from US sources',

                'ransompaid':'Amount Ransom Paid',

                'ransompaidus':'Amount Ransom Paid by US sources'},inplace=True)
df = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1', low_memory=False)

kdf = pd.read_csv('../input/gtd/globalterrorismdb_0718dist.csv', encoding='ISO-8859-1', low_memory=False)

df.shape
df.sample(n = 5)
df.describe()
#@title

# check the null values in each features and visual it

features_not_null_count = []



for feature in df.columns:

    features_not_null_count.append(df[feature].notnull().sum()) # null values in one feature



# add trace to plotly bar graph

trace1 = go.Bar(name = "Values are not null", x = df.columns, y = features_not_null_count)



layout1 = {'title': "Values not null in each feature",

           'margin': go.layout.Margin(l=10,r=10,b=10,t=60,pad=4),

            'width': 850, 'height' : 400 }



fig_1 = go.Figure(data = trace1, layout = layout1)



fig_1.show()
#@title

columns_dropped = ['approxdate', 'resolution', 'location', 'alternative', 

                   'alternative_txt', 'targtype2', 'targtype2_txt', 'targsubtype2', 'targsubtype2_txt', 'corp2',

                   'target2', 'natlty2', 'natlty2_txt', 'targtype3', 'targtype3_txt', 'targsubtype3',

                   'targsubtype3_txt', 'corp3', 'target3', 'natlty3', 'natlty3_txt',

                   'attacktype2', 'attacktype2_txt', 'attacktype3', 'attacktype3_txt',

                   'gsubname', 'gname2','gsubname2', 'gname3', 'gsubname3', 'guncertain2', 'guncertain3',

                   'nperps', 'nperpcap',

                   'claimmode', 'claimmode_txt', 'claim2', 'claimmode2',

                   'claimmode2_txt', 'claim3', 'claimmode3', 'claimmode3_txt','compclaim',

                   'scite2', 'scite3', 'weapdetail',

                   'propextent', 'propextent_txt', 'propcomment',

                   'weaptype2', 'weaptype2_txt', 'weapsubtype2', 'weapsubtype2_txt',

                   'weaptype3', 'weaptype3_txt', 'weapsubtype3', 'weapsubtype3_txt', 'weaptype4',

                   'weaptype4_txt', 'weapsubtype4', 'weapsubtype4_txt',

                   'nhostkid', 'nhostkidus', 'nhours', 'ndays', 'divert', 'kidhijcountry',

                   'ransom', 'ransomamt', 'ransomamtus', 'ransompaid', 'ransompaidus', 'ransomnote',

                   'hostkidoutcome', 'hostkidoutcome_txt', 'nreleased', 'addnotes',]
# drop irrelevant features

df = df.drop(columns=columns_dropped)
df.shape
#@title

# check the null values in each features and visual it

features_not_null_count = []



for feature in df.columns:

    features_not_null_count.append(df[feature].notnull().sum()) # null values in one feature



# add trace to plotly bar graph

trace2 = go.Bar(name = "Values are not null", x = df.columns, y = features_not_null_count)



layout2 = {'title': "Values not null in each feature",

           'margin': go.layout.Margin(l=10,r=10,b=10,t=60,pad=4),

            'width': 850, 'height' : 400 }



fig_2 = go.Figure(data = trace2, layout = layout1)



fig_2.show()
#@title

df.rename(columns={'iyear':'Year','imonth':'Month','iday':'Day',

                       'country_txt':'Country','region_txt':'Region',

                       'attacktype1_txt':'AttackType','target1':'Target',

                       'nkill':'Killed','nwound':'Wounded','summary':'Summary',

                       'gname':'Group','targtype1_txt':'Target_type',

                       'weaptype1_txt':'Weapon_type','motive':'Motive'}, inplace=True)
#@title

# year: ['1970', '1980', ...] + month: ['1', '2', ...] + day: ['1', '2', ...] for specific day

# using unique() fuctions to get all values 

# in ranges



year_check = ['2001']

month_check = df['Month'].unique()

day_check = df['Day'].unique() 
#@title

# select events happenden at the time

df_geo = df[ (df['Year'].isin(year_check) ) & 

             (df['Month'].isin(month_check)) & 

             (df['Day'].isin(day_check)) ]
#@title

# apply influence to get details

geo_data = df_geo.apply(influence, axis='columns')

unzip_geo_data = list(zip(*geo_data))



info_data = {'long': unzip_geo_data[0],  # list of longitute

             'lat': unzip_geo_data[1], # list of lattitude

             'killed': unzip_geo_data[2], # people be killed

             'group': unzip_geo_data[3],  # name of group is confirmed opened attack

             'motive': unzip_geo_data[4], # motive behind the attack

             'color': unzip_geo_data[5], # alert sign

             'size': unzip_geo_data[6], # influence of the attack

             'year': unzip_geo_data[7],

             'month': unzip_geo_data[8],

             'day': unzip_geo_data[9],

            }



# text appeared when hovering dots on the map

hover_txt = [f'Date: {ele[9]}/{ele[8]}/{ele[7]},\n'

             f'Killed: {ele[2]},\n'

             f'By group: {ele[3]},\n'

             f'Motive: {ele[4]},\n' 

             for ele in geo_data]
#@title

# add data for geo map

geo_map = go.Scattermapbox( mode = "markers", marker = {'size': info_data['size'], 'color': info_data['color'] },

                            lon = info_data['long'], 

                            lat = info_data['lat'], 

                            hovertext = hover_txt)



fig_map = go.Figure(geo_map)

fig_map.update_layout(mapbox = {'style': "stamen-terrain", 

                              'center': {'lon': info_data['long'][0], 'lat': info_data['lat'][0] }, 'zoom': 1}, 

                               showlegend = False,

                               margin = go.layout.Margin(l=10,r=10,b=10,t=10,pad=4),

                               width = 850, height = 800)

        

fig_map.show()
#@title

menadf = pd.read_csv('../input/gtdanovaresults/mideastnorthaf-results.csv')

replace_drugname(menadf)

menadf

tooltips = [('Feature', '@FEATURE'),('Label','@DRUG_NAME')]

p = make_expression_axes( tooltips, 'Middle East North Africa ANOVA associations',

                         'Signed Effect Size', '-log(Q)')

to_plot_not, to_plot_yes = selector(menadf,0.001,25)

p = add_points(p, to_plot_not, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#1a9641')

p = add_points(p, to_plot_yes, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#fc8d59', alpha=0.6, outline=True)

html = file_html(p, CDN, "my plot")

HTML(html)
#@title

noramdf = pd.read_csv('../input/gtdanovaresults/northam-results.csv')

replace_drugname(noramdf)

noramdf

tooltips = [('Feature', '@FEATURE'),('Label','@DRUG_NAME')]

p = make_expression_axes( tooltips, 'North America ANOVA associations',

                         'Signed Effect Size', '-log(Q)')

to_plot_not, to_plot_yes = selector(noramdf,0.001,25)

p = add_points(p, to_plot_not, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#1a9641')

p = add_points(p, to_plot_yes, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#fc8d59', alpha=0.6, outline=True)

html = file_html(p, CDN, "my plot")

HTML(html)
#@title

soasiadf = pd.read_csv('../input/gtdanovaresults/southasia-results.csv')

soasiadf

tooltips = [('Feature', '@FEATURE'),('Label','@DRUG_NAME')]

p = make_expression_axes( tooltips, 'South Asia ANOVA associations',

                         'Signed Effect Size', '-log(Q)')

to_plot_not, to_plot_yes = selector(soasiadf,0.001,25)

p = add_points(p, to_plot_not, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#1a9641')

p = add_points(p, to_plot_yes, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#fc8d59', alpha=0.6, outline=True)

html = file_html(p, CDN, "my plot")

HTML(html)
#@title

soeastasiadf = pd.read_csv('../input/gtdanovaresults/southeastasia-results.csv')

soeastasiadf

tooltips = [('Feature', '@FEATURE'),('Label','@DRUG_NAME')]

p = make_expression_axes( tooltips, 'Southeast Asia ANOVA associations',

                         'Signed Effect Size', '-log(Q)')

to_plot_not, to_plot_yes = selector(soeastasiadf,0.001,25)

p = add_points(p, to_plot_not, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#1a9641')

p = add_points(p, to_plot_yes, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#fc8d59', alpha=0.6, outline=True)

html = file_html(p, CDN, "my plot")

HTML(html)
#@title

souamdf = pd.read_csv('../input/gtdanovaresults/southam-results.csv')

replace_drugname(souamdf)

cenamdf = pd.read_csv('../input/gtdanovaresults/cenam-carib-results.csv')

replace_drugname(cenamdf)

easteudf = pd.read_csv('../input/gtdanovaresults/easteu-results.csv')

replace_drugname(easteudf)

westeudf = pd.read_csv('../input/gtdanovaresults/westeu-results.csv')

subafdf = pd.read_csv('../input/gtdanovaresults/subaf-results.csv')
#@title

tooltips = [('Feature', '@FEATURE'),('Label','@DRUG_NAME')]

#South America

p1 = make_expression_axes( tooltips, 'South America ANOVA associations',

                         'Signed Effect Size', '-log(Q)')

to_plot_not, to_plot_yes = selector(souamdf,0.001,25)

p1 = add_points(p1, to_plot_not, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#1a9641')

p1 = add_points(p1, to_plot_yes, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#fc8d59', alpha=0.6, outline=True)

#Central America & Carribean

p2 = make_expression_axes( tooltips, 'Central America & Caribbean ANOVA associations',

                         'Signed Effect Size', '-log(Q)')

to_plot_not2, to_plot_yes2 = selector(cenamdf,0.001,25)

p2 = add_points(p2, to_plot_not2, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#1a9641')

p2 = add_points(p2, to_plot_yes2, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#fc8d59', alpha=0.6, outline=True)

#Eastern Europe

p3 = make_expression_axes( tooltips, 'Eastern Europe ANOVA associations',

                         'Signed Effect Size', '-log(Q)')

to_plot_not3, to_plot_yes3 = selector(easteudf,0.001,25)

p3= add_points(p3, to_plot_not3, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#1a9641')

p3 = add_points(p3, to_plot_yes3, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#fc8d59', alpha=0.6, outline=True)

#Western Europe

p4 = make_expression_axes( tooltips, 'Western Europe ANOVA associations',

                         'Signed Effect Size', '-log(Q)')

to_plot_not4, to_plot_yes4 = selector(westeudf,0.001,25)

p4 = add_points(p4, to_plot_not4, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#1a9641')

p4 = add_points(p4, to_plot_yes4, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#fc8d59', alpha=0.6, outline=True)

#Sub-Saharan Africa

p5 = make_expression_axes( tooltips, 'Sub-Saharan Africa ANOVA associations',

                         'Signed Effect Size', '-log(Q)')

to_plot_not5, to_plot_yes5 = selector(subafdf,0.001,25)

p5 = add_points(p5, to_plot_not5, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#1a9641')

p5 = add_points(p5, to_plot_yes5, 'FEATURE_IC50_effect_size', 'ANOVA_FEATURE_pval', 'FEATURE_delta_MEAN_IC50', color='#fc8d59', alpha=0.6, outline=True)

#Make grid

grid = gridplot([[p1, p2], [p3, p4], [p5,None]], plot_width=500, plot_height=500)

html = file_html(grid, CDN, "my plot")

HTML(html)
#@title

df_success = df[df['success'] == 1] # success events happended

df_year_region = df_success.groupby(['Year', 'Region'])['eventid'].count()

df_year_region = df_year_region.reset_index()
#@title

df_fail = df[df.success == 0] # fail events

df_year_region_fail = df_fail.groupby(['Year', 'Region'])['eventid'].count()

df_year_region_fail = df_year_region_fail.reset_index()
#@title

aus_ocean = df_year_region[df_year_region['Region'] == "Australasia & Oceania"]

cen_ame_car = df_year_region[df_year_region['Region'] == "Central America & Caribbean"]

east_asia = df_year_region[df_year_region['Region'] == "East Asia"]

east_eu = df_year_region[df_year_region['Region'] == "Eastern Europe"]

mid_east_north_af = df_year_region[df_year_region['Region'] == "Middle East & North Africa"]

north_ame = df_year_region[df_year_region['Region'] == "North America"]

south_ame = df_year_region[df_year_region['Region'] == "South America"]

south_asia = df_year_region[df_year_region['Region'] == "South Asia"]

southest_asia = df_year_region[df_year_region['Region'] == "Southeast Asia"]

sub_af = df_year_region[df_year_region['Region'] == "Sub-Saharan Africa"]

west_eu = df_year_region[df_year_region['Region'] == "Western Europe"]

cen_asia = df_year_region[df_year_region['Region'] == "Central Asia"]
#@title

fig = go.Figure()

monthList = list(np.arange(1970,2018,1))



# Australasia & Oceania

monthList_ao  = aus_ocean['Year'].tolist()

ausocean_data = aus_ocean['eventid'].tolist()

monthListao_trace = go.Scatter(x=monthList_ao, y=ausocean_data, 

                               name="Australasia & Oceania", mode='lines+markers',)

fig.add_trace(monthListao_trace)



# Central America & Caribbean

monthList_caa  = cen_ame_car['Year'].tolist()

cenamecar_data = cen_ame_car['eventid'].tolist()

monthListcaa_trace = go.Scatter(x=monthList_caa, y=cenamecar_data, name="Central America & Caribbean", mode='lines+markers')

fig.add_trace(monthListcaa_trace)



# East Asia

monthList_ea  = east_asia['Year'].tolist()

eastasia_data = east_asia['eventid'].tolist()

monthListea_trace = go.Scatter(x=monthList_ea, y=eastasia_data,name="East Asia", mode='lines+markers')

fig.add_trace(monthListea_trace)



# Eastern Europe

monthList_ee  = east_eu['Year'].tolist()

easteu_data = east_eu['eventid'].tolist()

monthListee_trace = go.Scatter(x=monthList_ee, y=easteu_data,name="Eastern Europe", mode='lines+markers')

fig.add_trace(monthListee_trace)



# Middle East & North Africa

monthList_mena  = mid_east_north_af['Year'].tolist()

mideastnorthaf_data = mid_east_north_af['eventid'].tolist()

monthListmena_trace= go.Scatter(x=monthList_mena, y=mideastnorthaf_data,name="Middle East & North Africa", mode='lines+markers')

fig.add_trace(monthListmena_trace)



# North America

monthList_na  = north_ame['Year'].tolist()

northame_data = north_ame['eventid'].tolist()

monthListna_trace= go.Scatter(x=monthList_na, y=northame_data,name="North America", mode='lines+markers')

fig.add_trace(monthListna_trace)



# South America

monthList_sa  = south_ame['Year'].tolist()

southame_data = south_ame['eventid'].tolist()

monthListsa_trace= go.Scatter(x=monthList_sa, y=southame_data, name="South America", mode='lines+markers')

fig.add_trace(monthListsa_trace)



# South Asia

monthLists_as  = south_asia['Year'].tolist()

southasia_data = south_asia['eventid'].tolist()

monthListsas_trace= go.Scatter(x=monthLists_as, y=southasia_data, name="South Asia", mode='lines+markers')

fig.add_trace(monthListsas_trace)



# Southeast Asia

monthList_seas  = southest_asia['Year'].tolist()

southeastasia_data = southest_asia['eventid'].tolist()

monthListseas_trace= go.Scatter(x=monthList_seas, y=southeastasia_data, name="Southeast Asia", mode='lines+markers')

fig.add_trace(monthListseas_trace)



# Sub-Saharan Africa

monthLists_ua  = sub_af['Year'].tolist()

subaf_data = sub_af['eventid'].tolist()

monthListsubaf_trace = go.Scatter(x=monthLists_ua, y=subaf_data,name="Sub-Saharan Africa", mode='lines+markers')

fig.add_trace(monthListsubaf_trace)



# Western Europe

monthList_we  = west_eu['Year'].tolist()

westeu_data = west_eu['eventid'].tolist()

monthListwe_trace= go.Scatter(x=monthList_we, y=westeu_data,name="Western Europe", mode='lines+markers')

fig.add_trace(monthListwe_trace)



# Central Asia

monthList_cas  = cen_asia['Year'].tolist()

cenasia_data = cen_asia['eventid'].tolist()

monthListcas_trace= go.Scatter(x=monthList_cas, y=cenasia_data, name="Central Asia", mode='lines+markers')

fig.add_trace(monthListcas_trace)





# # Edit the layout

fig.update_layout(title='Successful attack counts by regions',

                   xaxis_title='Year',

                   yaxis_title='Success attacks')





fig.show()
#@title

# select and count events happended successful for each year

year_event_success = df_year_region.apply(count_event_with_year, axis='columns', args=("Middle East & North Africa", )).dropna()

year_success = [ele[0] for ele in year_event_success]

event_success = [ele[1] for ele in year_event_success]



# select and count events happended fail for each year

year_event_fail = df_year_region_fail.apply(count_event_with_year, axis='columns', args=("Middle East & North Africa", )).dropna()

year_fail = [ele[0] for ele in year_event_fail]

event_fail = [ele[1] for ele in year_event_fail]
#@title

fig3 = go.Figure(data=[go.Bar(name='Success', x=year_success, y=event_success, marker_color='red'),

                       go.Bar(name='Fail', x=year_fail, y=event_fail, marker_color='darkturquoise')

                      ])



fig3.update_layout(title='Attacks versus year of Middle East & North Africa region',

                   xaxis_title='Year', yaxis_title='Number'

                   ,barmode='stack')

fig3.show()
#@title

# select and count events happended successful for each year

year_event_success = df_year_region.apply(count_event_with_year, axis='columns', args=("South Asia", )).dropna()

year_success = [ele[0] for ele in year_event_success]

event_success = [ele[1] for ele in year_event_success]



# select and count events happended fail for each year

year_event_fail = df_year_region_fail.apply(count_event_with_year, axis='columns', args=("South Asia", )).dropna()

year_fail = [ele[0] for ele in year_event_fail]

event_fail = [ele[1] for ele in year_event_fail]
#@title

fig4 = go.Figure(data=[go.Bar(name='Success', x=year_success, y=event_success, marker_color='red'),

                       go.Bar(name='Fail', x=year_fail, y=event_fail, marker_color='darkturquoise')

                      ])



fig4.update_layout(title='Attacks versus year of South Asia region',

                   xaxis_title='Year', yaxis_title='Number'

                   ,barmode='stack')

fig4.show()
#@title

# select and count events happended successful for each year

year_event_success = df_year_region.apply(count_event_with_year, axis='columns', args=("North America", )).dropna()

year_success = [ele[0] for ele in year_event_success]

event_success = [ele[1] for ele in year_event_success]



# select and count events happended fail for each year

year_event_fail = df_year_region_fail.apply(count_event_with_year, axis='columns', args=("North America", )).dropna()

year_fail = [ele[0] for ele in year_event_fail]

event_fail = [ele[1] for ele in year_event_fail]
#@title

fig5 = go.Figure(data=[go.Bar(name='Success', x=year_success, y=event_success, marker_color='red'),

                       go.Bar(name='Fail', x=year_fail, y=event_fail, marker_color='darkturquoise')

                      ])



fig5.update_layout(title='Attacks versus year of North America region',

                   xaxis_title='Year', yaxis_title='Number'

                   ,barmode='stack')

fig5.show()
#@title

df_region_at_ty_success = df_success.groupby(['Year', 'Region', 'AttackType'])['eventid'].count()

df_region_at_ty_success = df_region_at_ty_success.reset_index()
#@title

graph_year_type = []



# loop through each attack types

for attack_type in df['AttackType'].unique():

    year_type_success = df_region_at_ty_success.apply(test_success_type, axis='columns', args=("Middle East & North Africa",attack_type, )).dropna()

    year_success = [ele[0] for ele in year_type_success] # year which event happened

    type_success = [ele[1] for ele in year_type_success] # each attack type added in bar graph

    

    # add bar graph to plotly fig

    graph_year_type.append(go.Bar(name=attack_type, y=year_success, x=type_success, orientation='h'))

       

fit3_1 = go.Figure(data=graph_year_type)



# stack all bar graphs versus year

fit3_1.update_layout(title='Success attack events each year in Middle East & North Africa region' ,

                     xaxis_title='Number of attacks',

                     yaxis_title='Year',

                     barmode='stack')

fit3_1.show()
#@title

graph_year_type = []



# loop through each attack types

for attack_type in df['AttackType'].unique():

    year_type_success = df_region_at_ty_success.apply(test_success_type, axis='columns', args=("South Asia",attack_type, )).dropna()

    year_success = [ele[0] for ele in year_type_success] # year which event happened

    type_success = [ele[1] for ele in year_type_success] # each attack type added in bar graph

    

    # add bar graph to plotly fig

    graph_year_type.append(go.Bar(name=attack_type, y=year_success, x=type_success, orientation='h'))

       

fit4_1 = go.Figure(data=graph_year_type)



# stack all bar graphs versus year

fit4_1.update_layout(title='Success attack events each year in South Asia region' ,

                     xaxis_title='Number of attacks',

                     yaxis_title='Year',

                     barmode='stack')

fit4_1.show()
#@title

graph_year_type = []



# loop through each attack types

for attack_type in df['AttackType'].unique():

    year_type_success = df_region_at_ty_success.apply(test_success_type, axis='columns', args=("North America",attack_type, )).dropna()

    year_success = [ele[0] for ele in year_type_success] # year which event happened

    type_success = [ele[1] for ele in year_type_success] # each attack type added in bar graph

    

    # add bar graph to plotly fig

    graph_year_type.append(go.Bar(name=attack_type, y=year_success, x=type_success, orientation='h'))

       

fit5_1 = go.Figure(data=graph_year_type)



# stack all bar graphs versus year

fit5_1.update_layout(title='Success attack events each year in North America region' ,

                     xaxis_title='Number of attacks',

                     yaxis_title='Year',

                     barmode='stack')

fit5_1.show()
#@title

dropcols = ['eventid', 'iyear', 'imonth', 'iday', 'extended','country','country_txt',

        'region','region_txt','provstate','city','latitude','longitude',

        'specificity','vicinity','summary','crit1','crit2','crit3',

        'doubtterr','multiple',

        'success',

        'suicide','attacktype1','attacktype1_txt','targtype1','targtype1_txt',

        'target1','gname','targsubtype1','targsubtype1_txt','corp1',

        'natlty1','natlty1_txt','motive','guncertain1',

        'individual','claimed','weaptype1','weaptype1_txt',

        'weapsubtype1','weapsubtype1_txt','property','ishostkid',

        'scite1','dbsource','INT_LOG','INT_IDEO','INT_MISC','INT_ANY','related',

    'approxdate', 'resolution', 'location', 'alternative', 

       'alternative_txt', 'targtype2', 'targtype2_txt', 'targsubtype2',

        'targsubtype2_txt', 'corp2',

       'target2', 'natlty2', 'natlty2_txt', 'targtype3', 

        'targtype3_txt', 'targsubtype3',

       'targsubtype3_txt', 'corp3', 'target3', 'natlty3', 'natlty3_txt',

       'attacktype2', 'attacktype2_txt', 'attacktype3', 'attacktype3_txt',

       'gsubname', 'gname2','gsubname2', 'gname3', 'gsubname3', 'guncertain2', 'guncertain3',

       'nperps', 'nperpcap',

       'claimmode', 'claimmode_txt', 'claim2', 'claimmode2',

       'claimmode2_txt', 'claim3', 'claimmode3', 'claimmode3_txt',

       'compclaim',

       'scite2', 'scite3', 'weapdetail',

       'propextent', 'propextent_txt', 'propcomment',

       'weaptype2', 'weaptype2_txt', 'weapsubtype2', 'weapsubtype2_txt',

       'weaptype3', 'weaptype3_txt', 'weapsubtype3', 

        'weapsubtype3_txt', 'weaptype4',

       'weaptype4_txt', 'weapsubtype4', 'weapsubtype4_txt',

        'nhours', 'divert', 'kidhijcountry',

       'ransom',  'ransomnote',

       'hostkidoutcome', 'hostkidoutcome_txt', 'addnotes'

      ]
feature_names = [c for c in kdf.columns if c not in dropcols]

feature_names
#@title

X = np.array(kdf[feature_names].fillna(0))  # quick and dirty imputation

y = np.array(kdf['weaptype1_txt'])

ykill = np.array(kdf['nkill'].fillna(0))
#@title

# We create a custom 1-D lens with Isolation Forest

model = ensemble.IsolationForest(random_state=1729)

model.fit(X)

lens1 = model.decision_function(X).reshape((X.shape[0], 1))



# We create another 1-D lens with L2-norm

mapper = km.KeplerMapper(verbose=3)

lens2 = mapper.fit_transform(X, projection="l2norm")



# Combine both lenses to create a 2-D [Isolation Forest, L^2-Norm] lens

clens = np.c_[lens1, lens2]
#@title

# Create the simplicial complex

cgraph = mapper.map(clens,

                   X,

                   cover=km.Cover(n_cubes=15, perc_overlap=0.4),

                   clusterer=sklearn.cluster.KMeans(n_clusters=2,

                                                    random_state=1618033))



# Visualization

#We create two graphs with tooltips weapontype and number of people killed

mapper.visualize(cgraph,

                 path_html="weapontype1.html",

                 title="Global Terrorism Dataset - Weapon type tooltip",

                 color_function=np.array(kdf['weaptype1'].fillna(0)),

                 custom_tooltips=y)



mapper.visualize(cgraph,

                 path_html="weapontype1-nkill.html",

                 title="Global Terrorism Dataset - Number killed tooltip",

                 color_function = ykill,

                 custom_tooltips=ykill)

#@title

HtmlFile = open('../input/gtdtdaresults/l2normweapontype.html', 'r', encoding='utf-8')

cancerread = HtmlFile.read()
#@title

#Find Firearms fraction in weapon type 

ccrs = cancerread.split('<body id="display">')[1].split('<div id="json_colorscale" data-colorscale=')[0].split('Membership information')

ccrss = [x.strip('--\\u003e\\n\\n\\n\\n\\u003chr\\u003e \\u003cbr/\\u003e\\n\\n\\u003ch3\\u003eMembers\\u003c/h3\\u003e\\n\\n') for x in ccrs]



test = [x.split('\\n\\n\\n\\n\\u003cbr\\u003e\\u003cbr\\u003e\\n\\u003chr\\u003e\\u003cbr\\u003e"')[0] for x in ccrss]

members = test[1:]

memberss = [x.split('\\n\\n') for x in members]

firef = []

for i in memberss:

  count=0

     

  for j in i:

    if 'Firearms' == j:

      count += 1

  frac = count/len(i)

  firef.append(frac)

#@title

#Find mean nkill for each node

HtmlFile = open('../input/gtdtdaresults/l2normnkill.html', 'r', encoding='utf-8')



nki = HtmlFile.read()

nkil = nki.split('<body id="display">')[1].split('<div id="json_colorscale" data-colorscale=')[0].split('Membership information')

nkils = [x.strip('--\\u003e\\n\\n\\n\\n\\u003chr\\u003e \\u003cbr/\\u003e\\n\\n\\u003ch3\\u003eMembers\\u003c/h3\\u003e\\n\\n') for x in nkil]



nkilss = [x.split('\\n\\n\\n\\n\\u003cbr\\u003e\\u003cbr\\u003e\\n\\u003chr\\u003e\\u003cbr\\u003e"')[0] for x in nkils]

kmembers = nkilss[1:]

kmemberss = [x.split('\\n\\n') for x in kmembers]

nkillm = []

for i in kmemberss:

     l = []

     for j in i:

       if 'nan' != j and 'an' != j:

        l.append(int(float(j)))

     nkillm.append(sum(l)/len(l))
#@title

plt.subplots(figsize = (10,6))

plt.plot(firef, nkillm, 'bo')

plt.xlabel('Fraction of Firearms')

plt.ylabel('Mean Number of People Killed')

plt.title('Fraction of Firearms and Average Number of People Killed')

plt.show()
coef, pval = stats.pearsonr(nkillm,firef)

print(f"The correlation coefficient is {coef}, the pvalue is {pval}")
#@title

projected_data = mapper.fit_transform(X,

                                      projection=sklearn.manifold.TSNE())



graph = mapper.map(projected_data,

                   clusterer=sklearn.cluster.DBSCAN(eps=0.3, min_samples=15),

                   cover=km.Cover(35, 0.4))



#We create two graphs with tooltips weapontype and number of people killed

mapper.visualize(graph,

                 title="Global Terrorism TSNE DBSCAN",

                 path_html="weapontype1tsne.html",

                 color_function=np.array(kdf['weaptype1'].fillna(0)),

                 custom_tooltips=y)



mapper.visualize(graph,

                 title="Global Terrorism TSNE DBSCAN",

                 path_html="weapontype1nkilltsne.html",

                 color_function=ykill,

                 custom_tooltips=ykill)
#@title

HtmlFile = open('../input/gtdtdaresults/tsneweapontype.html', 'r', encoding='utf-8')

cancerread = HtmlFile.read()

#Find Firearms fraction in weapon type 

ccrs = cancerread.split('<body id="display">')[1].split('<div id="json_colorscale" data-colorscale=')[0].split('Membership information')

ccrss = [x.strip('--\\u003e\\n\\n\\n\\n\\u003chr\\u003e \\u003cbr/\\u003e\\n\\n\\u003ch3\\u003eMembers\\u003c/h3\\u003e\\n\\n') for x in ccrs]



test = [x.split('\\n\\n\\n\\n\\u003cbr\\u003e\\u003cbr\\u003e\\n\\u003chr\\u003e\\u003cbr\\u003e"')[0] for x in ccrss]

members = test[1:]

memberss = [x.split('\\n\\n') for x in members]

fireftsne = []

for i in memberss:

  count=0

     

  for j in i:

    if 'Firearms' == j:

      count += 1

  frac = count/len(i)

  fireftsne.append(frac)
#@title

HtmlFile = open('../input/gtdtdaresults/tsnenkill.html', 'r', encoding='utf-8')



nki = HtmlFile.read()

nkil = nki.split('<body id="display">')[1].split('<div id="json_colorscale" data-colorscale=')[0].split('Membership information')

nkils = [x.strip('--\\u003e\\n\\n\\n\\n\\u003chr\\u003e \\u003cbr/\\u003e\\n\\n\\u003ch3\\u003eMembers\\u003c/h3\\u003e\\n\\n') for x in nkil]



nkilss = [x.split('\\n\\n\\n\\n\\u003cbr\\u003e\\u003cbr\\u003e\\n\\u003chr\\u003e\\u003cbr\\u003e"')[0] for x in nkils]

kmembers = nkilss[1:]

kmemberss = [x.split('\\n\\n') for x in kmembers]

nkillmtsne = []

for i in kmemberss:

     l = []

     for j in i:

       if 'nan' != j and 'an' != j:

        l.append(int(float(j)))

     nkillmtsne.append(sum(l)/len(l))
#@title

plt.subplots(figsize = (10,6))

plt.plot(fireftsne, nkillmtsne, 'bo')

plt.xlabel('Fraction of Firearms')

plt.ylabel('Mean Number of People Killed')

plt.title('Fraction of Firearms and Average Number of People Killed')

plt.show()
coeft, pvalt = stats.pearsonr(nkillmtsne,fireftsne)

print(f"The correlation coefficient is {coeft}, the pvalue is {pvalt}")
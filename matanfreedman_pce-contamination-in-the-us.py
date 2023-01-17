import pandas as pd 

import numpy as np

import os

from tqdm import tqdm_notebook

from ipywidgets import interact_manual





import plotly.offline as py

import plotly.graph_objs as go

import plotly.figure_factory as ff

py.init_notebook_mode(connected=True)



pd.options.display.max_columns = 110

pd.options.display.max_rows = 300



import warnings

warnings.simplefilter('ignore')
# # Next few kernels will quickly clean the data:

# dtypes

dtypes = {

'YEAR':object,

# 'TRI_FACILITY_ID': object,

# 'FRS_ID':object,

# 'FACILITY_NAME':object,

# # 'STREET_ADDRESS':object,

'CITY':object,

'COUNTY':object,

'ST':'object',

# 'ZIP':object,

# 'BIA_CODE':'category',

# 'TRIBE':object,

'LATITUDE':object,

'LONGITUDE':object,

# 'FEDERAL_FACILITY':'category',

# 'INDUSTRY_SECTOR_CODE':'category',

'INDUSTRY_SECTOR':'category',

# 'PRIMARY_SIC':'category',

# 'SIC_2':'category',

# 'SIC_3':'category',

# 'SIC_4':'category',

# 'SIC_5':'category',

# 'SIC_6':'category',

# 'PRIMARY_NAICS':'category',

# 'NAICS_2':'category',

# 'NAICS_3':'category',

# 'NAICS_4':'category',

# 'NAICS_5':'category',

# 'NAICS_6':'category',

# 'DOC_CTRL_NUM':object,

'CHEMICAL':'object',

# 'CAS_#/COMPOUND_ID':object,

# 'SRS_ID':object,

# 'CLEAR_AIR_ACT_CHEMICAL':'category',

# 'CLASSIFICATION':'category',

# 'METAL':'category',

# 'METAL_CATEGORY':int,

# 'CARCINOGEN':'category',

# 'FORM_TYPE':'category',

'UNIT_OF_MEASURE':object,

# '5.1_FUGITIVE_AIR':object,

# '5.2_STACK_AIR':float,

# '5.3_WATER':float,

# '5.4_UNDERGROUND':float,

# '5.4.1_UNDERGROUND_CLASS_I':float,

# '5.4.2_UNDERGROUND_CLASS_II-V':float,

# '5.5.1_LANDFILLS':float,

# '5.5.1A_RCRA_C_LANDFILLS':float,

# '5.5.1B_OTHER_LANDFILLS':float,

# '5.5.2_LAND_TREATMENT':float,

# '5.5.3_SURFACE_IMPOUNDMENT':float,

# '5.5.3A_RCRA_C_SURFACE_IMP.':float,

# '5.5.3B_Other_SURFACE_IMP.':float,

# '5.5.4_OTHER_DISPOSAL':float,

# 'ON-SITE_RELEASE_TOTAL':float,

# '6.1_POTW-TRANSFERS_FOR_RELEASE':float,

# '6.1_POTW-TRANSFERS_FOR_TREATM.':float,

# '6.1_POTW-TOTAL_TRANSFERS':float,

# '6.2_M10':float,

# '6.2_M41':float,

# '6.2_M62':float,

# '6.2_M71':float,

# '6.2_M81':float,

# '6.2_M82':float,

# '6.2_M72':float,

# '6.2_M63':float,

# '6.2_M66':float,

# '6.2_M67':float,

# '6.2_M64':float,

# '6.2_M65':float,

# '6.2_M73':float,

# '6.2_M79':float,

# '6.2_M90':float,

# '6.2_M94':float,

# '6.2_M99':float,

# 'OFF-SITE_RELEASE_TOTAL':float,

# '6.2_M20':float,

# '6.2_M24':float,

# '6.2_M26':float,

# '6.2_M28':float,

# '6.2_M93':float,

# 'OFF-SITE_RECYCLED_TOTAL':float,

# '6.2_M56':float,

# '6.2_M92':float,

# 'OFF-SITE_RECOVERY_TOTAL':float,

# '6.2_M40':float,

# '6.2_M50':float,

# '6.2_M54':float,

# '6.2_M61':float,

# '6.2_M69':float,

# '6.2_M95':float,

# 'OFF-SITE_TREATED_TOTAL':float,

'TOTAL_RELEASES':object,

# '8.1_RELEASES':object,

# '8.1A_ON-SITE_CONTAINED_REL.':float,

# '8.1B_ON-SITE_OTHER_RELEASES':float,

# '8.1C_OFF-SITE_CONTAINED_REL.':float,

# '8.1D_OFF-SITE_OTHER_RELEASES':float,

# '8.2_ENERGY_RECOVERY_ON-SITE':float,

# '8.3_ENERGY_RECOVERY_OFF-SITE':float,

# '8.4_RECYCLING_ON-SITE':float,

# ' 8.5_RECYCLING_OFF-SITE':float,

# '8.6_TREATMENT_ON-SITE':float,

# # '8.7_TREATMENT_OFF-SITE':float,

# 'PROD._WASTE_(8.1_THRU_8.7)':float

# '8.8_ONE-TIME_RELEASE':float,

# 'PROD_RATIO_OR_ACTIVITY':object,

# '8.9_PRODUCTION_RATIO':float,

# 'PARENT_COMPANY_NAME':object,

# 'PARENT_COMPANY_DB_NUMBER':object,

}
%%time

data_1 = pd.read_csv('../input/basic_data_files.csv', nrows=2548770, low_memory=True, dtype=dtypes, usecols=dtypes.keys())

data_2 = pd.read_csv('../input/basic_data_files.csv', skiprows=2548771, low_memory=True, dtype=dtypes, usecols=dtypes.keys())



data_clean = data_1.append(data_2, ignore_index=True)

data_clean = data_clean.drop(index=data_clean[data_clean.YEAR == 'YEAR'].index).reset_index(drop=True)



# redefine data types to speed up manipulation:

dtypes = {'YEAR': int, 'ST': 'category', 'COUNTY': 'category', 'CITY': 'category', 'CHEMICAL':'category', 'TOTAL_RELEASES': float,

         'LATITUDE': float, 'LONGITUDE': float, 'INDUSTRY_SECTOR':'category'}

data = data_clean.astype(dtypes)
data.head()
years = [i for i in range(1987, 2016+1)]



# create steps for slider

def get_steps(years):

    steps = []

    for i in range(0,len(years)):

        step = dict(method = "restyle",

                    args = ["visible", [False]*len(years)],

                    label = years[i]) 

        step['args'][1][i] = True

        steps.append(step)

    return steps



# Sliders layout:

def get_sliders(steps):

    sliders = [dict(active = 10,

                    currentvalue = {"prefix": "Year: "},

                    pad = {"t": 50},

                    steps = steps)]

    return sliders



# get dataframe of only selected chemical 

def get_cont_sites(chemicals): 

    return data[data.CHEMICAL.isin([chemicals])]



# make dataframe with values for choropleth colors

def get_state_counts(cont_sites):

    # fill each year column with the number of contaminated sites in each state

    state_counts = pd.DataFrame(index=data[data.YEAR == 2016].ST.value_counts().sort_index().index.tolist(), columns=years)

    for i in years: 

        totals_temp = cont_sites[cont_sites.YEAR == i][['ST', 'TOTAL_RELEASES']].groupby(['ST']).sum().unstack()

        state_counts[i] = pd.DataFrame(index=totals_temp.index.levels[1].tolist(), data=totals_temp.values, columns=[i])



    state_counts.loc['norm'] = state_counts.max().max()

    return state_counts







steps = get_steps(years)

sliders = get_sliders(steps)



def plot_choro(Chemical='TETRACHLOROETHYLENE'):

    cont_sites = get_cont_sites(Chemical)

    state_counts = get_state_counts(cont_sites)



    # create a list and loop through every year, store the trace in data_bal and then update with a new year

    data_bal = []

    for i in years:

        data_upd = [dict(type='choropleth',

                         name=i,

                          colorscale = 'Reds',

                          reversescale=False,

                          locations = state_counts[i].index,

                          z = state_counts[i].values,

                          locationmode = 'USA-states',

                          colorbar = dict(title='Pounds Released Per Year')

                        )

                   ]



        data_bal.extend(data_upd)





    # Set the layout

    layout = dict(title = 'Total Released ' + Chemical + ' In Pounds',

                  geo = dict(scope='usa',

                             projection=dict( type='albers usa')),

                  sliders = sliders)



    fig = dict(data=data_bal, layout=layout)

    py.iplot(fig)
plot_choro(Chemical='TETRACHLOROETHYLENE')
data_2016 = data[(data['CHEMICAL'] == 'TETRACHLOROETHYLENE') & (data['YEAR'] == 2016)]

data_2016 = data_2016.groupby(['LATITUDE', 'LONGITUDE', 'INDUSTRY_SECTOR']).agg({'TOTAL_RELEASES': 'sum'}).reset_index()
heat_data = [[row['LATITUDE'],row['LONGITUDE'], row['TOTAL_RELEASES']] for index, row in data_2016.iterrows()]
# folium heatmap:

import folium

from folium.plugins import HeatMap

import branca.colormap as cm



map_demo = folium.Map([39.09973, -94.57857], zoom_start=4)



# #define colorbar

# steps = 20

# color_map=cm.linear.viridis.scale(0,1).to_step(steps)

# gradient_map=dict()

# for i in range(steps):

#     gradient_map[1/steps*i] = color_map.rgb_hex_str(1/steps*i)



# HeatMap(heat_data, gradient = gradient_map).add_to(map_demo)



# color_map.caption = 'Colorbar'

# map_demo.add_child(color_map)



#Function to change colors

def color_change(releases):

    max_ = max(data['TOTAL_RELEASES'])

    if(releases < max_ / 3):

        return('green')

    elif(max_ <= releases < max_):

        return('orange')

    else:

        return('red')



for lat, lon, weight, sector in zip(data_2016['LATITUDE'], data_2016['LONGITUDE'], data_2016['TOTAL_RELEASES'], data_2016['INDUSTRY_SECTOR']):

    folium.CircleMarker(location=[lat, lon], radius = 9, popup=sector + ": " + str(weight)+" lbs", 

                        fill_color=color_change(weight)).add_to(map_demo)





map_demo
# folium heatmap:

import folium

from folium.plugins import HeatMap

import branca.colormap as cm



data_HM = data[(data['CHEMICAL'] == 'TETRACHLOROETHYLENE') & (data['YEAR'] == 2016)]

data_HM = data_HM.groupby(['LATITUDE', 'LONGITUDE']).agg({'TOTAL_RELEASES': 'sum'}).reset_index()



map_HM = folium.Map([39.09973, -94.57857], zoom_start=4)



heat_data = [[row['LATITUDE'],row['LONGITUDE'], row['TOTAL_RELEASES']] for index, row in data_HM.iterrows()]



HeatMap(heat_data).add_to(map_HM)

map_HM
TCE_data = data[(data['CHEMICAL'] == 'TETRACHLOROETHYLENE')]

TCE_data = TCE_data.groupby(['YEAR', 'LATITUDE', 'LONGITUDE']).agg({'TOTAL_RELEASES': 'sum'}).reset_index()

TCE_data = TCE_data.sort_values(by='YEAR', axis=0, ascending=False)



#year 1987:2016

heat_data = [[[row['LATITUDE'],row['LONGITUDE'], row['TOTAL_RELEASES']] for index, row in TCE_data[TCE_data['YEAR'] == i].iterrows()] for i in range(1987, 2017)]



from folium import plugins

time_map_HM = folium.Map([39.09973, -94.57857], zoom_start=4)



# Plot it on the map

hm = plugins.HeatMapWithTime(heat_data,auto_play=True,max_opacity=0.8)

hm.add_to(time_map_HM)

time_map_HM
#plot chlorinated ethylene releases over time

temp = data.groupby(['CHEMICAL', 'YEAR']).agg({'TOTAL_RELEASES': 'sum'})

temp = temp.reset_index()



traces = []

for i in ['TETRACHLOROETHYLENE', 'TRICHLOROETHYLENE', '1,2-DICHLOROETHYLENE',

         'VINYL CHLORIDE']:

    temp_ = temp[temp['CHEMICAL'] == i]

    trace = go.Scatter(x=temp_['YEAR'], y=temp_['TOTAL_RELEASES'], name=i)

    traces.append(trace)

    

layout = go.Layout(title='Chlorinated Ethylene Reported Releases in the US',

                  yaxis=dict(title='Pounds'), xaxis=dict(title='Year'))    

fig = go.Figure(data = traces, layout=layout)

py.iplot(fig)
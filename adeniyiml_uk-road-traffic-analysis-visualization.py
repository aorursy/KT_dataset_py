import math 

import calendar

import pandas as pd

import datetime





import geopandas as gpd

from shapely.geometry import Point

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



from plotly.subplots import make_subplots

import plotly.graph_objects as go

import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# plotly

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.express as px
##TRAFFIC DATASETS ###



uk_traffic = pd.read_csv('../input/uk-traffic-data/dft_traffic_counts_aadf.csv')
#size of the df

uk_traffic.shape
#check for missing values

uk_traffic.isnull().sum()
##lets drop the two columns with the most missing values

uk_traffic = uk_traffic.drop(['start_junction_road_name', 'end_junction_road_name'], axis=1)
uk_traffic.dropna(inplace = True)
uk_traffic.head()
uk_traffic['AllVehicles'] = uk_traffic['all_hgvs'] + uk_traffic['all_motor_vehicles'] + uk_traffic['pedal_cycles']
###calculate miles driven, the miles driven per year by a vehicle == Volume of traffic for that particular vehicle on a road



vehicles = ['pedal_cycles', 'two_wheeled_motor_vehicles', 'cars_and_taxis', 'buses_and_coaches', 'lgvs', 

            'hgvs_2_rigid_axle', 'hgvs_3_rigid_axle', 

            'hgvs_3_or_4_articulated_axle', 'hgvs_4_or_more_rigid_axle', 

            'hgvs_5_articulated_axle', 'hgvs_6_articulated_axle', 'all_hgvs', 'all_motor_vehicles', 'AllVehicles']
for vehicle in vehicles:

    uk_traffic[vehicle + 'miles_driven'] = uk_traffic[vehicle] * uk_traffic['link_length_miles'] * 365
uk_traffic['road_category'].value_counts()
###Note



##the AADF values for MB and MCU are Zero, this means they are NOT USEFUL for our traffic analyses



## This means will drop the both of them
uk_traffic = uk_traffic[(uk_traffic['road_category'] != 'MCU') & (uk_traffic['road_category'] != 'MB')]
uk_traffic['road_type'].value_counts()
rd_category = uk_traffic.road_category.unique().tolist()

print(rd_category)
road_category_description = ['Class A Principal Road', 'M or Class A Trunk Motorway', 'Class A Trunk Road', 

                             'M or Class A Principal Motorway']
for road in range(len(rd_category)):

    uk_traffic.loc[uk_traffic['road_category'] == rd_category[road], 'road_category'] = road_category_description[road]
uk_traffic['northing'].max()
uk_traffic.shape
uk_traffic = uk_traffic.loc[uk_traffic['northing']!=uk_traffic['northing'].max()]
uk_gpd = uk_traffic.copy()
#uk 2018 road network map

gb_shape = gpd.read_file('../input/shape-file/2018-MRDB-minimal.shp')
sns.set_context('talk')

ax = gb_shape.plot(figsize=(25,15), color='crimson')

ax.set(title='Road Networks in UK, 2018')

ax.set_axis_off()
#convert the uk_traffic (uk_gpd) df to GeopandasDf

points = uk_gpd.apply(lambda row: Point(row.easting, row.northing), axis=1)

uk_gpd = gpd.GeoDataFrame(uk_gpd, geometry=points)
#lets plot points where AADF were measured based on UK regions

ax = gb_shape.plot(figsize=(30,15), color='black', linewidth=0.75)

uk_gpd.plot(column='region_name', ax=ax, markersize=60, legend = True, cmap='hsv_r', edgecolor='white')

leg = ax.get_legend()

leg.set_bbox_to_anchor((1.1,0.5))

leg.set_title('Regions')

leg.set_frame_on(False)

ax.set_axis_off()

ax.set_title('Road Networks (points) where AADF measurement were taken between 2000 - 2018 in UK', fontsize=16, pad=10)
####To start with let extract top 10 busiest roads

##sort starting with the busiest road based on its average vehicle mile per year



rd_agg = uk_traffic.groupby(['year', 

                             'road_name']).AllVehiclesmiles_driven.agg(['count', 

                                                                        'sum', 'mean']).sort_values(by = 'mean', ascending = False)

rd_agg.reset_index(inplace = True)
rd_agg_100 = rd_agg[:100].copy()
#take just the first 10 roads

print(rd_agg_100['road_name'].unique().tolist())
#extract top 10 busiest Highway authority based on average vehicle mile per year



rd_agg = uk_traffic.groupby(['year', 'local_authority_name']).AllVehiclesmiles_driven.agg(['count', 'sum', 'mean']).sort_values(by = 'sum', ascending = False)

rd_agg.reset_index(inplace = True)
rd_agg_100 = rd_agg[:140].copy()

#take just the first 10 highway authority

print(rd_agg_100['local_authority_name'].unique().tolist())
##### EXTRACT VEHICLE TRAFFIC FLOW  BASED ON DIFFERENT CRITERIAS WRITTEN ABOVE #####

#### NOTE: The Traffic flow / trends are based on the volume of traffic i.e. miles driven by each vehicle per year ####



def vehicles_traffic_flow(df, vehicle, column):

    

    if column == 'road_name':

        loc_agg = uk_traffic.groupby(['year', column])[vehicle].agg(['count', 'sum', 'mean']).reset_index()

        loc_names = ['A1(M)', 'M25', 'A1(T)', 'M26', 'M5', 'M1', 'M40', 'M3', 'M6', 'M23']

    elif column == 'local_authority_name':

        loc_agg = uk_traffic.groupby(['year', column])[vehicle].agg(['count', 'sum', 'mean']).reset_index()

        loc_names = ['Kent', 'Hampshire', 'Surrey', 'Essex', 'Hertfordshire', 

                     'Lancashire', 'Staffordshire', 'Warwickshire', 'Northamptonshire', 'Cheshire']

    else:

        loc_agg = df.groupby(['year', column])[vehicle].agg(['count', 'sum', 'mean']).reset_index()

        loc_names = loc_agg[column].unique().tolist()

        

    list_traces = []

    colors = ['lightslategray', 'crimson', 'darkcyan', 'darkgoldenrod', 'cornsilk', 'turquoise', 'limegreen', \

              'darkorchid', 'palevioletred', 'forestgreen', 'silver', 'lightsteelblue']

    for name in range(len(loc_names)):

        name_agg = loc_agg[loc_agg[column] == loc_names[name]]

        data_agg = go.Scatter(x = name_agg['year'], y= name_agg['mean'], mode="lines+markers", name= loc_names[name], 

                              line= dict(color = colors[name], width=2.5))

        

        list_traces.append(data_agg)

    if column == 'road_category':

        context = 'on Major Road Classes in UK'

    elif column == 'region_name':

        context = 'in UK Regions'

    elif column == 'road_name':

        context = 'on the 10 Busiest Roads in UK'

    elif column == 'local_authority_name':

        context = 'on the 10 Busiest Highway Authorities in UK'

    

    if vehicle == 'pedal_cyclesmiles_driven':

        content = 'Pedal Cycles'

    elif vehicle == 'all_motor_vehiclesmiles_driven':

        content = 'Motor Vehicles'

    elif vehicle == 'all_hgvsmiles_driven':

        content = 'Heavy Goods Vehicles (Lorries)'

    elif vehicle == 'cars_and_taxismiles_driven':

        content = 'Cars and Taxis'

    elif vehicle == 'two_wheeled_motor_vehiclesmiles_driven':

        content = 'Motor Cycles'

    elif vehicle == 'buses_and_coachesmiles_driven':

        content = 'Buses and Coaches'

    elif vehicle == 'AllVehiclesmiles_driven':

        content = 'All vehicles'

    elif vehicle == 'lgvsmiles_driven':

        content = 'Vans'

    

        

    layout = go.Layout(title='<b> '+ content +' traffic trends ' + context + ', 2000 - 2018 <b>', width=1000, 

                       height=600, xaxis=dict(title='<b> Year <b>',titlefont=dict(size=16, color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue')), 

                       yaxis=dict(title='<b> Miles Driven per Year <b>',titlefont=dict(size=16,color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue')))

    fig = go.Figure(data=list_traces, layout = layout)

    fig.update_xaxes(dtick=2)

    iplot(fig)
### All Vehicles --- UK Regions###

vehicles_traffic_flow(uk_traffic, 'AllVehiclesmiles_driven', 'region_name')
### All heavy good vehicles --- UK Regions###

vehicles_traffic_flow(uk_traffic, 'all_hgvsmiles_driven', 'region_name')
### All motor vehicles vehicles --- UK Regions###

vehicles_traffic_flow(uk_traffic, 'all_motor_vehiclesmiles_driven', 'region_name')
### Cars and Taxis --- UK Regions###

vehicles_traffic_flow(uk_traffic, 'cars_and_taxismiles_driven', 'region_name')
### Motor Cycles --- UK Regions###

vehicles_traffic_flow(uk_traffic, 'two_wheeled_motor_vehiclesmiles_driven', 'region_name')
### Pedal Cycles --- UK Regions###

vehicles_traffic_flow(uk_traffic, 'pedal_cyclesmiles_driven', 'region_name')
### Vans --- UK Regions###

vehicles_traffic_flow(uk_traffic, 'lgvsmiles_driven', 'region_name')
### Buses and Coaches --- UK Regions###

vehicles_traffic_flow(uk_traffic, 'buses_and_coachesmiles_driven', 'region_name')
### All Vehicles --- Major Road Classes###

vehicles_traffic_flow(uk_traffic, 'AllVehiclesmiles_driven', 'road_category')
### All Motor Vehicles --- Major Road Classes###

vehicles_traffic_flow(uk_traffic, 'all_motor_vehiclesmiles_driven', 'road_category')
### Heavy Goods Vehicles --- Major Road Classes###

vehicles_traffic_flow(uk_traffic, 'all_hgvsmiles_driven', 'road_category')
### Cars and Taxis --- Major Road Classes###

vehicles_traffic_flow(uk_traffic, 'cars_and_taxismiles_driven', 'road_category')
### Motor Cycles --- Major Road Classes###

vehicles_traffic_flow(uk_traffic, 'two_wheeled_motor_vehiclesmiles_driven', 'road_category')
### Pedal Cycles --- Major Road Classes###

vehicles_traffic_flow(uk_traffic, 'pedal_cyclesmiles_driven', 'road_category')
### Vans --- Major Road Classes###

vehicles_traffic_flow(uk_traffic, 'lgvsmiles_driven', 'road_category')
### Buses and Coaches --- Major Road Classes###

vehicles_traffic_flow(uk_traffic, 'buses_and_coachesmiles_driven', 'road_category')
### All Vehicles --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'AllVehiclesmiles_driven', 'road_name')
### All Motor Vehicles --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'all_motor_vehiclesmiles_driven', 'road_name')
### Cars and Taxis --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'cars_and_taxismiles_driven', 'road_name')
### Vans --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'lgvsmiles_driven', 'road_name')
### Motor Cycles --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'two_wheeled_motor_vehiclesmiles_driven', 'road_name')
### Heavy Good Vehicles --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'all_hgvsmiles_driven', 'road_name')
### All Vehicles --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'AllVehiclesmiles_driven', 'local_authority_name')
### Heavy Goods Vehicles --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'all_hgvsmiles_driven', 'local_authority_name')
### Motor Vehicles --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'all_motor_vehiclesmiles_driven', 'local_authority_name')
### Cars and Taxis --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'cars_and_taxismiles_driven', 'local_authority_name')
### Vans --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'lgvsmiles_driven', 'local_authority_name')
### Pedal Cycles --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'pedal_cyclesmiles_driven', 'local_authority_name')
### Motor Cycles --- 10 busiest roads###

vehicles_traffic_flow(uk_traffic, 'two_wheeled_motor_vehiclesmiles_driven', 'local_authority_name')
#### CALCULATE AVERAGE VOLUME OF TRAFFIC FOR EACH MAJOR ROAD CATEGORY ####

#### AND CALCULATE AADF FOR EACH REGION IN UK; FOR 10 BUSIEST ROADS AND HIGHWAYS ####



def average_volume_traffic_aadf(df, column, color_scale):

    

    if column == 'road_name':

        vol_agg = df.groupby(['year', \

                              column])['AllVehicles'].agg(['count', 'sum', 'mean']).sort_values(by = 'mean', ascending = False).reset_index()

        vol_agg = vol_agg[:130].copy()

        vol_agg.sort_values(by = 'mean', ascending = True, inplace = True)

    elif column == 'local_authority_name':

        vol_agg = df.groupby(['year', \

                              column])['AllVehicles'].agg(['count', 'sum', 'mean']).sort_values(by = 'sum', ascending = False).reset_index()

        vol_agg = vol_agg[:130].copy()

        vol_agg.sort_values(by = 'mean', ascending = True, inplace = True)

    elif column == 'road_category':

        vol_agg = df.groupby(['year', \

                              column])['AllVehiclesmiles_driven'].agg(['count', 'sum', 'mean']).sort_values(by = 'mean').reset_index()

    

    elif column == 'region_name':

        vol_agg = df.groupby(['year', \

                              column])['AllVehicles'].agg(['count', 'sum', 'mean']).sort_values(by = 'mean').reset_index()

    

    

    #Data Visualization

    

    fig = px.bar(vol_agg, x= 'mean', y= column, color='year', barmode = 'group', orientation = 'h', 

                 labels={'year':'<b> Years <b>'}, width=1000, height=700, 

                 color_continuous_scale= color_scale)

    

    if column == 'road_category':

        content = 'Average Volume of Traffic for each Road Category in UK'

        context = 'Road Category'

        x_value = 'Volume of Traffic (Vehicle Miles for each year are stacked on top of each other)'

    elif column == 'region_name':

        content = 'Average number of Vehicles that traverse (all roads in) UK Regions per day'

        context = 'Regions'

        x_value = 'Annual Average Daily Flow (AADF for each year are stacked on top of eachother)'

    elif column == 'road_name':

        content = 'Average number of Vehicles that traverse 10 busiest roads in UK per day'

        context = 'Roads'

        x_value = 'Annual Average Daily Flow (AADF for each year are stacked on top of eachother)'

    elif column == 'local_authority_name':

        content = 'Average number of Vehicles that traverse 10 busiest Highway Authorities in UK per day'

        context = 'Highway Authority'

        x_value = 'Annual Average Daily Flow (AADF for each year are stacked on top of eachother)'

        

        

    fig.update_layout(title='<b> ' + content + ', 2000 - 2018 <b>', 

                      xaxis=dict(title='<b> ' + x_value + ' <b>',titlefont=dict(size=16, color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue')), 

                      yaxis=dict(title='<b> ' + context + ' <b>',titlefont=dict(size=16,color='#7f7f7f'), tickfont=dict(size=15, color='darkslateblue'))) 

    

    

    iplot(fig)
#plot average volume of traffic for each year --> Major Road Category



average_volume_traffic_aadf(uk_traffic, 'road_category', px.colors.sequential.Plasma)
#plot AADF yearly for each Region in UK



average_volume_traffic_aadf(uk_traffic, 'region_name', px.colors.diverging.Earth)
#plot AADF yearly for 10 busiest roads in UK



average_volume_traffic_aadf(uk_traffic, 'road_name', px.colors.sequential.Agsunset)
#plot AADF yearly for 10 busiest highway authority in UK



average_volume_traffic_aadf(uk_traffic, 'local_authority_name', px.colors.cyclical.Twilight)
### VEHICLE TRAFFIC TRENDS AND MILES COVERED BY EACH VEHICLES BETWEEN 2000 AND 2018 ###



def vehicle_trends(df, vehicle, column, color):

    

    veh_agg = df.groupby(['road_category'])[vehicle].agg(['count', 'sum', 'mean']).reset_index()

    vehyr_agg = df.groupby(['year'])[vehicle].agg(['count', 'sum', 'mean']).reset_index()

    ##average growth rate

    average_growth_rate = np.power(((vehyr_agg.loc[vehyr_agg.shape[0]-1, 'sum']) \

                                    / vehyr_agg.loc[0, 'sum']), 1/vehyr_agg.shape[0]) - 1

    

    avg_rate = round(average_growth_rate * 100, 2)

    avg_value = vehyr_agg['mean'].min()

    

    def line_plot(df, column, avg_rate, avg_value):

        

        if (column == 'Heavy Goods Vehicle') or (column == 'Buses and Coaches') or (column == 'Motor Cycles'):

            y_value = 1.1

        else:

            y_value = 0.8

        

        annotations = []

        fig = go.Figure()

        fig.add_trace(go.Scatter(x = df['year'], y= df['mean'], mode="lines", name= column + ' trend', 

                                 line= dict(color = 'rgb(67,67,67)', width=2.5)))

        

        annotations.append(dict(xref='paper', yref='paper', x=0.35, y=y_value, 

                                              xanchor='center', yanchor='top', 

                                              text= '<b>' + column + ' traffic trend in UK from 2000 - 2018 <b>', 

                                              font=dict(family='Arial', 

                                                        size=16, color='darkslategray'), 

                                              showarrow=False))

        annotations.append(dict(xref='paper', x=0.25, y=avg_value,

                                  xanchor='left', yanchor='middle',

                                  text='Average annual growth rate {}%'.format(avg_rate),

                                  font=dict(family='Arial',

                                            size=15),

                                  showarrow=False))

        

        fig.update_layout(xaxis=dict(title='<b> Year <b>',titlefont=dict(size=14, color='#7f7f7f'), 

                                     showline=True, showgrid=False, showticklabels=True, linecolor='rgb(204, 204, 204)', linewidth=2, 

                                     ticks='outside', tickfont=dict(family='Arial', size=14, color='rgb(82, 82, 82)')), 

                          yaxis=dict(title='<b> Miles Driven per year <b>',titlefont=dict(size=14,color='#7f7f7f'), 

                                     showgrid=False, zeroline=False, showline=True, showticklabels=True, 

                                     tickfont=dict(family='Arial', size=14, color='rgb(82, 82, 82)')), 

                          autosize=False, margin=dict(autoexpand=False, l=100, r=20, t=110), plot_bgcolor='white', 

                         annotations = annotations)

        

        fig.show()

        

        

    def pie_plot(df, column, color):

        

        vehicle_labels = df.road_category.unique().tolist()

        vehicle_values = df['sum'].values.tolist()

        vehicle_sum = int(sum(vehicle_values))

        def roundup(x):

            if len(str(x)) >= 13:

                num = str(int(math.ceil(x / 1000000000000.0)))

                num = num + ' Trillion'

            else:

                num = str(int(math.ceil(x / 1000000000.0)))

                num = num + ' Billion'

            return num

        vehicle_sum = roundup(vehicle_sum)

        

        fig = go.Figure()

        

        fig.add_trace(go.Pie(labels= vehicle_labels, values= vehicle_values, name="", 

                             hole=.7, textinfo='label+percent', insidetextorientation='radial', marker_colors= color))

        

        fig.update_layout(title_text="<b> " + column + " miles covered <b>", 

                          annotations=[dict(text= vehicle_sum + ' miles covered', x=0.5, y=0.5, font_size=18, showarrow=False)])

        

        fig.show()

        

    line_plot(vehyr_agg, column, avg_rate, avg_value)

    pie_plot(veh_agg, column, color)
# Define color sets of paintings -- Plotly selections

night_colors = ['rgb(56, 75, 126)', 'rgb(18, 36, 37)', 'rgb(34, 53, 101)',

                'rgb(36, 55, 57)', 'rgb(6, 4, 4)']

sunflowers_colors = ['rgb(177, 127, 38)', 'rgb(205, 152, 36)', 'rgb(99, 79, 37)',

                     'rgb(129, 180, 179)', 'rgb(124, 103, 37)']

irises_colors = ['rgb(33, 75, 99)', 'rgb(79, 129, 102)', 'rgb(151, 179, 100)',

                 'rgb(175, 49, 35)', 'rgb(36, 73, 147)']

cafe_colors =  ['rgb(146, 123, 21)', 'rgb(177, 180, 34)', 'rgb(206, 206, 40)',

                'rgb(175, 51, 21)', 'rgb(35, 36, 21)']

random_colors = ['rgb(67,67,67)', 'rgb(115,115,115)', 'rgb(49,130,189)', 'rgb(189,189,189)']
# motor vehicle traffic trend and miles covered btw 2000 - 2018



vehicle_trends(uk_traffic, 'all_motor_vehiclesmiles_driven', 'Motor Vehicles', irises_colors)
# All vehicles traffic trend and miles covered btw 2000 - 2018



vehicle_trends(uk_traffic, 'AllVehiclesmiles_driven', 'All Vehicles', cafe_colors)
# heavy goods vehicles traffic trend and miles covered btw 2000 - 2018



vehicle_trends(uk_traffic, 'all_hgvsmiles_driven', 'Heavy Goods Vehicle', night_colors)
# heavy goods vehicles traffic trend and miles covered btw 2000 - 2018



vehicle_trends(uk_traffic, 'cars_and_taxismiles_driven', 'Cars and Taxis', sunflowers_colors)
# pedal cycles traffic trend and miles covered btw 2000 - 2018



vehicle_trends(uk_traffic, 'pedal_cyclesmiles_driven', 'Pedal Cycles', irises_colors)
# Buses and Coaches vehicles traffic trend and miles covered btw 2000 - 2018



vehicle_trends(uk_traffic, 'buses_and_coachesmiles_driven', 'Buses and Coaches', random_colors)
# Vans traffic trend and miles covered btw 2000 - 2018



vehicle_trends(uk_traffic, 'lgvsmiles_driven', 'Vans', irises_colors)
# Motor Cycles traffic trend and miles covered btw 2000 - 2018



vehicle_trends(uk_traffic, 'two_wheeled_motor_vehicles', 'Motor Cycles', cafe_colors)
uk_traffic.columns
def growth_rate(df, vehicle):

    

    vehyr_agg = df.groupby(['year'])[vehicle].agg(['count', 'sum', 'mean']).reset_index()

    

    #growth rate at the end of 2018

    growth_rate_18 = round(((vehyr_agg.loc[vehyr_agg.shape[0]-1, 'sum'] \

                             - vehyr_agg.loc[0, 'sum']) / vehyr_agg.loc[0, 'sum']) * 100, 2)

    

    #calculating growth rate over regular time period

    #Average annual growth rate btw 2000 - 2018

    average_growth_rate = np.power(((vehyr_agg.loc[vehyr_agg.shape[0]-1, 'sum']) \

                                    / vehyr_agg.loc[0, 'sum']), 1/vehyr_agg.shape[0]) - 1

    

    avg_rate = round(average_growth_rate * 100, 2)

    

    #print

    print('Growth rate at the end of 2018:', str(growth_rate_18) + '%')

    print('Average annual growth rate between 2000 - 2018:', str(avg_rate) + '%')

    

    return growth_rate_18, avg_rate
##motor vehicle growth rate

motor_growth_rate, motor_avg_rate = growth_rate(uk_traffic, 'all_motor_vehiclesmiles_driven')
##heavy goods vehicle growth rate

hvg_growth_rate, hvg_avg_rate = growth_rate(uk_traffic, 'all_hgvsmiles_driven')
##pedal cycles growth rate

pc_growth_rate, pc_avg_rate = growth_rate(uk_traffic, 'pedal_cyclesmiles_driven')
##motor cycles growth rate

mc_growth_rate, mc_avg_rate = growth_rate(uk_traffic, 'two_wheeled_motor_vehiclesmiles_driven')
##buses and coaches growth rate

bus_growth_rate, bus_avg_rate = growth_rate(uk_traffic, 'buses_and_coachesmiles_driven')
##Light vans growth rate

vans_growth_rate, vans_avg_rate = growth_rate(uk_traffic, 'lgvsmiles_driven')
##cars and taxis growth rate

cars_growth_rate, cars_avg_rate = growth_rate(uk_traffic, 'cars_and_taxismiles_driven')
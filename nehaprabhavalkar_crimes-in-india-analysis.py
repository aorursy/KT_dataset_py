import pandas as pd

import numpy as np

import geopandas as gpd

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.colors import n_colors

from plotly.subplots import make_subplots

init_notebook_mode(connected=True)

import cufflinks as cf

cf.go_offline()
victims = pd.read_csv('../input/crime-in-india/20_Victims_of_rape.csv')

police_hr = pd.read_csv('../input/crime-in-india/35_Human_rights_violation_by_police.csv')

auto_theft = pd.read_csv('../input/crime-in-india/30_Auto_theft.csv')

prop_theft = pd.read_csv('../input/crime-in-india/10_Property_stolen_and_recovered.csv')
inc_victims = victims[victims['Subgroup']=='Victims of Incest Rape']



g = pd.DataFrame(inc_victims.groupby(['Year'])['Rape_Cases_Reported'].sum().reset_index())

g.columns = ['Year','Cases Reported']



fig = px.bar(g,x='Year',y='Cases Reported',color_discrete_sequence=['blue'])

fig.show()
g1 = pd.DataFrame(inc_victims.groupby(['Area_Name'])['Rape_Cases_Reported'].sum().reset_index())

g1.columns = ['State/UT','Cases Reported']

g1.replace(to_replace='Arunachal Pradesh',value='Arunanchal Pradesh',inplace=True)



shp_gdf = gpd.read_file('../input/india-gis-data/India States/Indian_states.shp')

merged = shp_gdf.set_index('st_nm').join(g1.set_index('State/UT'))



fig, ax = plt.subplots(1, figsize=(10, 10))

ax.axis('off')

ax.set_title('State-wise Rape-Cases Reported (2001-2010)',

             fontdict={'fontsize': '15', 'fontweight' : '3'})

fig = merged.plot(column='Cases Reported', cmap='Reds', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)

above_50 = inc_victims['Victims_Above_50_Yrs'].sum()

ten_to_14 = inc_victims['Victims_Between_10-14_Yrs'].sum()

fourteen_to_18 = inc_victims['Victims_Between_14-18_Yrs'].sum()

eighteen_to_30 = inc_victims['Victims_Between_18-30_Yrs'].sum()

thirty_to_50 = inc_victims['Victims_Between_30-50_Yrs'].sum()

upto_10 = inc_victims['Victims_Upto_10_Yrs'].sum()



age_grp = ['Upto 10','10 to 14','14 to 18','18 to 30','30 to 50','Above 50']

age_group_vals = [upto_10,ten_to_14,fourteen_to_18,eighteen_to_30,thirty_to_50,above_50]



fig = go.Figure(data=[go.Pie(labels=age_grp, values=age_group_vals,sort=False,

                            marker=dict(colors=px.colors.qualitative.G10),textfont_size=12)])



fig.show()
g2 = pd.DataFrame(police_hr.groupby(['Area_Name'])['Cases_Registered_under_Human_Rights_Violations'].sum().reset_index())

g2.columns = ['State/UT','Cases Reported']



g2.replace(to_replace='Arunachal Pradesh',value='Arunanchal Pradesh',inplace=True)



shp_gdf = gpd.read_file('../input/india-gis-data/India States/Indian_states.shp')

merged = shp_gdf.set_index('st_nm').join(g2.set_index('State/UT'))



fig, ax = plt.subplots(1, figsize=(10, 10))

ax.axis('off')

ax.set_title('State-wise Cases Registered under Human Rights Violations',

             fontdict={'fontsize': '15', 'fontweight' : '3'})

fig = merged.plot(column='Cases Reported', cmap='Oranges', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)

g3 = pd.DataFrame(police_hr.groupby(['Year'])['Cases_Registered_under_Human_Rights_Violations'].sum().reset_index())

g3.columns = ['Year','Cases Registered']



fig = px.bar(g3,x='Year',y='Cases Registered',color_discrete_sequence=['green'])

fig.show()
police_hr.Group_Name.value_counts()
fake_enc_df = police_hr[police_hr['Group_Name']=='HR_Fake encounter killings'] 

fake_enc_df.Cases_Registered_under_Human_Rights_Violations.sum()
g4 = pd.DataFrame(police_hr.groupby(['Year'])['Policemen_Chargesheeted','Policemen_Convicted'].sum().reset_index())



year=['2001','2002','2003','2004','2005','2006','2007','2008','2009','2010']



fig = go.Figure(data=[

    go.Bar(name='Policemen Chargesheeted', x=year, y=g4['Policemen_Chargesheeted'],

           marker_color='purple'),

    go.Bar(name='Policemen Convicted', x=year, y=g4['Policemen_Convicted'],

          marker_color='red')

])



fig.update_layout(barmode='group',xaxis_title='Year',yaxis_title='Number of policemen')

fig.show()
g5 = pd.DataFrame(auto_theft.groupby(['Area_Name'])['Auto_Theft_Stolen'].sum().reset_index())

g5.columns = ['State/UT','Vehicle_Stolen']

g5.replace(to_replace='Arunachal Pradesh',value='Arunanchal Pradesh',inplace=True)



shp_gdf = gpd.read_file('../input/india-gis-data/India States/Indian_states.shp')

merged = shp_gdf.set_index('st_nm').join(g5.set_index('State/UT'))



fig, ax = plt.subplots(1, figsize=(10, 10))

ax.axis('off')

ax.set_title('State-wise Auto Theft Cases Reported(2001-2010)',

             fontdict={'fontsize': '15', 'fontweight' : '3'})

fig = merged.plot(column='Vehicle_Stolen', cmap='Wistia', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)

auto_theft_traced = auto_theft['Auto_Theft_Coordinated/Traced'].sum()

auto_theft_recovered = auto_theft['Auto_Theft_Recovered'].sum()

auto_theft_stolen = auto_theft['Auto_Theft_Stolen'].sum()



vehicle_group = ['Vehicles Stolen','Vehicles Traced','Vehicles Recovered']

vehicle_vals = [auto_theft_stolen,auto_theft_traced,auto_theft_recovered]



colors = ['crimson','gold','green']



fig = go.Figure(data=[go.Pie(labels=vehicle_group, values=vehicle_vals,sort=False,

                            marker=dict(colors=colors),textfont_size=12)])



fig.show()
g5 = pd.DataFrame(auto_theft.groupby(['Year'])['Auto_Theft_Stolen'].sum().reset_index())



g5.columns = ['Year','Vehicles Stolen']



fig = px.bar(g5,x='Year',y='Vehicles Stolen',color_discrete_sequence=['#17becf'])

fig.show()
vehicle_list = ['Motor Cycles/ Scooters','Motor Car/Taxi/Jeep','Buses',

               'Goods carrying vehicles (Trucks/Tempo etc)','Other Motor vehicles']



sr_no = [1,2,3,4,5]



fig = go.Figure(data=[go.Table(header=dict(values=['Sr No','Vehicle type'],

                                          fill_color='turquoise',

                                           height=30),

                 cells=dict(values=[sr_no,vehicle_list],

                            height=30))

                     ])

fig.show()
motor_c = auto_theft[auto_theft['Sub_Group_Name']=='1. Motor Cycles/ Scooters']



g8 = pd.DataFrame(motor_c.groupby(['Area_Name'])['Auto_Theft_Stolen'].sum().reset_index())

g8_sorted = g8.sort_values(['Auto_Theft_Stolen'],ascending=True)

fig = px.bar(g8_sorted.iloc[-10:,:], y='Area_Name', x='Auto_Theft_Stolen',

             orientation='h',color_discrete_sequence=['#008080'])

fig.show()
g7 = pd.DataFrame(prop_theft.groupby(['Area_Name'])['Cases_Property_Stolen'].sum().reset_index())

g7.columns = ['State/UT','Cases Reported']

g7.replace(to_replace='Arunachal Pradesh',value='Arunanchal Pradesh',inplace=True)



shp_gdf = gpd.read_file('../input/india-gis-data/India States/Indian_states.shp')

merged = shp_gdf.set_index('st_nm').join(g7.set_index('State/UT'))



fig, ax = plt.subplots(1, figsize=(10, 10))

ax.axis('off')

ax.set_title('State-wise Property Stolen-Cases Reported',

             fontdict={'fontsize': '15', 'fontweight' : '3'})

fig = merged.plot(column='Cases Reported', cmap='RdPu', linewidth=0.5, ax=ax, edgecolor='0.2',legend=True)
prop_theft_recovered = prop_theft['Cases_Property_Recovered'].sum()

prop_theft_stolen = prop_theft['Cases_Property_Stolen'].sum()



prop_group = ['Property Stolen Cases','Property Recovered Cases']

prop_vals = [prop_theft_stolen,prop_theft_recovered]



colors = ['red','green']



fig = go.Figure(data=[go.Pie(labels=prop_group, values=prop_vals,sort=False,

                            marker=dict(colors=colors),textfont_size=12)])



fig.show()
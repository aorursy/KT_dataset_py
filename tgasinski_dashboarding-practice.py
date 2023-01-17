#import packages

import numpy as np 

import pandas as pd



#import data

red_cam_viol_org = pd.read_csv('../input/chicago-red-light-and-speed-camera-data/red-light-camera-violations.csv')

red_cam_loc_org = pd.read_csv('../input/chicago-red-light-and-speed-camera-data/red-light-camera-locations.csv')
# install csvvalidator 

import sys

!{sys.executable} -m pip install csvvalidator
# import packages 

from csvvalidator import *



# fields for first dataframe

field_names_1 = ('INTERSECTION', 'VIOLATION DATE', 'VIOLATIONS')



# create validator object

validator_1 = CSVValidator(field_names_1)



# write checks

validator_1.add_value_check('INTERSECTION', str, 'EX1.1', 'Intersection must be a string')

validator_1.add_value_check('VIOLATION DATE', datetime_string('%Y-%m-%d'), 'EX1.2', 'Invalid date')

validator_1.add_value_check('VIOLATIONS', int, 'EX1.3', 'Number of violations not an integer')



# fields for second dataframe

field_names_2 = ('INTERSECTION', 'LONGITUDE', 'LATITUDE')



# create validator object

validator_2 = CSVValidator(field_names_2)



# write checks

validator_2.add_value_check = ('INTERSECTION', str, 'EX2.1', 'Intersection must be a string')

validator_2.add_value_check = ('LONGITUDE', float, 'EX2.2', 'Longitude not a float')

validator_2.add_value_check = ('LATITUDE', float, 'EX2.3', 'Latitude not a float')
# import libraries 

import csv

from io import StringIO



# first sample csv

good_data_1 = StringIO("""INTERSECTION,VIOLATION DATE,VIOLATIONS

Test1-Test2,2014-08-05, 5

Test3-Test4,2014-07-11,12

Test5-Test6,2014-07-04,30

""")



# read text in as a csv

test_csv_1 = csv.reader(good_data_1)



# validate first good csv

validator_1.validate(test_csv_1)
# second sample csv

good_data_2 = StringIO("""INTERSECTION,LONGITUDE,LATITUDE

Test1-Test2, 41.931791, -87.726979

Test3-Test4, 41.924237, -87.746302

Test5-Test6, 41.923676, -87.785441

""")



# read text in as a csv

test_csv_2 = csv.reader(good_data_2)



# validate first good csv

validator_2.validate(test_csv_2)
#examine dataframe

red_cam_viol_org.head()
#remove unnecessary columns



red_cam_viol = red_cam_viol_org[["INTERSECTION", "CAMERA ID", "ADDRESS", "VIOLATION DATE", "VIOLATIONS"]].copy()

red_cam_viol.head()
# number of red light camera violations 

red_cam_viol["VIOLATIONS"].sum()
# number of red light camera violations per day



# convert column to "date time"

red_cam_viol["VIOLATION DATE"] = pd.to_datetime(red_cam_viol["VIOLATION DATE"])



# add new column with day of week 

red_cam_viol["Day of Week"] = red_cam_viol["VIOLATION DATE"].dt.day_name()



# create two dictionarys to sort by day of week 

days = {'Monday' : 1, 'Tuesday' : 2, 'Wednesday' : 3, 'Thursday' : 4, 'Friday' : 5, 'Saturday' : 6, 'Sunday' : 7}

days2 = {1: 'Monday', 2: 'Tuesday', 3 : 'Wednesday', 4: 'Thursday', 5 : 'Friday', 6 : 'Saturday', 7 : 'Sunday'}



# group by day of week, sum number of violations, and sort by day of week

viol_per_day = red_cam_viol.groupby(["Day of Week"])["VIOLATIONS"].count()

viol_per_day = viol_per_day.reset_index()

viol_per_day["Day of Week"] = viol_per_day["Day of Week"].map(days)

viol_per_day = viol_per_day.sort_values(by = "Day of Week")

viol_per_day["Day of Week"] = viol_per_day["Day of Week"].map(days2)

viol_per_day.set_index("Day of Week", drop = True, inplace = True)



# plot data

ax = viol_per_day.plot.bar(color = 'b', ylim=[59000, 73000])

ax.set_ylabel("Total Number of Violations")
red_cam_loc_org.head()
#remove unneeded columns

red_cam_loc = red_cam_loc_org[["INTERSECTION", "LATITUDE", "LONGITUDE"]].copy()

red_cam_loc.head()
#remove all caps

red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.title()



#replace "and" with "-"

red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace(" And ","-")



red_cam_viol.head()
#Before merging, I want to learn a bit more about my datasets. 



null_counts_viol = red_cam_viol.isnull().sum()

print(null_counts_viol)
red_cam_viol.info()
null_counts_loc = red_cam_loc.isnull().sum()

print(null_counts_loc)
red_cam_loc.info()
#create unique lists of intersections within each dataset



loca = np.sort(red_cam_loc['INTERSECTION'].unique())

viol = np.sort(red_cam_viol['INTERSECTION'].unique())
#find values that do not appear in viol lists

def missing(loca, viol): 

    return (list(set(loca) - set(viol)))



missing = missing(loca, viol)

missing.sort()

print(missing)
# find missing values that do not appear in loc list



def missing2(viol, loca): 

    return (list(set(viol) - set(loca)))



missing2 = missing2(viol, loca)

missing2.sort()

print(missing2)
red_cam_loc['INTERSECTION'] = red_cam_loc['INTERSECTION'].str.upper()

red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.upper()
# replace errors found in spot check of lists



red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("?", " ")

red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("STONEY", "STONY")

red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("31ST ST-MARTIN LUTHER KING DRIVE", "DR MARTIN LUTHER KING DRIVE-31ST")

red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("4700 WESTERN", "47TH-WESTERN")
# resave first list with capitalized letters

loca = np.sort(red_cam_loc['INTERSECTION'].unique())
# create new column

red_cam_viol["Corrected Intersection"] = 'Unchecked'



# divides intersections by hyphen and insert in new column

red_cam_viol["First Street"], red_cam_viol["Second Street"] = red_cam_viol.INTERSECTION.str.split("-", 1).str

red_cam_viol['New Intersection'] = red_cam_viol["Second Street"] + "-" + red_cam_viol["First Street"]



def match(df, loca): 

    df.loc[df['INTERSECTION'].isin(loca), "Corrected Intersection"] = df['INTERSECTION']

    df.loc[(~df['INTERSECTION'].isin(loca)) & (df["New Intersection"].isin(loca)), "Corrected Intersection"] = df['New Intersection'] 

    errors = df.loc[df['Corrected Intersection'] == 'Unchecked']

    errors_list = np.sort(errors['INTERSECTION'].unique())

    return df, errors_list



# call function 

red_cam_viol, errors_list = match(red_cam_viol, loca)

print(errors_list)
red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("/", "-")

red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("HIGHWAY", "HWY")

red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("83RD-STONY ISLAND", "STONY ISLAND-83RD")

red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("95TH-STONY ISLAND", "STONY ISLAND-95TH")

red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("ARCHER-NARRAGANSETT-55TH", "ARCHER-NARRAGANSETT")

red_cam_viol["INTERSECTION"] = red_cam_viol["INTERSECTION"].str.replace("LAKE-UPPER WACKER", "LAKE-WACKER")

red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("DR MARTIN LUTHER KING-31ST", "DR MARTIN LUTHER KING DRIVE-31ST")

red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("LAKE SHORE-BELMONT", "LAKE SHORE DR-BELMONT")

red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("PULASKI-ARCHER-50TH", "PULASKI-ARCHER")

red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("KOSTNER-GRAND-NORTH", "KOSTNER-GRAND")

red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("HOMAN-KIMBALL-NORTH", "HOMAN-KIMBALL")

red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("WESTERN-DIVERSEY-ELSTON", "WESTERN-DIVERSEY")

red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("KEDZIE-79TH-COLUMBUS", "KEDZIE-79TH")

red_cam_loc["INTERSECTION"] = red_cam_loc["INTERSECTION"].str.replace("HALSTED-FULLERTON-LINCOLN", "HALSTED-FULLERTON")
#resave list with corrected intersection names

loca = np.sort(red_cam_loc['INTERSECTION'].unique())



#rewrite columns with corrected values

red_cam_viol["First Street"], red_cam_viol["Second Street"] = red_cam_viol.INTERSECTION.str.split("-", 1).str

red_cam_viol['New Intersection'] = red_cam_viol["Second Street"] + "-" + red_cam_viol["First Street"]



# call function again 

red_cam_viol, errors_list = match(red_cam_viol, loca)

print(errors_list)
red_cam_viol.info()
red_cam_final_org = pd.merge(red_cam_viol, red_cam_loc, left_on = "Corrected Intersection",right_on= "INTERSECTION", how = "left")

red_cam_final_org.head()
null_counts_merged = red_cam_final_org.isnull().sum()

print(null_counts_merged)
red_cam_final_org.info()
red_cam_final_org["INTERSECTION_y"] = np.where(red_cam_final_org["INTERSECTION_y"].isnull(), red_cam_final_org["INTERSECTION_x"], red_cam_final_org["INTERSECTION_x"])
red_cam_final = red_cam_final_org[["INTERSECTION_y", "CAMERA ID", "LATITUDE", "LONGITUDE", "VIOLATION DATE", "VIOLATIONS"]].copy()

red_cam_final.rename(columns = {"INTERSECTION_y" : "INTERSECTION"}, inplace = True)

red_cam_final.head()
intersection_grouped = red_cam_final.groupby("INTERSECTION")

intersection_summed = pd.DataFrame(intersection_grouped["VIOLATIONS"].sum())

intersection_summed.head()
red_cam_totals = pd.merge(intersection_summed,red_cam_final,left_on = "VIOLATIONS", right_index = True)

red_cam_totals = red_cam_totals[["VIOLATIONS", "LATITUDE", "LONGITUDE"]].copy()



# set 100 as scale factor

red_cam_totals["SCALE"] = red_cam_totals["VIOLATIONS"] / 100

red_cam_totals.head()
# import libraries

from shapely.geometry import Point, Polygon 

import matplotlib.pyplot as plt

import geopandas as gpd 

import descartes
#convert to a geo-dataframe

geometry = [Point(xy) for xy in zip(red_cam_totals['LONGITUDE'], red_cam_totals['LATITUDE'])]

crs = {'init','epsg:4326'}

gdf = gpd.GeoDataFrame(red_cam_totals, crs=crs, geometry=geometry)
#plot red light cameras onto city map

street_map = gpd.read_file('../input/chicago-streets-shapefiles/geo_export_75808441-05b9-4a51-a665-cf23dcf0a285.shx')

fig,ax = plt.subplots(figsize = (15,15))

street_map.plot(ax = ax, alpha = 0.4, color = "grey")

gdf.plot(ax=ax, markersize=red_cam_totals["SCALE"], marker="o", color="red")
viol_per_day = viol_per_day.reset_index()

viol_per_day.head()
#install package 

!pip install chart-studio



#import plotly 

import chart_studio.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode()
data = [

    go.Scatter(x=viol_per_day['Day of Week'], y=viol_per_day['VIOLATIONS'], marker=dict(color='rgb(158,202,225)', line=dict(color='rgb(8,48,107)', width=1.5,)),

    opacity=0.6)]



layout = go.Layout(autosize = True, title="Red Light Camera Violations per Day", xaxis={'title':'Days of Week'}, yaxis={'title':'Total Violations'})



fig = go.Figure(data = data, layout = layout)

iplot(fig)
# reset index to convert dataframe from geo to plotly

gdf_2 = gdf.reset_index()

gdf_2['TEXT'] = gdf_2['INTERSECTION'] + ": " + gdf_2['VIOLATIONS'].astype(str) + ' total violations'
#Instead of using scaled data, I will normalize the "VIOLATIONS" column.



# normalize data using min-max 

gdf_2["NORMALIZED"] = (gdf_2["VIOLATIONS"] - gdf_2["VIOLATIONS"].min()) / (gdf_2["VIOLATIONS"].max() - gdf_2["VIOLATIONS"].min())
mapbox_access_token = 'pk.eyJ1IjoidGdhc2luc2tpIiwiYSI6ImNqcXc3MjhpNzEzMnYzeG9ieDNkb2M5ZmQifQ.m3MsgcBIXdwOT6hxvi007g'



data = [

    go.Scattermapbox(

        lat= gdf_2['LATITUDE'],

        lon= gdf_2['LONGITUDE'],

        mode='markers',

        text = gdf_2['TEXT'],

        hoverinfo = 'text',

        marker=dict(

            size= 8,

            color = gdf_2['NORMALIZED'],

            colorscale= 'Jet', 

            showscale=True,

            cmax=1,

            cmin=0),),]



layout = go.Layout(

    title = "Number of Total Red Light Violations by Intersection in Chicago", 

    autosize=True,  

    hovermode='closest',

    mapbox=dict(

        accesstoken=mapbox_access_token,

        bearing=0,

        center=dict(

            lat=41.881832,

            lon=-87.623177),

        pitch=0,

        zoom=10),)



fig = dict(data=data, layout=layout)

iplot(fig)
print("Total Number of Red Camera Violations in Chicago:")

red_cam_viol["VIOLATIONS"].sum()
mapbox_access_token = 'pk.eyJ1IjoidGdhc2luc2tpIiwiYSI6ImNqcXc3MjhpNzEzMnYzeG9ieDNkb2M5ZmQifQ.m3MsgcBIXdwOT6hxvi007g'



trace1 = go.Scatter(x=viol_per_day['Day of Week'], y=viol_per_day['VIOLATIONS'], mode='lines+markers+text', xaxis='x1', yaxis='y1')#, subplot = 'plot1')



trace2 = go.Scattermapbox(

        lat= gdf_2['LATITUDE'],

        lon= gdf_2['LONGITUDE'],

        mode='markers',

        text = gdf_2['TEXT'],

        hoverinfo = 'text',

        marker=dict( 

            size= 8,

            color = gdf_2['NORMALIZED'],

            colorscale= 'Jet',

            showscale=True,

            colorbar=dict(title = dict(text="Violations (Scaled)", side="right"), x = 1), 

            cmax=1,

            cmin=0), 

        subplot = 'mapbox')



data = [trace1, trace2]



layout = go.Layout(

    autosize=True,

    hovermode='closest',

    xaxis=dict(

        domain=[0, 0.45]

    ),

    yaxis=dict(

        domain=[0, 1]

    ),

    mapbox=dict(

        accesstoken=mapbox_access_token,

        domain = {'x' : [.5, 1], 'y' : [0,1]},

        bearing=0,

        center=dict(

            lat=41.8746,

            lon=-87.6687),

        pitch=0,

        zoom=10),)



fig = go.Figure(data=data, layout=layout)



fig['layout']['xaxis1'].update(title='Days per Week')

fig['layout']['yaxis1'].update(title='Total Number of Violations')

fig['layout'].update(showlegend=False)

fig['layout'].update(height=600, width=800, title='Total Red Light Camera Violations in Chicago: By Day and Intersection')



iplot(fig) 
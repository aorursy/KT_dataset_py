import gpxpy

import pandas as pd

import numpy as np

import math

import matplotlib.pyplot as plt

pd.set_option('display.max_rows',50)

import os 

import re

import time

%matplotlib inline
#May want to replace with geopy

def calculate_bearing(pointA, pointB):



    if (type(pointA) != tuple) or (type(pointB) != tuple):

        raise TypeError("Only tuples are supported as arguments")



    lat1 = math.radians(pointA[0])

    lat2 = math.radians(pointB[0])    

    diffLong = math.radians(pointB[1] - pointA[1])



    x = math.sin(diffLong) * math.cos(lat2)

    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)

            * math.cos(lat2) * math.cos(diffLong))



    bearing = np.rad2deg(np.arctan2(x, y))

    compass_bearing = bearing % 360;

  

    

    return compass_bearing
def add_angle_difference(df_session,Angle_Difference,Wind_Direction):

    df_session[Angle_Difference] = np.nan

    df_session[Angle_Difference] = 180 - abs(abs(df_session[Wind_Direction] - df_session['Bearing']) - 180) 

        

    return df_session
def filter_by_speed(df_session,lower_bound,upper_bound):

    df_session = df_session[(df_session['Knots'] > lower_bound) & (df_session['Knots'] < upper_bound)]

    return df_session
def find_upwind_tack_angles(df_session):

    

    list_tack_angles = [None] * 4

    angle_bounds = [[None,None]] * 4

    not_string = ""

    buffer_around_mode_bearing = 30 #to make sure we don't classify the same tack twice

    

    #find the tack angle

    for i in range(0,4):

        if i == 0:

            list_tack_angles[i] = df_session['Bearing_Rounded'].mode()[0] 

        else:

            if i > 1:

                not_string += " | "    

            if angle_bounds[i-1][0] < angle_bounds[i-1][1]: #so between 60 and 120 is an example

                not_string += "(df_session['Bearing_Rounded'].between(%d,%d))" % (angle_bounds[i-1][0],angle_bounds[i-1][1])

            else: #so between 345 and 45 is an example

                not_string += "(df_session['Bearing_Rounded'] > %d) | (df_session['Bearing_Rounded'] < %d)" % (angle_bounds[i-1][0],angle_bounds[i-1][1])                

                

            not_string_exec = "list_tack_angles[%d] = df_session[~(%s)]['Bearing_Rounded'].mode()[0]" % (i,not_string)                                               

            exec(not_string_exec)            

        angle_bounds[i] = [((list_tack_angles[i]-buffer_around_mode_bearing)%360),((list_tack_angles[i]+buffer_around_mode_bearing)%360)]

    speed_dict = dict()

    

    for tack_angle in list_tack_angles:

        speed_dict[tack_angle] = df_session[df_session['Bearing_Rounded'] == tack_angle]['Knots'].mean()



    list_tack_angles = sorted(speed_dict, key=speed_dict.get)

    

    

    direction_tack = dict()

    if ((list_tack_angles[0]-list_tack_angles[1]) % 360) < ((list_tack_angles[1]-list_tack_angles[0]) % 360):

    

        direction_tack['upwind_port'] = list_tack_angles[0]

        direction_tack['upwind_starboard'] = list_tack_angles[1]

    else:

        direction_tack['upwind_port'] = list_tack_angles[1]

        direction_tack['upwind_starboard'] = list_tack_angles[0]

     

    return direction_tack
def add_wind_direction(df_session):

    

    direction_tack = find_upwind_tack_angles(df_session)

    

    approx_wind_direction_calc_from_upwind = (direction_tack['upwind_port'] - (((direction_tack['upwind_port'] - direction_tack['upwind_starboard']) % 360)/2))%360

    

    df_session['Wind_Direction'] = approx_wind_direction_calc_from_upwind

    return df_session



    #approx_wind_direction_calc_from_downwind = direction_tack['downwind_port'] - ((direction_tack['downwind_port'] - direction_tack['downwind_starboard'])%360/2)



   
def add_VMG(df_session,Angle_Difference,VMG):

    df_session[VMG] = df_session['Knots'] * np.cos(np.radians((df_session[Angle_Difference])))

    return df_session
def add_tack(df_session):

    df_session['tack'] = ''

    df_session.loc[((df_session['Bearing'] + df_session['Angle_Difference']) % 360) == df_session['Wind_Direction'],'tack'] = 'Starboard'

    df_session.loc[df_session['tack'] == '','tack'] = 'Port'

    return df_session
def infer_wind_direction(df_session):

    upwind_anglediff_mean = df_session[(df_session['Direction'] == 'Upwind') & (df_session['Pushing'] == True)][['Angle_Difference']].mean()

    #commented out downwind because downwind angles are less reliable

    #downwind_anglediff_mean = df_session[(df_session['Direction'] == 'Downwind') & (df_session['Pushing'] == True)][['Angle_Difference']].mean()

    df_session.loc[(df_session['Direction'] == 'Upwind') & (df_session['Pushing'] == True),'Wind_Direction_Inferred'] = df_session[(df_session['Direction'] == 'Upwind') & (df_session['Pushing'] == True)]['Angle_Difference'] + df_session[(df_session['Direction'] == 'Upwind') & (df_session['Pushing'] == True)]['Wind_Direction']  - float(upwind_anglediff_mean) 



    df_session['Wind_Direction_Inferred_Smooth'] = np.nan

    df_session.loc[df_session['Wind_Direction_Inferred'].notnull(),'Wind_Direction_Inferred_Smooth'] = df_session[df_session['Wind_Direction_Inferred'].notnull()]['Wind_Direction_Inferred'].rolling(window=100,center=True).mean()

    df_session['Wind_Direction_Inferred_Smooth'] = df_session['Wind_Direction_Inferred_Smooth'].fillna(method='pad')

    

    df_session = add_angle_difference(df_session,'Angle_Difference_Inferred','Wind_Direction_Inferred_Smooth')

    df_session = add_VMG(df_session,'Angle_Difference_Inferred','VMG_Inferred')

    return df_session
def create_dataframe(gpx_file,time=None):

    gpx = gpxpy.parse(open(gpx_file))

    track = gpx.tracks[0]

    segment = track.segments[0]

    

    data = []

    segment_length = segment.length_3d()

    for point_idx, point in enumerate(segment.points):

        data.append([point.longitude, point.latitude,

                  point.time, segment.get_speed(point_idx)])



    columns = ['Longitude', 'Latitude', 'Time', 'Speed']

    df_session = pd.DataFrame(data, columns=columns)

    

    df_session['Bearing'] = np.nan

    for i in range(1,len(df_session)):  

        df_session.iloc[i,df_session.columns.get_loc('Bearing')] = calculate_bearing((df_session['Latitude'][i-1], df_session['Longitude'][i-1]),(df_session['Latitude'][i], df_session['Longitude'][i]))

        

    #round the bearning to the nearest "base" degrees

    base = 3 #if base = 5, then round to the nearest five degrees

    df_session['Bearing_Rounded'] =  np.round(base * np.round(df_session['Bearing']/base),0)

    

    if time == None:

        df_session.index = df_session['Time']

        df_session['Knots'] = df_session['Speed']*1.94384

        

    df_session = filter_by_speed(df_session,14,30)

    if (len(df_session) < 500): #not enough observations  

        return False

    

    del(df_session['Time'])

    del(df_session['Speed'])

      

    df_session = add_wind_direction(df_session)

    df_session = add_angle_difference(df_session,'Angle_Difference','Wind_Direction') 

    df_session = add_VMG(df_session,'Angle_Difference','VMG')

    df_session = add_tack(df_session)

    

    #Defined pushing as the 40th percentile. Assumes that I'm pushing for 60% of the sessions

    UPWIND_LOWER_ANGLEDIFF = 35

    UPWIND_UPPER_ANGLEDIFF = 65

    DOWNWIND_LOWER_ANGLEDIFF = 130



    df_session['Direction'] = None

    df_session.loc[(df_session['Angle_Difference'] < UPWIND_UPPER_ANGLEDIFF) & (df_session['Angle_Difference'] > UPWIND_LOWER_ANGLEDIFF),'Direction'] = 'Upwind'

    df_session.loc[df_session['Angle_Difference'] > DOWNWIND_LOWER_ANGLEDIFF,'Direction'] = 'Downwind'



    pushing_upwind = df_session[df_session['Direction'] == 'Upwind']['Knots'].quantile(q=.4)

    pushing_downwind = df_session[df_session['Direction'] == 'Downwind']['Knots'].quantile(q=.4)



    df_session['Pushing'] = False

    df_session.loc[(df_session['Direction'] == 'Upwind') & (df_session['Knots'] > pushing_upwind) & (df_session['Knots'] < 26),'Pushing'] = True

    df_session.loc[(df_session['Direction'] == 'Downwind') & (df_session['Knots'] > pushing_downwind) & (df_session['Knots'] < 32),'Pushing'] = True

    

    df_session = infer_wind_direction(df_session)

    

    return df_session
directory = "../input/gps-watch-data/anthony_activities/"



to_analyze = os.listdir(directory)[-5:] #just does the last 5s



df_session_summaries = pd.DataFrame(columns=['Upwind_VMG_Mean','Upwind_Knots','Upwind_Angle_Difference','Upwind_VMG_075','Downwind_VMG_Mean','Downwind_Knots','Downwind_Angle_Difference','Downwind_Knots_075','Upwind_VMG_Starboard','Upwind_Knots_Starboard','Upwind_Angle_Starboard','Upwind_VMG_Port','Upwind_Knots_Port','Upwind_Angle_Port','Downwind_VMG_Starboard','Downwind_Knots_Starboard','Downwind_Angle_Starboard','Downwind_VMG_Port','Downwind_Knots_Port','Downwind_Angle_Port'],index=to_analyze)



for gpx_file in to_analyze:

    if (os.stat(directory + gpx_file).st_size > 50000):

        df_session = create_dataframe(directory + gpx_file)

        if (not isinstance(df_session, int)):  

    

            df_session = df_session[df_session['Pushing'] == True]

    

            VMG = 'VMG_Inferred'

            Angle_Difference = 'Angle_Difference_Inferred'

    

            summary_list = df_session[(df_session['Direction'] == 'Upwind')][[VMG,'Knots',Angle_Difference]].mean().tolist()

            summary_list = summary_list + df_session[(df_session['Direction'] == 'Upwind')][[VMG]].quantile(q=0.75).tolist()

            summary_list = summary_list + df_session[(df_session['Direction'] == 'Downwind')][[VMG,'Knots',Angle_Difference]].mean().tolist()

            summary_list = summary_list + df_session[(df_session['Direction'] == 'Downwind')][['Knots']].quantile(q=0.75).tolist()

            summary_list = summary_list + df_session[(df_session['Direction'] == 'Upwind') & (df_session['tack'] == 'Starboard')][[VMG,'Knots',Angle_Difference]].mean().tolist()

            summary_list = summary_list + df_session[(df_session['Direction'] == 'Upwind') & (df_session['tack'] == 'Port')][[VMG,'Knots',Angle_Difference]].mean().tolist()

            summary_list = summary_list + df_session[(df_session['Direction'] == 'Downwind') & (df_session['tack'] == 'Starboard')][[VMG,'Knots',Angle_Difference]].mean().tolist()

            summary_list = summary_list + df_session[(df_session['Direction'] == 'Downwind') & (df_session['tack'] == 'Port')][[VMG,'Knots',Angle_Difference]].mean().tolist()

    

            df_session_summaries.loc[df_session_summaries.index == gpx_file,['Upwind_VMG_Mean','Upwind_Knots','Upwind_Angle_Difference','Upwind_VMG_075','Downwind_VMG_Mean','Downwind_Knots','Downwind_Angle_Difference','Downwind_Knots_075','Upwind_VMG_Starboard','Upwind_Knots_Starboard','Upwind_Angle_Starboard','Upwind_VMG_Port','Upwind_Knots_Port','Upwind_Angle_Port','Downwind_VMG_Starboard','Downwind_Knots_Starboard','Downwind_Angle_Starboard','Downwind_VMG_Port','Downwind_Knots_Port','Downwind_Angle_Port']] = summary_list 

        

            print(df_session_summaries[df_session_summaries.index == gpx_file][['Upwind_VMG_Mean','Upwind_Knots','Upwind_Angle_Difference']])

        
df_session_summaries.index = df_session_summaries.index.str[0:15] 

df_session_summaries.index = pd.to_datetime(df_session_summaries.index,format="%Y%m%d-%H%M%S")



df_session_summaries.to_csv('df_session_summary_' + time.strftime("%Y%m%d")  + '.csv')
df_session_summaries
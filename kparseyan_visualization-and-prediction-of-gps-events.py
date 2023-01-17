import numpy as np

import pandas as pd

import datetime as dt

import geopy

from geopy.distance import vincenty, VincentyDistance
%matplotlib inline
# Read in the data and convert to the 'Date/Time' column to datetime format

That_Sweet_Data = '../input/uber-raw-data-aug14.csv'

data = pd.read_csv(That_Sweet_Data)

data['Date/Time'] = pd.to_datetime(data['Date/Time'], format='%m/%d/%Y %H:%M:%S')



# Display the data

data.head()
# This function excludes data that falls outside of input coordinates

def only_in_rectangle(data,

                      Lat_col_index, lat_min, lat_max,

                      Long_col_index, lng_min, lng_max,

                      time_index):

    col0 = data.iloc[:, time_index]

    col1 = data.iloc[:, Lat_col_index]

    col2 = data.iloc[:, Long_col_index]

    temp = pd.concat([col0, col1, col2], axis=1)

    return temp[(temp.iloc[:, 1] >= lat_min)

         & (temp.iloc[:, 1] <= lat_max)

         & (temp.iloc[:, 2] >= lng_min)

         & (temp.iloc[:, 2] <= lng_max)]
# Excluding data that falls outside of the Manhattan Island region

data = only_in_rectangle(data, 1, 40.699, 40.877, 2, -74.018, -73.910, 0)
# EventMapper returns an array of values representing event influence in steps of res_t and res_d.

# It accepts an array of event times (in datetime format), latitides, and longitudes.

# Each event increases the value of its epicenter by 1, with a decay constant for time and distance

# for the surrounding area to a limit of min_decay.

# Time +alpha or -alpha from pickup is 36.8% as effective at the exact event time.

# Distance +simga and -sigma from pickup is 36.8% as effective as at the exact event location.

# This decay amount represents 1 / exp(distance)

# The event values at each position are graphed over time to create an animation of events.



# ---------------------------------------------------------

# A rectangle is defined by the max/min latitude and longitude, which correspond approximately

# to the height and width of our animation.

# The rectangle boundary is pushed out to be a multiple of the resolution.

# Data that doesn't fall inside the rectangle is excluded.



# We determine the zone of influence from the event epicenter and define a buffer region.

# This region is for data that falls near the rectangle borders and would cause the zone

# to fall outside the event array.



# We find the time index, latitude index, and longitude index for each event.

# We create a square representing the event influence, with event epicenter in the center.

# We project the square into a cube, representing the change of influence over time.



# The cube is added to the event array at each specific event location and time.

# The buffer region is removed from the event array.



#-----------------------------------------------------------

# I've disabled a bunch of print outputs to keep it clean. Uncomment them to see.



def EventMapper(data,

                time_col_index=0, Lat_col_index=1, Long_col_index=2,

                min_decay=0.02,

                res_t='30min', res_d=100.0,

                alpha='30min', sigma=100.0):

    

    # Convert res_t and alpha to timedelta format.

    res_t = pd.to_timedelta(res_t)

    alpha = pd.to_timedelta(alpha)

    

    # Get the space and time boundaries and determine number of time index positions.

    min_time = np.min(data.iloc[:, time_col_index])

    max_time = np.max(data.iloc[:, time_col_index])

    min_lat = np.min(data.iloc[:, Lat_col_index])

    max_lat = np.max(data.iloc[:, Lat_col_index])

    min_lng = np.min(data.iloc[:, Long_col_index])

    max_lng = np.max(data.iloc[:, Long_col_index])

    

#     print('Max Lat Long:', max_lat, max_lng)

#     print('Min Lat Long:', min_lat, min_lng)

#     print('Min Time & Max Time:', min_time, max_time, '\n')

    

#     print('Spatial resolution (res_d) of {} meters supplied.'.format(res_d))

#     print('Temporal resolution (res_t) of {} supplied.\n'.format(res_t))

    

    # Get the distance between the rectangle edges and round up by the resolution.

    lat_distance = ((np.ceil(vincenty((min_lat, min_lng), (max_lat, min_lng)).meters / res_d))*res_d)

    lng_distance = ((np.ceil(vincenty((min_lat, min_lng), (min_lat, max_lng)).meters / res_d))*res_d)

    

    # Get the center of the rectangle.

    mid_lat = (max_lat + min_lat) / 2

    mid_lng = (max_lng + min_lng) / 2

    center = geopy.Point(mid_lat, mid_lng)

    

    # Set the bearing directions for expansion using GPS coordinates.

    Bearing_North = 0

    Bearing_East = 90

    Bearing_South = 180

    Bearing_West = 270

    

    # Expand the boundaries of inclusion rectangle as GPS coordinates.

    plt_max_lat = VincentyDistance(meters=(lat_distance/2)).destination(center, Bearing_North).latitude

    plt_min_lat = VincentyDistance(meters=(lat_distance/2)).destination(center, Bearing_South).latitude

    plt_max_lng = VincentyDistance(meters=(lng_distance/2)).destination(center, Bearing_East).longitude

    plt_min_lng = VincentyDistance(meters=(lng_distance/2)).destination(center, Bearing_West).longitude

      

#     print('Map has maximum Lat & Long: {} & {}'.format(np.round(plt_max_lat, 4), np.round(plt_max_lng, 4)))

#     print('Map has minimum Lat & Long: {} & {}'.format(np.round(plt_min_lat, 4), np.round(plt_min_lng, 4)))

#     print('width of area being examined: {}meters'.format(lng_distance))

#     print('Height of area being examined: {}meters\n'.format(lat_distance))

    

    # How many pixels and frames will there be?

    num_v = int(lat_distance/res_d)

    num_h = int(lng_distance/res_d)

    num_t = int(np.ceil((max_time - min_time) / res_t))

#     print('Map will have:')

#     print('{} vertical spaces for a real-world length of {} meters'.format(num_v, np.round(lat_distance)))

#     print('{} horizontal spaces for a real-world width of {} meters'.format(num_h, np.round(lng_distance)))

#     print('{} temporal spaces for a total time of {}.\n'.format(num_t, (max_time - min_time)))

    

    # Confine data to being inside of inclusion rectangle, passing only the desired columns.

    col0 = data.iloc[:, time_col_index]

    col1 = data.iloc[:, Lat_col_index]

    col2 = data.iloc[:, Long_col_index]

    temp = pd.concat([col0, col1, col2], axis=1)

    df = temp[(temp.iloc[:, 1] >= plt_min_lat)

         & (temp.iloc[:, 1] <= plt_max_lat)

         & (temp.iloc[:, 2] >= plt_min_lng)

         & (temp.iloc[:, 2] <= plt_max_lng)] 

    

    # Rename the columns and reset the indexes for predictability.

    df.columns = ['Time', 'Lat', 'Long']

    df = df.reset_index(drop = 'true')

       

    # Determine the extent of influence from each event, representing the buffer zone thickness.

    # How much decay per distance increment?

    if sigma == 0:

        distance_influence = 0

        distance_decay_ratio = 0

    else:

        distance_decay_ratio = sigma / res_d

        distance_influence = int(np.ceil(np.log(1 / min_decay) / distance_decay_ratio))



    # How much decay per time increment?

    if alpha == pd.to_timedelta('0min'):

        time_influence = 0

        time_decay_ratio = 0

    else:

        time_decay_ratio = pd.to_timedelta(alpha) / res_t

        time_influence = int(np.ceil(np.log(1 / min_decay) / time_decay_ratio))

    

    di = distance_influence

    ti = time_influence

    

#     print('Max influence distance from event is {}meters.'.format(di * sigma))

#     print('Max influence time from event is {}mins.\n'.format(ti * round((alpha.total_seconds()) / 60)))

      

    # Use the distance and time resolutions to produce the corresponding indexes, adding buffer thickness.

    # Store the indexes into the dataframe df

    grid_height = (plt_max_lat - plt_min_lat) * (res_d / lat_distance)

    grid_width = (plt_max_lng - plt_min_lng) * (res_d / lng_distance)

    df.loc[:, 'time_index'] = np.round((df.loc[:,'Time'] - min_time) / res_t) + ti

    df.loc[:, 'lat_index'] = np.round((df.loc[:,'Lat']-plt_min_lat)/grid_height) + di

    df.loc[:, 'lng_index'] = np.round((df.loc[:, 'Long']-plt_min_lng)/grid_width) + di

    

    # -----------------------------------------------------------------------------------------------

    # Create an 'event square' and then an 'event cube' to be added to the 'event array'.

    

    

    EventSquare = np.zeros([(di*2)+1, (di*2)+1])

    for i in range(di, -1, -1):

        distance_decay = 1 / np.exp(distance_decay_ratio * i)

        TempSquare = np.ones([(i*2)+1, (i*2)+1]) * distance_decay

        low = di-i

        high = di+i+1

        EventSquare[low:high, low:high] = TempSquare

#     print('ALL HAIL THE MIGHTY EVENT SQUARE:\n')

#     print(np.round(EventSquare,3))

#     print('\n')

    

    # Project the EventSquare into the EventCube.

    EventCube = np.zeros([(ti*2)+1, (di*2)+1, (di*2)+1])

    for i in range(ti, -1, -1):

        time_decay = 1 / np.exp(time_decay_ratio * i)

        low = ti-i

        high = ti+i

        EventCube[low, :, :] = EventSquare * time_decay

        EventCube[high, :, :] = EventSquare * time_decay



#     print('ALL HAIL THE EVEN MIGHTIER EVENT CUBE.')

#     print('You are not worthy of 3 dimensions. See a 2D slice cut through the middle:\n')

#     print(np.round(EventCube[:, ti, :],3))

#     print('\n')

    

    # ---------------------------------------------------------------------------------------------

    # Initialize the EventArray with a buffer zone to account for the EventCube hitting the edges.

    EA = np.zeros([num_t + (2 * ti)+1, num_v + (2 * di)+1, num_h + (2 * di)+1])

#     print( 'Shape of Event Array with buffer is {}'.format(EA.shape))



    # A loop adds the EventCube into the EventArray at the event location and time.

    for i in range(0, df.shape[0], 1):

        time_i = int(df.loc[i, 'time_index'])

        lat_i = int(df.loc[i, 'lat_index'])

        lng_i = int(df.loc[i, 'lng_index'])

        EA[time_i-ti:time_i+ti+1, lat_i-di:lat_i+di+1, lng_i-di:lng_i+di+1] += EventCube



    # Redefine the EventArray to exclude the buffer zone.

    EventArray = EA[ti:num_t+ti, di:num_v+di, di:num_h+di]

    print('Time starts at {}.'.format(min_time))

    print( 'Shape of Output Array is {}\n'.format(EventArray.shape))

    return EventArray
# Run the EventMapper with the Uber Data

# This will take 5-10 mins!

EA_5050_1515 = EventMapper(data, res_t = '15min', res_d = 50.0, alpha = '15min', sigma = 50.0)
# Here's where the animation magic happens



from matplotlib import animation

import matplotlib.pyplot as plt

#from JSAnimation import IPython_display

# No JSAnimation? Sadface.

#from Kaggle-ipython-support import :(



# Which array are we working with?

EA = EA_5050_1515



# How tall should the animation be? (inches)

k = 4

H_ratio = (EA.shape[1]/EA.shape[2])



# Start time position, temporal resolution, and start datetime

t = 0

res_t = pd.to_timedelta('15min')

init_t = pd.to_datetime('2014-04-01 00:00:00')

fig = plt.figure(figsize=(k, k*H_ratio))



# What data are we plotting?

def f(t):

    img = EA[t, :, :]

    return img



# Displaying the text for the time on the animation

time_xpos = int(np.round(EA.shape[2]*0.01))

time_ypos = int(np.round(EA.shape[1]-30))

time_text = plt.text(time_xpos, time_ypos, '', fontsize=20, color='white', weight='bold')



# Dispalying the data, but not the borders or axes.

im = plt.imshow(f(t), animated=True, origin='lower')

plt.axis('off')

fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=None, hspace=None)



# How is the animation changing per frame?

def updatefig(*args):

    global t

    t += 1

    im.set_array(f(t))

    time_text.set_text(init_t + (t * res_t))

    return im, time_text



# How many frames shall we produce?

frames = EA.shape[0]-3

# frames = 10



ani = animation.FuncAnimation(fig, updatefig, frames=frames)

writer=animation.FFMpegWriter(fps=30, bitrate=2048, extra_args=['-vcodec', 'libx264'])



# uncomment the next line if you can get JSAnimation to view the animation right in the browser.

# ani

# You'll need to install ffmpeg to your environment to make the next line work.

# If you can't install ffmpeg, get another writer and change the writer.

ani.save('UberNY_Apr_5050_1515.mp4', writer=writer)
import os

import pandas as pd

import numpy as np

import io

import base64

import time

from datetime import datetime, timedelta



import matplotlib.pyplot as plt

from matplotlib import animation, rc



from IPython.display import display, HTML

from tqdm import tqdm



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/train.csv')



xlim = [-74.03, -73.77]

ylim = [40.63, 40.85]



df = df[(df.pickup_longitude> xlim[0]) & (df.pickup_longitude < xlim[1])]

df = df[(df.dropoff_longitude> xlim[0]) & (df.dropoff_longitude < xlim[1])]

df = df[(df.pickup_latitude> ylim[0]) & (df.pickup_latitude < ylim[1])]

df = df[(df.dropoff_latitude> ylim[0]) & (df.dropoff_latitude < ylim[1])]



longitude = list(df.pickup_longitude) + list(df.dropoff_longitude)

latitude = list(df.pickup_latitude) + list(df.dropoff_latitude)



longitude_pick = list(df.pickup_longitude) 

latitude_pick = list(df.pickup_latitude) 

longitude_drop = list(df.dropoff_longitude)

latitude_drop = list(df.dropoff_latitude)
imageSize = (700,700)

latRange = ylim

longRange = xlim



allLatInds  = imageSize[0] - (imageSize[0] * (np.array(latitude)  - latRange[0])  / (latRange[1]  - latRange[0])).astype(int)

allLongInds = (imageSize[1] * (np.array(longitude) - longRange[0]) / (longRange[1] - longRange[0])).astype(int)



locationDensityImage = np.zeros(imageSize)

A = np.arange(700)

for lat, lon in zip(allLatInds,allLongInds):

    if lat in A and lon in A:

        locationDensityImage[lat,lon] += 1
def circle_arc(ax, ay, bx, by, alpha=1, time=None): 

    

    e = 1e-8

    

    dx, dy = (bx-ax)*1.0, (by-ay)*1.0    

    mid_x, mid_y = 0.5*(ax+bx), 0.5*(ay+by)

    D = np.sqrt(dx**2 + dy**2)

                           

    # Get a perpendicular line to (a, b) at its midpoint, and let it's length be L = alpha*D. 

    # This will be the cicle center.

    # Break L down to its X and y component Lx, Ly, we have:

    # Lx^2 + Ly^2 = L^2 --- eq1

    # Ly/Lx = g ---- eq2

    # Soving it we have Lx^2 = L^2 / (1+g^2); Ly = g*Lx

                

    g = 0    

    if (np.abs(dy) > e):

        g = -dx/dy

        Lx = np.sqrt( alpha*D / (1+g**2)) * -np.sign(g)

        Ly = g*Lx

    else:

        Lx = 0

        Ly = -np.sqrt(alpha*D)

                

    #Circle center        

    cx, cy =  mid_x+Lx, mid_y+Ly      

    R = np.sqrt((ax-cx)**2 + (ay-cy)**2)

    

    if np.abs((ax-cx)) > e:

        theta0 = np.arctan((ay-cy)/(ax-cx)) 

    else:

        theta0 = np.sign(ay - cy) * 0.5 * np.pi 

        

    if np.abs((bx-cx)) > e:

        theta1 = np.arctan((by-cy)/(bx-cx))

    else:

        theta1 = np.sign(by - cy) * 0.5 * np.pi                                 

    

    theta0 += np.pi if ax - cx < 0 else 0    

    theta1 += np.pi if bx - cx < 0 else 0          

               

    if np.abs(dx) > e:

        phi = np.arctan(dy/dx)

        if bx < ax:

            phi += np.pi

        elif by < ay:     

            phi += 2*np.pi

    else:       

        phi = 0.5*np.pi if by > ay else 1.5*np.pi                              

                

    if not time == None:    

        num = np.ceil(time/60.0) #Number of points set to be number of minutes in duration

        num = int(np.maximum(num, 2))

    else:

        num = None 



    X, Y = get_circle_arc(cx, cy, R, theta0, theta1, num)

    

    check = (np.abs(X[0]-ax)  < e and np.abs(Y[0]-ay) < e and \

             np.abs(X[-1]-bx) < e and np.abs(Y[-1]-by) < e)

    try:

        assert check                    

    except:

        print(ax,ay,bx,by)

        print(X[0],X[-1],Y[0],Y[-1])

        print('')



    return X, Y, phi



def get_circle_arc(cx, cy, R, theta0, theta1, num=None, debug=False):

    

    if num == None:                

        num = int(100 * 0.5*np.abs(theta1 - theta0)/np.pi) # in radian

        num = int(np.maximum(num,2))

        

    Theta = np.linspace(theta0, theta1, num)        

    Theta = np.append(Theta, Theta[-1]*np.ones(5))     #Repeat last value a few times to prolong the arc's appearance

    

    x = R * np.cos(Theta) + cx    

    y = R * np.sin(Theta) + cy

            

    if debug:

        return x, y, Theta

    else:

        return x, y

    

def scale_transform(x, side_len, start, end):

    x = (side_len * (x  - start)  / (end  - start))

    if isinstance(x, np.ndarray):

        return x.astype(int)

    else:

        return int(x)         
M = len(df)

rand_vec = np.random.randint(M, size=100)



lon_p = scale_transform( np.array(longitude_pick)[rand_vec], imageSize[1], longRange[0], longRange[1])

lat_p = scale_transform( np.array(latitude_pick)[rand_vec], imageSize[0], latRange[0], latRange[1])

lon_d = scale_transform( np.array(longitude_drop)[rand_vec], imageSize[1], longRange[0], longRange[1])

lat_d = scale_transform( np.array(latitude_drop)[rand_vec], imageSize[0], latRange[0], latRange[1])



longLim = scale_transform( np.array(xlim), imageSize[1], longRange[0], longRange[1])

latLim = scale_transform( np.array(ylim), imageSize[0], latRange[0], latRange[1])



fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(10,10))

ax1.imshow(np.log(locationDensityImage+1), cmap='Purples', zorder=0)



ax2 = ax1.twinx()

for i in range(len(rand_vec)):

    x, y, phi = circle_arc( lon_p[i],lat_p[i],lon_d[i],lat_d[i],alpha=200)      

    ax2.plot(x, y, color='magenta', linewidth = 0.4, zorder=1)    



ax2.set_ylim(latLim)

ax2.set_xlim(longLim)



ax1.set_axis_off()

ax2.set_axis_off()



plt.show()
df['pickup_datetime2'] = pd.to_datetime(df.pickup_datetime)

df = df.sort_values(by='pickup_datetime2')

df = df.reset_index(drop=True)



df['lon_pick'] = df['pickup_longitude'].apply(lambda x: scale_transform( x, imageSize[1], longRange[0], longRange[1]))

df['lat_pick'] = df['pickup_latitude'].apply(lambda x: scale_transform( x, imageSize[0], latRange[0], latRange[1]))

df['lon_drop'] = df['dropoff_longitude'].apply(lambda x: scale_transform( x, imageSize[1], longRange[0], longRange[1]))

df['lat_drop'] = df['dropoff_latitude'].apply(lambda x: scale_transform( x, imageSize[0], latRange[0], latRange[1]))
def get_CA(df):

    CA = []

    Phi = []

    for index, row in tqdm( df.iterrows(), miniters=10000):    

        X, Y, phi = circle_arc( row.lon_pick, row.lat_pick, \

                        row.lon_drop, row.lat_drop, \

                        alpha=200, time = row.trip_duration)       

        CA.append((X,Y))    

        Phi.append(phi)        

    return CA, Phi 



def get_frames(start, duration, df):

    

    end = start + timedelta(minutes=duration) 



    df2 = df[(df['pickup_datetime2'] >= start) & (df['pickup_datetime2'] <= end)]

    df2.is_copy = False  #This line disable SettingWithCopyWarning for df2

    

    CA, Phi = get_CA(df2)

    df2['circle_arc'] = CA  

    df2['angle'] = Phi



    time_arr = np.array( [ np.datetime64(start + timedelta(minutes=i)) for i in range(duration)])

    df2['pickup_datetime_in_minutes'] = df2['pickup_datetime2'].apply(lambda x: x.replace(second=0, microsecond=0))

    df2['trip_duration_in_minutes'] = df2['trip_duration'].apply(lambda x: np.ceil(x/60.0))

    df2['circel_arc_len'] = df2['circle_arc'].apply(lambda x: len(x[0])) 

    

    frame = []

    for i in range(duration):

        frame.append([])

        

    M = len(df2)

    print('Number of trips in animation: {}'.format(M))

    

    for i in range(M):

        idx = np.squeeze( np.where((time_arr == df2['pickup_datetime_in_minutes'].values[i]))[0])



        x = df2['circle_arc'].values[i][0]

        y = df2['circle_arc'].values[i][1]    



        theta = df2['angle'].values[i]

        R = 0.5 *( np.sin( theta + 0.66*np.pi) + 1)

        G = 0.5 *( np.sin( theta) + 1)

        B = 0.5 *( np.sin( theta + 1.33*np.pi) + 1)

        color = (R,G,B)



        k = len(x)

        for j in range(k-2):   

            if idx+j < duration:

                frame[idx+j].append( [x[0:j+2], y[0:j+2], color] )

                

    return frame, time_arr       
# initialization function: plot the background of each frame

def init():

    locationDensityImageRescaled = np.sqrt( np.log( locationDensityImage +1))    

    ax1.imshow(locationDensityImageRescaled, cmap='Purples', zorder=0)

    ax1.set_axis_off()

    

    #Legend diagram: color of path indicate dirrection from pickup to dropoff

    x1, y1, theta = get_circle_arc(0, 0, 0.56, np.radians(0), np.radians(360), debug=True)

    R = 0.5 *( np.sin( theta + 0.66*np.pi) +1)

    G = 0.5 *( np.sin( theta) +1)

    B = 0.5 *( np.sin( theta + 1.33*np.pi) +1)



    ax2 = plt.axes([0.7, 0.7, .2, .2])

    ax2.set_xlim([-1, 1])

    ax2.set_ylim([-1, 1])

    ax2.set_axis_off()

    

    for i in range(len(x1)):

        ax2.scatter(x1[i], y1[i], color=(R[i], G[i], B[i]))

        

# animation function. This is called sequentially

def animate(i):   

    ax3.clear()

    ax3.set_title('Taxi Trips represented as circle arc - DateTime: ' + \

                  pd.to_datetime(str(time_arr[i])).strftime('%Y.%m.%d %H:%M')) 

    ax3.set_ylim(latLim)

    ax3.set_xlim(longLim)      

    ax3.set_axis_off()

    

    for x, y, color in frame[i]:            

        ax3.plot(x, y, color=color, linewidth = 2)

        ax3.plot(x[-1], y[-1], 'o', color=color, markersize = 8)
# datetime(YYYY, MM, DD, HH, MM, SS)

start = datetime(2016, 3, 16, 5, 0, 0)

duration = 6 * 60 #in minutes



frame, time_arr = get_frames(start, duration, df)

fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(12,12))



#Set up ax3 for path animation

ax3 = ax1.twinx()



anim = animation.FuncAnimation(fig, animate, init_func=init, frames=duration)
start_time = time.time()



filename = 'animation.gif'

#The following line should save a gif file on your local drive if you run this locally.

anim.save(filename, writer='imagemagick', fps=5)

print('Video rendering takes: {:.1f} minutes'.format((time.time() - start_time)/60))



#For displaying it on the notebook

video = io.open(filename, 'r+b').read()

encoded = base64.b64encode(video)

HTML(data='''<img src="data:image/gif;base64,{0}" type="gif" />'''.format(encoded.decode('ascii'))) 
# #For saving it as mp4:

# start_time = time.time()

# Writer = animation.writers['ffmpeg']

# writer = Writer(fps=5, bitrate=1800)

# anim.save('anim.mp4', writer=writer)

# print('Video rendering takes: {:.1f} minutes'.format((time.time() - start_time)/60))



# #For display on notebook:

# start_time = time.time()

# HTML(anim.to_html5_video())

# print('Video rendering takes: {:.1f} minutes'.format((time.time() - start_time)/60))



# rc('animation', html='html5')

# anim
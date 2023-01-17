import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import matplotlib.animation as animation

from sklearn import neighbors

from PIL import Image,ImageDraw,ImageFont
## KNN PARAMETERS:

n_neighbors =100 # this parameter is optimized for 512x512 resolution and 30m temporal windows

weights = 'distance'



## LOAD DATA:

fields = ['pickup_datetime','pickup_latitude','pickup_longitude','dropoff_datetime','dropoff_latitude','dropoff_longitude']

parse_dates = ['pickup_datetime','dropoff_datetime']

df = pd.read_csv('../input/3march.csv',usecols=fields,parse_dates=parse_dates)

df.dropna(how='any',inplace=True)

dfsave = df



# DEFINE THE SPATIAL GRID

ymax = 40.85

ymin = 40.65

xmin = -74.06

xmax = xmin +(ymax-ymin)

X,Y = np.mgrid[xmin:xmax:512j,ymin:ymax:512j] # the spatial resolution can be tunned up 

positions = np.vstack([X.ravel(),Y.ravel()])
# RECONSTRUCT THE MAP FOR EVERY TIME FRAME 

Zs = []

time_step = 30 #time gates of 30 minutes

for h in range(4,10): #only between 4AM anbd 10AM

        for m in np.arange(0,60,time_step).astype(int):

                df = dfsave[dfsave.pickup_datetime.dt.weekday<5] # select only weekdays

                df = dfsave.groupby(dfsave.pickup_datetime.dt.hour).get_group(h)

                df = df[(df.pickup_datetime.dt.minute>=m) & (df.pickup_datetime.dt.minute<(m+time_step))]

                values_pickup = np.vstack([df.pickup_longitude.values,df.pickup_latitude.values])

                values_dropoff = np.vstack([df.dropoff_longitude.values,df.dropoff_latitude.values])

                values = np.hstack([values_pickup,values_dropoff])

                targets = np.hstack([np.ones((1,values_pickup.shape[1])),-np.ones((1,values_dropoff.shape[1]))])

                knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)

                Z = np.reshape(knn.fit(values.T,targets.T).predict(positions.T),X.shape)

                Zs.append(Z)
Writer = animation.writers['imagemagick']

writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)

fig = plt.figure()

movie = []

h = 0 

m = -30 

for i in range(len(Zs)):

        m = m + 30

        if m==60:

                h = h+1 

                m = 0 

        Z = 0.25*Zs[i-1]+0.5*Zs[i]   # temporal smoothing

        if (i+1)==len(Zs):

                Z = Z+0.25*Zs[0]

        else:

                Z = Z+0.25*Zs[i+1]

        Z = np.rot90(Z)

        frame = plt.imshow(Z,extent=[xmin,xmax,ymin,ymax],clim=[-1,1],cmap='RdBu',animated=True)

        movie.append([frame])



ani = animation.ArtistAnimation(fig,movie, interval=100, blit=False,repeat_delay=0)

ani.save('./animation_out.gif', writer=writer)
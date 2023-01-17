import numpy as np 

import pandas as pd 

import matplotlib

import matplotlib.pyplot as plt

import os







data_dir = '../input/nfl-big-data-bowl-2020/'

df = pd.read_csv(data_dir + 'train.csv')

plays = np.unique(df['PlayId'].values)
# Chose which play to plot 

play_n = 75



acc        = df[df['PlayId']==plays[play_n]]['A'].values

x_coord    = df[df['PlayId']==plays[play_n]]['X'].values

y_coord    = df[df['PlayId']==plays[play_n]]['Y'].values

direction  = df[df['PlayId']==plays[play_n]]['Dir'].values

play_dir   = df[df['PlayId']==plays[play_n]]['PlayDirection'].values

yard_line  = df[df['PlayId']==plays[play_n]]['YardLine'].values

rusher_id  = df[df['PlayId']==plays[play_n]]['NflIdRusher'].values

nfl_id     = df[df['PlayId']==plays[play_n]]['NflId'].values

team       = df[df['PlayId']==plays[play_n]]['Team'].values





a_x =  np.cos(direction * ((2 * np.pi)/360 + (np.pi/2)))

a_y =  np.sin(direction * ((2 * np.pi)/360 + (np.pi/2)))



rusher_idx =np.where(rusher_id == nfl_id)[0][0]

fig, ax = plt.subplots(figsize=(20,9), facecolor='grey')

ax.set_facecolor('green')

plt.ylim((0,53))

plt.xlim((0,120))

norm = matplotlib.colors.Normalize(vmin=acc.min(),vmax=acc.max())



plt.grid()

hw = 1.2

q_away = ax.quiver(x_coord[0:10], y_coord[0:10],

                   a_x[0:10], a_y[0:10],acc[0:10],

                   cmap='autumn', norm=norm, 

                   scale = 30, headwidth = hw)

q_home = ax.quiver(x_coord[11:21], y_coord[11:21],

                   a_x[11:21], a_y[11:21],acc[11:21],

                   cmap='winter', norm=norm, 

                   scale = 30, headwidth = hw)



plt.plot(x_coord[0:10],y_coord[0:10],'o',color='brown')

plt.plot(x_coord[11:21],y_coord[11:21],'o',color='blue')



plt.plot(x_coord[rusher_idx],y_coord[rusher_idx],'o',color='black')

ax.axvline(10,c='black') #Home Endzone

ax.axvline(110,c='black') #Away Endzone



cb_away = plt.colorbar(q_away)

cb_away.set_label('Away Team Acceleration')

cb_home = plt.colorbar(q_home)

cb_home.set_label('Home Team Acceleration')



plt.title('Play Direction: '+ play_dir[0])

plt.show()
# wip: Trying to make it more interactive

import holoviews as hv

from holoviews import opts, dim

hv.extension('bokeh')



hmap = hv.HoloMap({play_n: hv.VectorField((df[df['PlayId']==plays[play_n]]['X'].values,

                                           df[df['PlayId']==plays[play_n]]['Y'].values,

                                            df[df['PlayId']==plays[play_n]]['Dir'].values,

                                           df[df['PlayId']==plays[play_n]]['A'].values)).opts(color ='black')

                   for play_n in range(100) } , kdims='play number')  # Print only 100 for now



hmap_2 = hv.HoloMap({play_n: hv.VectorField((df[df['PlayId']==plays[play_n]]['X'].values[0:10],

                                           df[df['PlayId']==plays[play_n]]['Y'].values[0:10],

                                            df[df['PlayId']==plays[play_n]]['Dir'].values[0:10],

                                           df[df['PlayId']==plays[play_n]]['A'].values[0:10])).opts(color ='black')

                   for play_n in range(100) } , kdims='play number')  # Print only 100 for now



hmap.opts(width =600, height=320)

hmap.opts(bgcolor='green')
hmap_2.opts(width =600, height=320)

hmap_2.opts(bgcolor='green')
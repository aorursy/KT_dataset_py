# The %... is an iPython(Notebook) thing, and is not part of the Python language.

# In this case we're just telling the plotting library to draw things on

# the notebook, instead of on a separate window.

%matplotlib inline

# See all the "as ..." contructs? They're just aliasing the package names.

# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().

# All the required Packets

import numpy as np

import scipy as sp

import matplotlib as mpl

import matplotlib.cm as cm

import matplotlib.pyplot as plt

import pandas as pd

import time

pd.set_option('display.width', 1000)

pd.set_option('display.max_columns', 300)

pd.set_option('display.notebook_repr_html', True)

# Seaborn(Used for Graph plot)

import seaborn as sns

sns.set_style("whitegrid")

sns.set_context("poster")

plt.style.use('ggplot')

# Display of CSV file with required Parameters

data_frame = pd.read_csv("../input/Endurance.csv")

data_frame.head()
# Plot for Speed Per lap in form of Bar Garph.

data_lap_speed = data_frame.groupby('Lap').agg({'Speed': np.mean})

pop_state = data_lap_speed.Speed.sort_values(ascending=True)

plt.figure(figsize=(12, 5))

sns.barplot(y = pop_state,x = pop_state.index)



plt.title(" Speed Per Lap",size = 25)

plt.xlabel('Lap',size = 20)

plt.ylabel('Avg Speed',size = 20)
# Car Location plot with Speed Map

plt.figure(figsize=(20, 15))

data_frame_lap = data_frame[data_frame.Lap ==1]

data_frame_lap.Latitude = data_frame_lap.Latitude*100000

data_frame_lap.Longitude = data_frame_lap.Longitude*100000

plt.xlim(left = 800+7.558e6,right = 2600+7.558e6)

l = plt.scatter(data_frame_lap.Longitude , data_frame_lap.Latitude, c = data_frame_lap['Lap Travelled'],cmap='OrRd')

plt.colorbar()

plt.title(" Speed ColorMap : Lap 1",size = 25)

plt.xlabel('Latitude',size = 20)

plt.ylabel('Longitude',size = 20)

plt.show()
# Function to genreate graph for each Lap.

def print_colormap(x):

    plt.figure(figsize=(20, 16))

    data_frame_lap = data_frame[data_frame.Lap == x]

    data_frame_lap.Latitude = data_frame_lap.Latitude*100000

    data_frame_lap.Longitude = data_frame_lap.Longitude*100000

    plt.xlim(left = 800+7.558e6,right = 2600+7.558e6)

    plt.scatter(data_frame_lap.Longitude , data_frame_lap.Latitude, c = data_frame_lap.Speed,cmap='OrRd');

    plt.colorbar();

    plt.title(" Speed ColorMap : Lap " + str(x),size = 25)

    plt.xlabel('Latitude',size = 20)

    plt.ylabel('Longitude',size = 20)

    plt.show()

    #plt.savefig(str(x)+'.png')

    
#Only for the use of Saving The Plots

for i in range(1,19):

    print_colormap(i)
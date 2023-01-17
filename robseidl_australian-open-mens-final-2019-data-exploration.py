import os

import numpy as np

import pandas as pd



# Plotly and cufflinks for data analysis

from plotly.offline import init_notebook_mode, iplot

import cufflinks as cf

init_notebook_mode(connected=True) 



cf.set_config_file(theme='ggplot')

cf.go_offline()



# Matplotlib for drawing the court

import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse



%matplotlib inline
INPUT_DIR = '../input'

def load_data():

    return [pd.read_csv(os.path.join(INPUT_DIR, file_), index_col=0) for file_ in os.listdir(INPUT_DIR)]

        

serves, rallies, points, events = load_data()
# events

events.head()
# points

points.head()
# rallies

rallies.head()
# serves

serves.head()
points[['rallyid','winner']].groupby('winner').count()
points.groupby(['winner','serve']).size().reset_index(name='counts')
points.groupby('reason')['winner'].value_counts().unstack().iplot(kind='bar')


#### Tennis data



height_court = 10.97

width_court = 11.89*2

service_box = 6.4

double_field = 1.37

baseline_serviceline = 5.5

breite_einzel = 8.23

serviceline_net = 6.4





def draw_court(hide_axes=False):

    """Sets up field

    Returns matplotlib fig and axes objects.

    """

        

    fig = plt.figure(figsize=(height_court/2, width_court/2))

    #fig = plt.figure(figsize=(9, 9))

    fig.patch.set_facecolor('#5080B0')



    axes = fig.add_subplot(1, 1, 1, facecolor='#5080B0')



    if hide_axes:

        axes.xaxis.set_visible(False)

        axes.yaxis.set_visible(False)

        axes.axis('off')



    axes = draw_patches(axes)

    

    return fig, axes



def draw_patches(axes):

    plt.xlim([-2,height_court+2])

    plt.ylim([-6.5,width_court+6.5])

    

    #net

    axes.add_line(plt.Line2D([height_court, 0],[width_court/2, width_court/2],

                    c='w'))

    

    # court outline

    y = 0

    dy = width_court

    x = 0#height_court-double_field

    dx = height_court

    axes.add_patch(plt.Rectangle((x, y), dx, dy,

                       edgecolor="white", facecolor="#5581A6", alpha=1))

    # serving rect

    y = baseline_serviceline

    dy = serviceline_net*2

    x = 0 + double_field 

    dx = breite_einzel

    axes.add_patch(plt.Rectangle((x, y), dx, dy,

                       edgecolor="white", facecolor="none", alpha=1))

    

    #?

    #net

    axes.add_line(plt.Line2D([height_court/2, height_court/2], [width_court/2 - service_box, width_court/2 + service_box],

                    c='w'))

    

    axes.add_line(plt.Line2D([height_court/2, height_court/2], [0, 0 + 0.45], 

                    c='w'))



    axes.add_line(plt.Line2D([height_court/2, height_court/2], [width_court, width_court - 0.45], 

                c='w'))

    

    axes.add_line(plt.Line2D([1.37, 1.37], [0, width_court], 

            c='w'))

    

    axes.add_line(plt.Line2D( [height_court - 1.37, height_court - 1.37], [0, width_court],

        c='w'))



    return axes



fig, ax = draw_court();
def draw_players(axes):

    colors = {'djokovic': 'gray',

              'nadal': '#00529F'}

    

    size = 2

    color='white'

    edge=colors['djokovic']                        

    

    axes.add_artist(Ellipse((6,

                             -0.2),

                              size,size,

                              edgecolor=edge,

                              linewidth=2,

                              facecolor=color,

                              alpha=1,

                              zorder=20))

    axes.text(6-0.4,-0.2-0.2,'Dj',fontsize=14, color='black', zorder=30)

                

    

    edge=colors['nadal']

    axes.add_artist(Ellipse((1.75,

                             25),

                              size,size, 

                              edgecolor=edge,

                              linewidth=2,

                              facecolor=color,

                              alpha=1,

                              zorder=20))

    axes.text(1.75-0.4,25-0.15,'Na',fontsize=14, color='black', zorder=30)

    

    return axes



fig, ax = draw_court(hide_axes=True);

ax = draw_players(ax)
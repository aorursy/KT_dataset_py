from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.manifold import TSNE
# How to load a saved clustering later

data = pd.read_csv('/kaggle/input/clustering_B_final.csv')

labels = data['labels']

X = pd.read_csv('/kaggle/input/X_clustering_B_final.csv').to_numpy()
data.head()
tsne = TSNE(verbose=1)

X_embedded = tsne.fit_transform(X)
import bokeh

from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper, CustomJS, Slider, TapTool, TextInput, RadioButtonGroup

from bokeh.palettes import Category20

from bokeh.transform import linear_cmap

from bokeh.io import output_file, show

from bokeh.transform import transform

from bokeh.io import output_notebook

from bokeh.plotting import figure

from bokeh.layouts import column

from bokeh.models import RadioButtonGroup

from bokeh.models import TextInput

from bokeh.layouts import gridplot

from bokeh.models import Div

from bokeh.models import Paragraph

from bokeh.layouts import column, widgetbox

output_notebook()

y_labels = labels



title = data['title']

title = [text[0:40] for text in title]



# data sources

source = ColumnDataSource(data=dict(

    x= X_embedded[:,0], 

    y= X_embedded[:,1],

    x_backup = X_embedded[:,0],

    y_backup = X_embedded[:,1],

    desc= y_labels, 

    titles= title,

    abstract = data['abstract'],

    labels = ["C-" + str(x) for x in y_labels]

    ))



# hover over information

hover = HoverTool(tooltips=[

    ("Title", "@titles{safe}"),

    ("Abstract", "@abstract{safe}"),

],

                 point_policy="follow_mouse")



# map colors

mapper = linear_cmap(field_name='desc', 

                     palette=Category20[20],

                     low=min(y_labels) ,high=max(y_labels))



# prepare the figure

p = figure(plot_width=800, plot_height=800, 

           tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'], 

           title="t-SNE Covid-19 Articles, Clustered(K-Means), Tf-idf with Title, Abstract & Plain Text", 

           toolbar_location="right")



# plot

p.scatter('x', 'y', size=5, 

          source=source,

          fill_color=mapper,

          line_alpha=0.3,

          line_color="black",

          legend = 'labels')



#header

header = Div(text="""<h1>COVID-19 Research Papers Cluster - 2019/2020 </h1>""")



# show

show(column(header,p))
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import matplotlib.pyplot as plt

import bqplot

import numpy as np

import ipywidgets
# read in preprocessed data, saved to CSV last time

data = pd.read_csv('/kaggle/input/hxiuyu2dvpj/uci_har.csv')

data = data.astype({'label':'int'})

data = data[data.columns[1:]]
label_name = []

with open('/kaggle/input/hxydvpj/activity_labels.txt') as f:

    line = 'init'

    while line:

        line = f.readline().strip()

        if len(line) > 0:

            label_name.append(line.split()[1])
plt.hist(data.label,bins=6)

plt.xticks([1,2,3,4,5,6], label_name, rotation = 45)

plt.show()
# remove the first column from users' option

data = data[:100]

feature_names = data.columns

feature_names = feature_names[:5]

# make a color map

categories = np.unique(data['label'])

colors = ['#FA8072', '#800000', '#ADD8E6', '#228B22', '#FFFF00', '#800080']

colordict = dict(zip(categories, colors))  

# attach the color map to the dataframe

data["Color"] = data['label'].apply(lambda x: colordict[x])

# (1) scales

x_scl = bqplot.LinearScale() 

y_scl = bqplot.LinearScale()



# (2) Axis

ax_xcl = bqplot.Axis(label='X axis', scale=x_scl)

ax_ycl = bqplot.Axis(label='Y axis', scale=y_scl, orientation='vertical', side='left')



# (3) Marks

i,j = 0,0

x_name = feature_names[j]

y_name = feature_names[i]



activity_scatt = bqplot.Scatter(x = data[x_name],

                               y = data[y_name], 

                                colors = data.Color.values.tolist(),

                              scales={'x':x_scl, 'y':y_scl})

cor_matrix = data[feature_names].corr()

# (1) scales - colors, x & y

col_sc = bqplot.ColorScale(scheme="RdPu", 

                           min=np.nanmin(cor_matrix), 

                           max=np.nanmax(cor_matrix))

x_sc = bqplot.OrdinalScale()

y_sc = bqplot.OrdinalScale()



# (2) create axis - for colors, x & y

c_ax = bqplot.ColorAxis(scale = col_sc, 

                        orientation = 'vertical', 

                        side = 'right')



x_ax = bqplot.Axis(scale = x_sc, label='X Axis')

y_ax = bqplot.Axis(scale = y_sc, orientation = 'vertical', label = 'Y Axis')



# (3) Marks

heat_map = bqplot.GridHeatMap(color = cor_matrix,

                              row = feature_names, 

                              column = feature_names,

                              scales = {'color': col_sc,

                                        'row': y_sc,

                                        'column': x_sc},

                              interactions = {'click': 'select'},

                              anchor_style = {'fill':'blue'}, 

                              selected_style = {'opacity': 1.0},

                              unselected_style = {'opacity': 1.0})

myInfoLabel = ipywidgets.Label()

v = v = cor_matrix[x_name][y_name]

myInfoLabel.value = 'X axis is {}, y axis is {}, their correlation is {}'.format(x_name, y_name, v)





def get_data_value(change):

    if len(change['owner'].selected) == 1: #only 1 selected

        i,j = change['owner'].selected[0] 

        x_name = feature_names[j]

        y_name = feature_names[i]

        v = cor_matrix[x_name][y_name]

        myInfoLabel.value = 'X axis is {}, y axis is {}, their correlation is {}'.format(x_name, y_name, v)

        activity_scatt.x = data[x_name]

        activity_scatt.y = data[y_name]

        

        

heat_map.observe(get_data_value, 'selected')
# (5) create figures

fig_heatmap = bqplot.Figure(marks = [heat_map], axes = [c_ax, y_ax, x_ax])

fig_dur = bqplot.Figure(marks = [activity_scatt], axes = [ax_xcl, ax_ycl])

fig_heatmap.layout.min_width='500px'

fig_dur.layout.min_width='500px'



myDashboard = ipywidgets.VBox([myInfoLabel, ipywidgets.HBox([fig_heatmap,fig_dur])])

myDashboard
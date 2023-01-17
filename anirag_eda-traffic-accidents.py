# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.options.mode.chained_assignment = None



import plotly.plotly as py

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import iplot, init_notebook_mode

init_notebook_mode()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
monthly_data = pd.read_csv("../input/Traffic accidents by month of occurrence 2001-2014.csv")

time_data = pd.read_csv("../input/Traffic accidents by time of occurrence 2001-2014.csv")
monthly_data.head()
# Traffic accidents per year

accidents_peryear = np.asarray(monthly_data.groupby('YEAR').TOTAL.sum())



years = np.asarray(monthly_data.YEAR.unique())



data = [go.Scatter(

        x = years,

        y = accidents_peryear,

        mode = 'lines'

        )]



layout = go.Layout(

         title = 'Traffic Accidents by Year in India (2001-2014)',

         xaxis = dict(

             rangeslider = dict(thickness = 0.05),

             showline = True,

             showgrid = False

         ),

         yaxis = dict(

             range = [200000, 500000],

             showline = True,

             showgrid = False)

         )



figure = dict(data = data, layout = layout)

iplot(figure)
# Traffic Accidents by type per year

accidents_total = accidents_peryear

accidents_road = np.asarray(monthly_data[monthly_data.TYPE == 'Road Accidents'].groupby('YEAR').TOTAL.sum())

accidents_road_rail = np.asarray(monthly_data[monthly_data.TYPE == 'Rail-Road Accidents'].groupby('YEAR').TOTAL.sum())

accidents_other_rail = np.asarray(monthly_data[monthly_data.TYPE == 'Other Railway Accidents'].groupby('YEAR').TOTAL.sum())



accidents_road = np.round(np.divide(accidents_road, accidents_total) * 100, 1)

accidents_road_rail = np.round(np.divide(accidents_road_rail, accidents_total) * 100, 1)

accidents_other_rail = np.round(np.divide(accidents_other_rail, accidents_total) * 100, 1)



labels = ['Road Accidents', 'Rail-Road Accidents', 'Other Rail Accidents']

colors = ['rgb(252,141,89)', 'rgb(171,217,233)', 'rgb(52,109,67)']

x_data = years

y_data = np.asarray([accidents_road, accidents_road_rail, accidents_other_rail])



traces = []

for i in range(0, 3):

    traces.append(go.Scatter(

        x = x_data,

        y = y_data[i],

        mode = 'lines',

        name = labels[i],

        line = dict(color = colors[i], width = 3)

    ))

    



layout = go.Layout(

         title = 'Traffic accidents by Type in India (2001-2014)',

         showlegend = False,

         xaxis = dict(

             showline = True,

             showgrid = False

         ),

         yaxis = dict(

             ticksuffix = '%',

             showline = True,

             zeroline = False,

             showgrid = False,

             showticklabels = True

         ),

         margin = dict(

             autoexpand = False,

             l = 127, r = 38)

         )



annotations = []

for y_trace, label in zip(y_data, labels):

    annotations.append(dict(xref='paper', x=0.0475, y=y_trace[0],

                            xanchor='right', yanchor='middle',

                            text=label + ' {}%'.format(y_trace[0]),

                            showarrow=False))

    

annotations[1].update(yanchor='top')

layout['annotations'] = annotations



figure = dict(data = traces, layout = layout)

iplot(figure)
grouped = monthly_data.groupby('STATE/UT').TOTAL.sum().reset_index()

grouped.sort_values('TOTAL',ascending=True,inplace=True)

accidents_per_state = np.asarray(grouped['TOTAL'].values)

states = np.asarray(grouped['STATE/UT'].values)

accidents_percent = np.round(accidents_per_state / sum(accidents_per_state) * 100, 2).astype(str)

for i in range(len(accidents_percent)):

    accidents_percent[i] += '%'

data = [go.Bar(

        x = accidents_per_state,

        y = states,

        text = accidents_percent,

        orientation = 'h',

        hoverinfo = 'y+text',

        marker = dict(

            color = 'rgb(1, 97, 156)')

        )]



layout = go.Layout(

         title = 'Traffic Accidents by State in India (2001-2014)',

         xaxis = dict(

             showgrid = False,

             showticklabels = False

         ),

         yaxis = dict(

    tickfont=dict(

            family='Old Standard TT, serif',

            size=10,

            color='black'

        )),

         autosize = False,

         width = 1000,

         height = 1000

         )

annotations = []



for xd, yd in zip(accidents_per_state,states):

    annotations.append(dict(x = xd, y = yd,

                  text = str(xd),

                  xanchor = 'left',

                  yanchor = 'middle',

                  showarrow = False),

                )



layout['annotations'] = annotations



figure = dict(data = data, layout = layout)

iplot(figure)
from mpl_toolkits.basemap import Basemap



from matplotlib.patches import Polygon

from matplotlib.collections import PatchCollection

from matplotlib.colors import Normalize

import matplotlib.pyplot as plt

import matplotlib.cm



df_count = monthly_data.groupby('STATE/UT').agg({'TOTAL':'sum'}).reset_index()
df_count['STATE/UT'] = df_count['STATE/UT'].replace(['D & N Haveli'],'Dadra and Nagar Haveli')

df_count['STATE/UT'] = df_count['STATE/UT'].replace(['Delhi (Ut)'], 'NCT of Delhi')

df_count['STATE/UT'] = df_count['STATE/UT'].replace(['A & N Islands'], 'Andaman and Nicobar')

df_count['STATE/UT'] = df_count['STATE/UT'].replace(['Jammu & Kashmir'], 'Jammu and Kashmir')

df_count['STATE/UT'] = df_count['STATE/UT'].replace(['Daman & Diu'], 'Daman and Diu')
num_colors = 9

values = df_count['TOTAL']

cm = plt.get_cmap('Greens')

scheme = [cm(i / num_colors) for i in range(num_colors)]

bins = np.linspace(values.min(), values.max(), num_colors)

df_count['bin'] = np.digitize(values, bins) - 1

df_count.head()

''' matplotlib.style.use('map')

(http://)fig = plt.figure(figsize=(22, 12))

ax = fig.add_subplot(111, facecolor='w', frame_on=False)

fig.suptitle('Total Accidents by state (INDIA) in years 2001-2014', fontsize=30, y=.95)



m = Basemap(resolution='l',projection='merc',lat_0=20.59,lon_0=78.96,llcrnrlon=68.51, llcrnrlat= 6.93, urcrnrlon=97.4, urcrnrlat=35.51)

m.readshapefile("../input/INDIA_map_states","state");



for info, shape in zip(m.state_info, m.state):

    s = info['NAME_1']

    if s not in df_count['STATE/UT'].values:

        color = '#dddddd'

    else:

        color = scheme[int(df_count[df_count['STATE/UT']==s]['bin'])]



    patches = [Polygon(np.array(shape), True)]

    pc = PatchCollection(patches)

    pc.set_facecolor(color)

    ax.add_collection(pc)

    

# Draw color legend.

norm = Normalize()

mapper = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)

 

mapper.set_array(df_count['TOTAL'])

plt.colorbar(mapper, shrink=0.4)



# Set the map footer.

#plt.annotate(descripton, xy=(-.8, -3.2), size=14, xycoords='axes fraction')

'''
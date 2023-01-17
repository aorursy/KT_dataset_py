# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import zipfile



zf = zipfile.ZipFile('../input/bosch-production-line-performance/train_date.csv.zip') 

train_date_chunks = pd.read_csv(zf.open('train_date.csv'), iterator=True, chunksize=100000)

pd.options.display.max_columns = None

pd.options.display.max_rows = None

pd.options.display.max_colwidth = None
def get_date_frame():

    for data_frame in train_date_chunks:

        yield data_frame

        

get_df_date = get_date_frame()
df_date = next(get_df_date)



station_list = []

first_features_in_each_station = [] 



df_date_columns = df_date.columns.tolist()



for feature in df_date_columns[1:]:

    station = feature[:feature.index('_D')]

    if station in station_list:

        continue

    else:

        station_list.append(station)

        first_features_in_each_station.append(feature)
global_station_pairs = {}
while True:

    temp_df = pd.DataFrame (np.array(df_date[first_features_in_each_station]), columns = station_list)

    station_list_for_each_part = temp_df.stack().reset_index(level=1).groupby(level=0, sort=False)['level_1'].apply(list)

    

    

    temp_station_df = pd.DataFrame({"Id": np.array(df_date["Id"])})

    temp_station_df["Stations"] = station_list_for_each_part

    

    try:

        station_df = station_df.append(temp_station_df, ignore_index=True)

    except:

        station_df = pd.DataFrame({"Id": np.array(df_date["Id"])})

        station_df["Stations"] = station_list_for_each_part

    

    

    for each_part in station_list_for_each_part:

        for station_cursor in range(1, len(each_part)):

            pair = (each_part[station_cursor-1], each_part[station_cursor])

            try:

                global_station_pairs[pair] += 1

            except:

                global_station_pairs[pair] = 1

                

    try:

        df_date = next(get_df_date)

    except:

        break
station_df.head()
global_station_pairs = sorted(global_station_pairs.items())
global_station_pairs
node_occurrences = {}

for line in global_station_pairs:

    pair = line[0]

    try:

        node_occurrences[pair[0]] += 1

    except:

        node_occurrences[pair[0]] = 1

    try:

        node_occurrences[pair[1]] += 1

    except:

        node_occurrences[pair[1]] = 1
node_occurrences
'''

station_df.to_csv('part_station_info.csv', index=False)  



with zipfile.ZipFile('part_station_info.zip', 'w') as zipObj:

    zipObj.write("./part_station_info.csv")

    print("part_station_info.csv file is successfuly zipped")



try:

    os.remove("./part_station_info.csv")

    print("part_station_info.csv file is successfuly removed")

except:

    print("No such file")



'''
import networkx as nx

import matplotlib.pyplot as plt



import plotly.offline as py

import plotly.graph_objects as go
def create_pair_list(station_list):

    pair_list = []

    for each in station_list:

        for station_cursor in range(1, len(each)):

            pair = (each[station_cursor-1], each[station_cursor])

            pair_list.append(pair)

    return pair_list
def draw_station_pairs_in_a_part(pair_list, title):

    g = nx.Graph()

    for pair in pair_list:

        g.add_node(pair[0])

        g.add_node(pair[1])

        g.add_edge(pair[0], pair[1])

    

    fig = plt.figure(figsize=(10,5))

    nx.draw(g, with_labels=True)

    plt.title(title)

    plt.show()
random_sample = station_df.sample()

random_sample
draw_station_pairs_in_a_part(create_pair_list(random_sample['Stations']), int(random_sample['Id']))
random_sample = station_df.sample()

random_sample
draw_station_pairs_in_a_part(create_pair_list(random_sample['Stations']), int(random_sample['Id']))
g = nx.Graph()

for pair in global_station_pairs:

    stations = pair[0]



    g.add_node(stations[0], size = node_occurrences[stations[0]])

    g.add_node(stations[1], size = node_occurrences[stations[1]])

    g.add_edge(pair[0][0], pair[0][1], occurrence = pair[1])
pos_ = nx.spring_layout(g)
def make_edge(x, y, text, width):

    return  go.Scatter(x         = x,

                       y         = y,

                       line      = dict(width = width,

                                   color = 'red'),

                       hoverinfo = 'text',

                       text      = ([text]),

                       mode      = 'lines')
edge_trace = []

for edge in g.edges():

    ch1 = edge[0]

    ch2 = edge[1]

    

    x0, y0 = pos_[ch1]

    x1, y1 = pos_[ch2]

    

    trace  = make_edge([x0, x1, None], 

                       [y0, y1, None], 

                       text = g.edges()[edge]["occurrence"],

                       width = 0.000005*g.edges()[edge]["occurrence"])

    edge_trace.append(trace)
node_trace = go.Scatter(x         = [],

                        y         = [],

                        text      = [],

                        textposition = "top center",

                        textfont_size = 10,

                        mode      = 'markers+text',

                        hoverinfo = 'none',

                        marker    = dict(color = [],

                                         size  = [],

                                         line  = None))



for node in g.nodes():

    x, y = pos_[node]

    node_trace['x'] += tuple([x])

    node_trace['y'] += tuple([y])

    node_trace['marker']['color'] += tuple(['cornflowerblue'])

    node_trace['marker']['size'] += tuple([1*g.nodes()[node]['size']])

    node_trace['text'] += tuple(['<b>' + node + '</b>'])
# Customize layout

layout = go.Layout(

    paper_bgcolor='rgba(0,0,0,0)', # transparent background

    plot_bgcolor='rgba(0,0,0,0)', # transparent 2nd background

    xaxis =  {'showgrid': False, 'zeroline': False}, # no gridlines

    yaxis = {'showgrid': False, 'zeroline': False}, # no gridlines

)

# Create figure

fig = go.Figure(layout = layout)

# Add all edge traces

for trace in edge_trace:

    fig.add_trace(trace)

# Add node trace

fig.add_trace(node_trace)

# Remove legend

fig.update_layout(showlegend = False)

# Remove tick labels

fig.update_xaxes(showticklabels = False)

fig.update_yaxes(showticklabels = False)

# Show figure

fig.show()
fig = plt.figure(figsize=(20,20))

nx.draw(g, with_labels=True)

plt.show()
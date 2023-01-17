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



zf = zipfile.ZipFile('../input/bosch-production-line-performance/train_numeric.csv.zip') 

train_numeric_chunks = pd.read_csv(zf.open('train_numeric.csv'), iterator=True, chunksize=100000)



zf = zipfile.ZipFile('../input/bosch-production-line-performance/train_date.csv.zip') 

train_date_chunks = pd.read_csv(zf.open('train_date.csv'), iterator=True, chunksize=100000)



pd.options.display.max_columns = None

pd.options.display.max_rows = None

pd.options.display.max_colwidth = None
def get_numeric_frame():

    for data_frame in train_numeric_chunks:

        yield data_frame



get_df_numeric = get_numeric_frame()     

df_numeric = next(get_df_numeric)

        

def get_date_frame():

    for data_frame in train_date_chunks:

        yield data_frame

        

get_df_date = get_date_frame()

df_date = next(get_df_date)
df_date.insert(1, 'Response', df_numeric['Response'])

fail_parts_df = df_date.loc[df_date['Response'] == 1]
fail_parts_df.head(10)
# generates the station names

station_list = []

first_features_in_each_station = [] 



fail_parts_df_columns = fail_parts_df.columns.tolist()



for feature in fail_parts_df_columns[2:]:

    station = feature[:feature.index('_D')]

    if station in station_list:

        continue

    else:

        station_list.append(station)

        first_features_in_each_station.append(feature)
global_station_pairs = {}



L0_to_L0 = {}

L0_to_L1 = {}

L0_to_L2 = {}

L0_to_L3 = {}



L1_to_L1 = {}

L1_to_L2 = {}

L1_to_L3 = {}



L2_to_L2 = {}

L2_to_L3 = {}



L3_to_L3 = {}
def find_line_dict(pair):

    from_line = pair[0][:2]

    to_line = pair[1][:2]

    

    if from_line == 'L0' and to_line == 'L0':

        place_line_dict(pair, L0_to_L0)

    elif from_line == 'L0' and to_line == 'L1': 

        place_line_dict(pair, L0_to_L1)

    elif from_line == 'L0' and to_line == 'L2': 

        place_line_dict(pair, L0_to_L2)

    elif from_line == 'L0' and to_line == 'L3': 

        place_line_dict(pair, L0_to_L3)

        

    

    elif from_line == 'L1' and to_line == 'L1': 

        place_line_dict(pair, L1_to_L1)

    elif from_line == 'L1' and to_line == 'L2': 

        place_line_dict(pair, L1_to_L2)

    elif from_line == 'L1' and to_line == 'L3': 

        place_line_dict(pair, L1_to_L3)



       

    elif from_line == 'L2' and to_line == 'L2': 

        place_line_dict(pair, L2_to_L2)

    elif from_line == 'L2' and to_line == 'L3': 

        place_line_dict(pair, L2_to_L3)

    

    elif from_line == 'L3' and to_line == 'L3': 

        place_line_dict(pair, L3_to_L3)



        

def place_line_dict(pair, line_dict):

    try:

        line_dict[pair] += 1

    except:

        line_dict[pair] = 1
while True:

    temp_df = pd.DataFrame (np.array(fail_parts_df[first_features_in_each_station]), columns = station_list)

    station_list_for_each_part = temp_df.stack().reset_index(level=1).groupby(level=0, sort=False)['level_1'].apply(list)

    

    for each_part in station_list_for_each_part:

        for station_cursor in range(1, len(each_part)):

            pair = (each_part[station_cursor-1], each_part[station_cursor])

            

            find_line_dict(pair)

            try:

                global_station_pairs[pair] += 1

            except:

                global_station_pairs[pair] = 1



    try:

        df_date = next(get_df_date)

    except:

        break
def find_max_min_dict(station_pair, pair_name):

    try:

        values = station_pair.values()

        return {"max":max(values), "min":min(values)}

    except:

        print("No fail parts passed from the line {}".format(pair_name))



min_max_global = find_max_min_dict(global_station_pairs, "all")



min_max_L0_L0 = find_max_min_dict(L0_to_L0, "L0_to_L0")

min_max_L0_L1 = find_max_min_dict(L0_to_L1, "L0_to_L1")

min_max_L0_L2 = find_max_min_dict(L0_to_L2, "L0_to_L2")

min_max_L0_L3 = find_max_min_dict(L0_to_L3, "L0_to_L3")



min_max_L1_L1 = find_max_min_dict(L1_to_L1, "L1_to_L1")

min_max_L1_L2 = find_max_min_dict(L1_to_L2, "L1_to_L2")

min_max_L1_L3 = find_max_min_dict(L1_to_L3, "L1_to_L3")



min_max_L2_L2 = find_max_min_dict(L2_to_L2, "L2_to_L2")

min_max_L2_L3 = find_max_min_dict(L2_to_L3, "L2_to_L3")



min_max_L3_L3 = find_max_min_dict(L3_to_L3, "L3_to_L3")
def sorting_method(item):

    el_1 = item[0][0]

    x = int(el_1[1:el_1.index('_')])

    y = int(el_1[el_1.index('S')+1:])

    

    el_2 = item[0][1]

    z = int(el_2[1:el_2.index('_')])

    t = int(el_2[el_2.index('S')+1:])

    return(x, y, z, t)



global_station_pairs = sorted(global_station_pairs.items(), key = sorting_method)



L0_to_L0 = sorted(L0_to_L0.items(), key = sorting_method)

L0_to_L1 = sorted(L0_to_L1.items(), key = sorting_method)

L0_to_L2 = sorted(L0_to_L2.items(), key = sorting_method)

L0_to_L3 = sorted(L0_to_L3.items(), key = sorting_method)



L1_to_L1 = sorted(L1_to_L1.items(), key = sorting_method)

L1_to_L2 = sorted(L1_to_L2.items(), key = sorting_method)

L1_to_L3 = sorted(L1_to_L3.items(), key = sorting_method)



L2_to_L2 = sorted(L2_to_L2.items(), key = sorting_method)

L2_to_L3 = sorted(L2_to_L3.items(), key = sorting_method)



L3_to_L3 = sorted(L3_to_L3.items(), key = sorting_method)
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
import networkx as nx

import matplotlib.pyplot as plt



import plotly.offline as py

import plotly.graph_objects as go
prog = ['dot', 'neato', 'fdp', 'twopi', 'circo'] #layout types



my_darkorchid = 'rgba(153,50,204,0.5)'

my_yellow = 'rgba(255,255,153,0.7)'

my_blue= 'rgba(221,243,245,1)'



mapping_range_minimum = 1

mapping_range_maximum = 10



mapping_range = mapping_range_maximum - mapping_range_minimum
def make_graph(pair_set, layout_style):

    graph = nx.Graph()

    for pair in pair_set:

        stations = pair[0]

        graph.add_node(stations[0], size = node_occurrences[stations[0]])

        graph.add_node(stations[1], size = node_occurrences[stations[1]])

        graph.add_edge(pair[0][0], pair[0][1], occurrence = pair[1])

    

    pos_ = nx.drawing.nx_pydot.graphviz_layout(graph, prog=layout_style)

    return graph, pos_



def make_edge(x, y, color, width, text):

    return  go.Scatter(x         = x,

                       y         = y,

                       line      = dict(width = width,

                                        color = color),

                       mode      = 'lines',

                       hoverinfo = "text",

                       text      = text)



def make_edge_trace(min_max_flow, graph, pos_, color = my_darkorchid):

    edge_trace = []

    

    min_flow = min_max_flow["min"]

    max_flow = min_max_flow["max"]

    

    for edge in graph.edges():

        ch1 = edge[0]

        ch2 = edge[1]



        x0, y0 = pos_[ch1]

        x1, y1 = pos_[ch2]

        

        flow = graph.edges()[edge]["occurrence"]

        

        try:

            line_width = (float(mapping_range * (flow - min_flow)) / (max_flow-min_flow)) + mapping_range_minimum

        except:

            line_width = 1 # exception occurs if there is a div by 0

        

        trace  = make_edge(x = [x0, (x0+x1)/2, x1], 

                           y = [y0, (y0+y1)/2, y1], 

                           color = color,

                           width = line_width,

                           text = ["",str(flow),""])

        

        edge_trace.append(trace)

    return edge_trace





def make_node_trace(graph, pos_, is_all_stations):

    node_x = []

    node_y = []

    node_size_l = []

    node_text_l = []

    

    for node in graph.nodes():

        x, y = pos_[node]

        node_x.append(x)

        node_y.append(y)

        

        if is_all_stations:

            node_size = graph.nodes()[node]['size']

        else:

            node_size = 20

        

        node_size_l.append(node_size)

        node_text_l.append(str(node))

    

    node_trace = go.Scatter(x         = node_x,

                            y         = node_y,

                            text      = node_text_l,

                            textposition = "top center",

                            textfont_size = 12,

                            mode      = 'markers+text',

                            hoverinfo = "text",

                            marker    = dict(size  = node_size_l,

                                             line_width=3,

                                             color=[],

                                             showscale=True,

                                             reversescale=True,

                                             colorscale='Viridis',

                                             colorbar=dict(

                                                    thickness=15,

                                                    title='Number of Node Connections',

                                                    xanchor='left',

                                                    titleside='right'),

                                             ))

    

    node_adjacencies = []

    for node, adjacencies in enumerate(graph.adjacency()):

        node_adjacencies.append(len(adjacencies[1]))

    node_trace.marker.color = node_adjacencies

    

    return node_trace

        

    

def draw(title, pair_set, min_max_flow=min_max_global, is_all_stations = False, layout_style = prog[4]):

    graph, pos_ = make_graph(pair_set, layout_style)

    edge_trace = make_edge_trace(min_max_flow, graph, pos_)

    node_trace = make_node_trace(graph, pos_, is_all_stations)

    

    layout = go.Layout(

        title = {'text':'Fail parts ' + title, 'x':0.5},

        paper_bgcolor=my_blue, 

        plot_bgcolor=my_blue, 

        xaxis =  {'showgrid': False, 'zeroline': False},

        yaxis = {'showgrid': False, 'zeroline': False},

        title_font_color="red",

        hovermode='closest',

        hoverlabel=dict(

            font_size=30, 

            font_family="Rockwell")

    )



    fig = go.Figure(layout = layout)

    

    fig.add_trace(node_trace)

    for edge in edge_trace:

        fig.add_trace(edge)

        

    fig.update_layout(showlegend = False)

    fig.update_xaxes(showticklabels = False)

    fig.update_yaxes(showticklabels = False)

    fig.show() 
draw(title = "L0_to_L0",

     pair_set = L0_to_L0, 

    )
print('NO FAIL PARTS IN L0_to_L1')
draw(title = "L0_to_L2",

     pair_set = L0_to_L2, 

    )
draw(title = "L0_to_L3",

     pair_set = L0_to_L3, 

    )
print('NO FAIL PARTS IN L1_to_L1')
draw(title = "L1_to_L2",

     pair_set = L1_to_L2, 

    )
draw(title = "L1_to_L3",

     pair_set = L1_to_L3, 

    )
print('NO FAIL PARTS IN L2_to_L2')
draw(title = "L2_to_L3",

     pair_set = L2_to_L3, 

    )
draw(title = "L3_to_L3",

     pair_set = L3_to_L3, 

    )
draw(title = "ALL STATIONS",

     pair_set = global_station_pairs,

     is_all_stations = True,

    )
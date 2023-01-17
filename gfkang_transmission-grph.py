# importing packages

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!pip install geopy

!pip install geotext

from geotext import GeoText

from geopy.geocoders import Nominatim
travel_data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

travel_data = travel_data[['location', 'summary','reporting date']]

travel_data = travel_data.dropna().reset_index(drop=True)

#travel_data = travel_data.reset_index(drop=True)

#travel_data = travel_data[['location', 'summary']]

travel_data2 = travel_data

from geotext import GeoText

for i in range (0, 1080):

    travel_data2['summary'][i] = GeoText((travel_data2['summary'][i])).cities

for i in range (0, 1080):

    if travel_data2['summary'][i] == []:

        travel_data2 = travel_data2.drop([i])

#travel_data_smp['summary'] = travel_data_smp['summary'].map()



pd.set_option('display.max_rows', None)

travel_data2 = travel_data2.reset_index(drop=True)

travel_data2.head()
travel_data3 = travel_data2

travel_data4 = travel_data3[['location', 'summary', 'reporting date']]

travel_data_final = pd.DataFrame({"location": [], "summary": [], "reporting date": []})

for i in range (0, 481):

    #if len(travel_data4['summary'][i]) < 2:

        #travel_data4['summary'][i] = travel_data4['summary'][i][0]

    #else:

        #travel_data4['summary'][i] = travel_data4['summary'][i][1]

    for j in range(0, len(travel_data4['summary'][i])):

        df2 = pd.DataFrame([[travel_data4['location'][i],travel_data4['summary'][i][j], 

                            travel_data4['reporting date'][i]]], columns=['location','summary', 'reporting date'])

        travel_data_final = pd.concat([df2, travel_data_final])

        #travel_data_final = travel_data_final.append({'location': travel_data4['location'][i]}, 

                                                     #{'summary': travel_data4['summary'][i][j]})

#for i in range (0, len(travel_data_final)):

    #if (a == b):

        #travel_data_final.drop([i])

travel_data_final = travel_data_final.reset_index(drop=True)

travel_data_final = travel_data_final.reindex(index=travel_data_final.index[::-1])

travel_data_final = travel_data_final.reset_index(drop=True)

travel_data_final.head()

#travel_data_final.head()
travel_data_final['reporting date'] = pd.to_datetime(travel_data_final['reporting date']).dt.strftime('%m/%d/%Y')

travel_data_final = travel_data_final.sort_values(by=['reporting date'])

travel_data_final = travel_data_final.reset_index(drop=True)

print(travel_data_final.shape)

travel_data_final
curr = '01/13/2020'

count = 0

for i in range(0, 683):

    if (travel_data_final['reporting date'][i] == curr):

        travel_data_final['reporting date'][i] = count

    else:

        curr = travel_data_final['reporting date'][i]

        count += 1

        travel_data_final['reporting date'][i] = count

travel_data_final
import matplotlib.pyplot as plt

import networkx as nx

import pylab

from matplotlib.pyplot import pause

import networkx as nx

pylab.ion()
travel_data_final = travel_data_final[travel_data_final['location']!='Hubei']

#travel_data_final = travel_data_final[travel_data_final['location']!='Texas']

travel_data_final = travel_data_final[travel_data_final['summary']!='Hubei']

travel_data_final = travel_data_final[travel_data_final['location']!='York']

travel_data_final = travel_data_final[travel_data_final['summary']!='York']

travel_data_final = travel_data_final[travel_data_final['summary']!='Kumamoto']

travel_data_final = travel_data_final[travel_data_final['summary']!='Tyumen']

travel_data_final = travel_data_final[travel_data_final['location']!='Weihai']

travel_data_final = travel_data_final[travel_data_final['summary']!='Weihai']

travel_data_final = travel_data_final[travel_data_final['location']!='Fo Tan']

travel_data_final = travel_data_final[travel_data_final['summary']!='Fo Tan']

travel_data_final = travel_data_final[travel_data_final['location']!='Incheon']

travel_data_final = travel_data_final[travel_data_final['summary']!='Okinawa']

travel_data_final = travel_data_final[travel_data_final['summary']!='Incheon']

travel_data_final = travel_data_final[travel_data_final['location']!='Lile']

travel_data_final = travel_data_final[travel_data_final['summary']!=travel_data_final['location']]

travel_data_final = travel_data_final.reset_index(drop=True)

travel_data_final.insert(loc = 2, column = 'weight', value = 0)

travel_data_final.head()
from sklearn import preprocessing
travel_data_final = travel_data_final[travel_data_final['weight']!=float('inf')]

travel_data_final = travel_data_final.reset_index(drop=True)

travel_data_final.shape

travel_data_2 = travel_data_final
travel_data_final = travel_data_final.reset_index(drop=True)

travel_data_final['weight'] = travel_data_final['reporting date']

x = travel_data_final[['weight']].values.astype(float)



# Create a minimum and maximum processor object

min_max_scaler = preprocessing.MinMaxScaler()



# Create an object to transform the data to fit minmax processor

x_scaled = min_max_scaler.fit_transform(x)



# Run the normalizer on the dataframe

df_normalized = pd.DataFrame(x_scaled)



travel_data_final['weight'] = df_normalized

travel_data_final['weight'] = 10 * travel_data_final['weight'] 

travel_data_final
for i in range (0, 468):

    if travel_data_final['weight'][i] < 1:

        travel_data_final['weight'][i] = 1

travel_data_final
os.makedirs('/kaggle/working/frames')
G = []

for i in range (0, 431):

    #if (travel_data_final['location'][i] != 'Jilin'):

    G.append((travel_data_final['summary'][i], travel_data_final['location'][i], travel_data_final['reporting date'][i]))

    

g = nx.Graph()

h = nx.Graph()

#for i in range (0, 100):

 #       h.add_node(G['location'][i])

#for i in range (0, 100):

#        edge = (travel_data_final['location'][i], travel_data_final['summary'][i])

#        h.add_edge(*edge)



for i in range (0, len(G)):

    edge = (G[i][0], G[i][1])

    g.add_edge(*edge, len = G[i][2])

#g.add_edges_from(G)



num_nodes = len(g)

node_list = list(g.nodes)

for i in range (0, num_nodes):

    if (not nx.has_path(g, target = node_list[i], source = 'Wuhan')):

        g.remove_node(node_list[i])



#for i in range (0, len(G)):

#    if (not nx.has_path(g, target = G[i][1], source = 'Wuhan')):

#        g.remove_node(G[i][1])



h.add_node('Wuhan')



        

#h.remove_edge('Tokyo','Japan')

#h.remove_edge('Tokyo','Chiba Prefecture')

#h.remove_edge('Malaysia','Macau')

#h.remove_edge('Macau','Taiwan')

#h.remove_edge('Beijing','Tianjin')

#h.remove_edge('Shaanxi','Xianyang')

#h.remove_edge('Hokkaido','Tokyo')

#h.remove_edge('Chongqing','Thailand')

#h.remove_edge('Macau','Hong Kong')

#h.remove_edge('Shenzhen','Hong Kong')

#h.remove_edge('Shenzhen','NSW')

#h.remove_node('Mudanjiang')

g.remove_edge('Guangzhou', 'Toronto')

#h.remove_node('Fo Tan')

#h.remove_node('Chiba')

#h.remove_node('Qingdao')

#h.remove_node('Male')

#h.remove_node('Hangzhou')

#h.remove_node('Heilongjiang')

#h.remove_node('Guangzhou')



node_list = list(h.nodes)





#my_pos = nx.spring_layout(g, seed = 100)

my_pos = nx.nx_pydot.graphviz_layout(g, prog='neato', root = 'Wuhan')

#my_pos = nx.nx_pydot.graphviz_layout(h, prog='twopi', root = 'Wuhan')

#my_pos = nx.nx_pydot.graphviz_layout(h, prog='twopi', root = 'Wuhan')

#my_pos[2] =  np.array([0, 0])

#my_pos = nx.nx_pydot.graphviz_layout(h, prog='neato')



NODES = ['Wuhan']

EDGES = []

LABELS = []



def get_fig(date):

    plt.clf()

    new_nodes = travel_data_final[travel_data_final['reporting date']==date]

    new_nodes = new_nodes.reset_index(drop=True)

    for i in range (0, len(new_nodes)):

        h.add_node(new_nodes['location'][i])

        h.add_node(new_nodes['summary'][i])

        h.add_edge(new_nodes['location'][i], new_nodes['summary'][i])

        if (not nx.has_path(h, target = new_nodes['location'][i], source = 'Wuhan')):

            h.remove_node(new_nodes['location'][i])

        if (not nx.has_path(h, target = new_nodes['summary'][i], source = 'Wuhan')):

            h.remove_node(new_nodes['summary'][i])



    node_list = []

    for i in range (0, len(new_nodes)):

        if (g.has_node(new_nodes['location'][i])):

            node_list.append(new_nodes['location'][i])

        if (g.has_node(new_nodes['summary'][i])):

            node_list.append(new_nodes['summary'][i])

    #my_pos = nx.spring_layout(h, seed = 100)

    #my_pos = nx.nx_pydot.graphviz_layout(h, prog='twopi', root = 'Wuhan')

    

    #pos = nx.nx_pydot.graphviz_layout(g, prog='neato')

    #nx.draw(g, pos=pos, with_labels=True, node_size= 50, edge_color="red",arrows = False, connectionstyle="rad=0.2")

    nx.draw(g, pos=my_pos, with_labels=False, node_color = "black", alpha = 0, node_size= 10, 

            edge_color="black",arrows = False, connectionstyle="rad=0.2", weight = 'length')

    nx.draw_networkx_edges(h, pos = my_pos

    #                   edgelist=EDGES,

                        ,edge_color='grey')

    #for j in range (0, i):

    #    NODES.append(EDGES[j][0])

    #    NODES.append(EDGES[j][1])

    nx.draw_networkx_nodes(h, pos = my_pos, #with_labels = True, 

    #                   nodelist=NODES,

                        #node_color='r',

                       node_size=20,

                       alpha=0.8)

    

    nx.draw_networkx_nodes(h, pos = my_pos, #with_labels = True, 

                       nodelist=node_list,

                       node_color='r',

                       node_size=20,

                       alpha=0.8)

    

    #nx.draw_networkx_nodes(h, pos = my_pos, #with_labels = True, 

    #                   nodelist=CURR,

    #                   node_color='r',

    #                   node_size=50,

    #                   alpha=0.8)

    



    #hub_ego = nx.ego_graph(h, 'Wuhan', radius = r)

    #pos = nx.circular_layout(hub_ego)

    #pos = nx.nx_pydot.graphviz_layout(h, prog='twopi')

    #nx.draw(hub_ego, pos = my_pos,edge_color = 'grey', node_size=50, with_labels=False)

    #nx.draw_networkx_labels(g, pos=my_pos, font_size = 8)

    

    

    

    



num_plots = 46

j = 0

pylab.show()



#for i in range(num_plots):

#for i in range(0, 99):

    #print(chr(27) + "[2J")

#for r in range (0, 3):

#for i in range (1, 100):

for i in range (1, 178):

#if (True):

    #i = 1

    print(i)

    get_fig(i)

    pylab.draw()

    plt.title(str(i))

    pylab.savefig('frames/frame' + str(i) + '.png')

    #pause(0.01)

    #j = j+1
import imageio

images = []

for i in range (1, 145):

    images.append(imageio.imread('/kaggle/working/test' + str(i) + '.png'))

imageio.mimsave('hgrph.gif', images, duration = 0.3)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
##################################################################################################################
# Corona Virus India Network based on individualdetails.csv file from Kaggle Dataset on Novel Corona Virus Disease 2019 in India downloaded on 26 March 2020.
# The aim of this notebook is to visualize Unique id and Contacts as a network.
# Nodes represent Unique id or Contacts. 
# Edges represent the countries where the Unique id or Contacts have travelled.
#####################################################################################################################


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.



#Import Modules
import pandas as pd
import numpy as np
import re
from time import strptime
from datetime import datetime
import nltk
import matplotlib.pyplot as plt
import networkx as nx
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.sem.relextract import extract_rels, rtuple
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk import Tree
import csv
from operator import itemgetter
import networkx as nx
from networkx.algorithms import community 
import os
import folium
from folium.plugins import FloatImage
from folium.features import DivIcon
import geopy
from  geopy.geocoders import Nominatim
import itertools
from IPython.display import clear_output
import time
import plotly.express as px
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 1000)


def getRelBetnPersonLoc(x):
    chunked = ne_chunk(pos_tag(word_tokenize(x)))
    pat = re.compile('.*\S.*')
    rels = extract_rels('PER', 'GPE', chunked, corpus = 'ace', pattern = pat)
    for rel in rels:
        return(rtuple(rel))


def get_locations(text):
    # Tokenize work, POS tag, and NER
    tokenizeText  = word_tokenize(text)
    posTagText = nltk.pos_tag(tokenizeText)
    namedEnt1= nltk.ne_chunk(posTagText)
    #namedEnt1.draw()
    namedEnt2 = []
    for i in namedEnt1:
        if hasattr(i, 'label'):
            NE_Name = ' '.join(x[0] for x in i.leaves())
            NE_Type = i.label()
            namedEnt2.append([NE_Name, NE_Type]) 
    #display(namedEnt2)
    tmp = list(filter(lambda x:x[1]=='GPE',namedEnt2))
    countries = []
    for i in range(len(tmp)):
        countries.append(tmp[i][0])
    return countries 


#def fun1(x):
    #return datetime.strptime(x, '%d/%b/%Y').date().strftime("%d-%m-%Y")

def fun1(x):
    return datetime.strptime(x, '%d-%b-%y').date().strftime("%d-%b-%Y")

def fun2(x):
    return datetime.strptime(x, '%m/%d/%Y').date().strftime("%d-%m-%Y") 

def fun3(x):
    tmp = ''.join(map(str, x))
    return tmp

def fun4(x):
    tmp = re.findall(r'CP\d+',x)
    return (", ".join(tmp))


# Read data
input_data_chunk = pd.read_csv("/kaggle/input/covid19-india/individualdetails.csv", chunksize=100, iterator=True, skipinitialspace=True, index_col=False)  
input_data = pd.concat(input_data_chunk, ignore_index=True)
input_data = input_data[['Unique id', 'Diagnosed date', 'Age', 'Gender', 'Detected state', 'Detected district', 'Nationality', 'Current status', 'Status change date', 'Notes', 'Contacts' ]]
display("Input data shape", input_data.shape)
display("Input data head", input_data.head())


# Process data
input_data[['Unique id', 'Diagnosed date', 'Gender', 'Detected state', 'Detected district', 'Nationality', 'Current status', 'Status change date', 'Notes', 'Contacts']] = input_data[['Unique id', 'Diagnosed date', 'Gender', 'Detected state', 'Detected district', 'Nationality', 'Current status', 'Status change date', 'Notes',  'Contacts']].astype(str)
input_data[['Age']] = input_data[['Age']].apply(pd.to_numeric)
input_data = input_data.dropna(how='any')
input_data = input_data.drop(input_data[input_data.Contacts == 'nan'].index)



input_data['Diagnosed date'] = input_data['Diagnosed date'].apply(fun1)
input_data['Diagnosed date'] =  pd.to_datetime(input_data['Diagnosed date'], format="%d-%b-%Y")
display("Input data head", input_data.head())





input_data['Status change date'] = input_data['Status change date'].apply(fun2)
input_data['Status change date'] =  pd.to_datetime(input_data['Status change date'], format="%d-%m-%Y")

input_data['DiagDate-StatusChDate'] = input_data['Status change date'] - input_data['Diagnosed date']

input_data['Contacts'] = input_data['Contacts'].str.replace('Patient ', 'CP') 

# Create separate columns for contacted patients using the Contacts column
input_data_expanded = input_data.join(input_data['Contacts'].str.split(',', expand=True).add_prefix('Contacts').fillna(np.nan))
input_data_expanded = input_data_expanded.drop('Contacts', 1)

input_data_expanded.rename(columns={'Unique id': 'PatientID'}, inplace=True)
input_data_expanded['PatientID'] = 'P' + input_data_expanded['PatientID'].astype(str)

input_data_expanded['travelledFrom'] = input_data_expanded['Notes'].apply(get_locations)

input_data_expanded['travelledFrom'] = input_data_expanded['travelledFrom'].apply(fun3)
input_data_expanded = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts0', 'Contacts1', 'Contacts2', 'Contacts3', 'Contacts4', 'Contacts5', 'Contacts6', 'Contacts7', 'Contacts8', 'Contacts9', 'Contacts10', 'Contacts11', 'Contacts12', 'Contacts13', 'Contacts14', 'Contacts15']]

# Create edgelist for network
tmp = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts0']]

tmp1 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts1']]
tmp1.columns = tmp.columns
tmp1 = pd.concat([tmp, tmp1],  axis=0, ignore_index=True)

tmp2 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts2']]
tmp2.columns = tmp.columns
tmp2 = pd.concat([tmp1, tmp2],  axis=0, ignore_index=True)

tmp3 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts3']]
tmp3.columns = tmp.columns
tmp3 = pd.concat([tmp2, tmp3],  axis=0, ignore_index=True)

tmp4 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts4']]
tmp4.columns = tmp.columns
tmp4 = pd.concat([tmp3, tmp4],  axis=0, ignore_index=True)

tmp5 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts5']]
tmp5.columns = tmp.columns
tmp5 = pd.concat([tmp4, tmp5],  axis=0, ignore_index=True)

tmp6 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts6']]
tmp6.columns = tmp.columns
tmp6 = pd.concat([tmp5, tmp6],  axis=0, ignore_index=True)

tmp7 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts7']]
tmp7.columns = tmp.columns
tmp7 = pd.concat([tmp6, tmp7],  axis=0, ignore_index=True)

tmp8 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts8']]
tmp8.columns = tmp.columns
tmp8 = pd.concat([tmp7, tmp8],  axis=0, ignore_index=True)

tmp9 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts9']]
tmp9.columns = tmp.columns
tmp9 = pd.concat([tmp8, tmp9],  axis=0, ignore_index=True)

tmp10 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts10']]
tmp10.columns = tmp.columns
tmp10 = pd.concat([tmp9, tmp10],  axis=0, ignore_index=True)

tmp11 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts11']]
tmp11.columns = tmp.columns
tmp11 = pd.concat([tmp10, tmp11],  axis=0, ignore_index=True)

tmp12 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts12']]
tmp12.columns = tmp.columns
tmp12 = pd.concat([tmp11, tmp12],  axis=0, ignore_index=True)

tmp13 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts13']]
tmp13.columns = tmp.columns
tmp13 = pd.concat([tmp12, tmp13],  axis=0, ignore_index=True)

tmp14 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts14']]
tmp14.columns = tmp.columns
tmp14 = pd.concat([tmp13, tmp14],  axis=0, ignore_index=True)

tmp15 = input_data_expanded[['travelledFrom', 'PatientID', 'Contacts15']]
tmp15.columns = tmp.columns
tmp15 = pd.concat([tmp14, tmp15],  axis=0, ignore_index=True)

edgelist = tmp15[pd.notnull(tmp15['Contacts0'])]
edgelist = edgelist.replace(r'^\s*$', 'NA', regex=True)

edgelist = edgelist.groupby(['PatientID','Contacts0', 'travelledFrom']).size().reset_index(name='weight')

edgelist['Contacts0'] = edgelist['Contacts0'].apply(fun4)

#edgelist.loc[0, 'weight'] = 1

display("Processed data with edgelist of Patients and their Contact Patients", edgelist.head())



# Plot network of patients and their contacts
plt.figure(figsize=(15,15)) 
G = nx.Graph()
G = nx.from_pandas_edgelist(edgelist, 'PatientID', 'Contacts0', edge_attr = True)

# Get information about the network nodes
#list(G.edges)
#list(G.nodes)
#list(G.adj['P6'])
#G.degree['P6'] 
degree_sequence = sorted([d for n, d in G.degree()], reverse=True)
# print top 10 patients/contact patients and their degrees
tmp = [(n, d) for n, d in G.degree()]
display("Top 10 patients/contact patients and their degrees", sorted(tmp, key=lambda element: (element[1], element[0]), reverse=True)[:11])

PatientIDs = list(edgelist.PatientID.unique())
contacts = list(edgelist.Contacts0.unique())

#layout = nx.spring_layout(G)
#layout = nx.kamada_kawai_layout(G)
layout = nx.fruchterman_reingold_layout(G,
                                        k=.5,
                                        pos=None,
                                        fixed=None,
                                        iterations=100,
                                        #threshold=1e-4,
                                        weight='weight',
                                        scale=1,
                                        center=None,
                                        dim=2,
                                        seed=None)
nx.draw_networkx_edges(G, pos=layout, edge_color='#ABABAB')

PatientIDs = [node for node in G.nodes() if node in edgelist.PatientID.unique()]
size = [G.degree(node) * 70 for node in G.nodes() if node in edgelist.PatientID.unique()]
nx.draw_networkx_nodes(G, pos=layout, nodelist=PatientIDs, node_size=size, node_color='lightblue')

contacts = [node for node in G.nodes() if node in edgelist.Contacts0.unique()]
nx.draw_networkx_nodes(G, pos=layout, nodelist=contacts, node_size=10, node_color='#ABABAB')

high_degree_Patient0 = [node for node in G.nodes() if node in edgelist.Contacts0.unique() and G.degree(node) > 1]
nx.draw_networkx_nodes(G, pos=layout, nodelist=high_degree_Patient0, node_size=10, node_color='#ac5d36')


PatientIDs_dict = dict(zip(PatientIDs, PatientIDs))
nx.draw_networkx_labels(G, pos=layout, labels=PatientIDs_dict)
contactsIDs_dict = dict(zip(contacts, contacts))
nx.draw_networkx_labels(G, pos=layout, labels=contactsIDs_dict)

edges = G.edges()
weights = [G[u][v]['weight'] for u,v in edges]

nx.draw(G, pos=layout, edges=edges, width=weights)

edgeLabels = {}  
for a, b in G.edges():
    edgeLabels[(a, b)] = str(G.get_edge_data(a, b, {"travelledFrom":0})["travelledFrom"])

nx.draw_networkx_edge_labels(G, pos=layout, edge_labels=edgeLabels, font_color='black', alpha=.7) 

nx.draw_networkx_nodes(G, nodelist=PatientIDs, pos=layout, node_color='lightblue', label='P-Patient')
nx.draw_networkx_nodes(G, nodelist=contacts, pos=layout, node_color='#ABABAB', label='CP-Contact with a Patient')
nx.draw_networkx_nodes(G, nodelist=high_degree_Patient0, pos=layout, node_color='#ac5d36', label='CP-Contact With More Than 1 Patient')
plt.legend(numpoints = 1)


plt.axis('off')
plt.title("Corona Virus India Network based on individualdetails.csv file as on 26 March 2020")
plt.show()

############################################################################
############### Plot Covid19 India initial day cases on map with time.sleep() ################
############################################################################

input_data_animation = input_data[['Diagnosed date','Detected district']]
input_data_animation['DetectedCountry']='India'
input_data_animation_grpby = input_data_animation.groupby(['Diagnosed date', 'Detected district', 'DetectedCountry'])['Diagnosed date'].agg(['count']).reset_index().rename(columns={"count":"CountOfPerson"})
display("Input data shape", input_data_animation_grpby.shape)


def createLat(city):
    geolocator = Nominatim()
    country = 'India'
    loc = geolocator.geocode(city+','+ country, timeout=10)
    lat = loc.latitude
    #lon = loc.longitude
    return lat
input_data_animation_grpby['Lat'] =  input_data_animation_grpby['Detected district'].head(20).apply(createLat)

def createLon(city):
    geolocator = Nominatim()
    country = 'India'
    loc = geolocator.geocode(city+','+ country, timeout=10)
    #lat = loc.latitude
    lon = loc.longitude
    return lon
input_data_animation_grpby['Lon'] =  input_data_animation_grpby['Detected district'].head(20).apply(createLon)

input_data_animation_grpby = input_data_animation_grpby.head(20)

sep = ['-'] * len(input_data_animation_grpby)
list_of_day = pd.DatetimeIndex(input_data_animation_grpby['Diagnosed date']).day.tolist()
list_of_day = [str(i) for i in list_of_day]
list_of_month = pd.DatetimeIndex(input_data_animation_grpby['Diagnosed date']).month.tolist()
list_of_month = [str(i) for i in list_of_month]
list_of_year = pd.DatetimeIndex(input_data_animation_grpby['Diagnosed date']).year.tolist()
list_of_year = [str(i) for i in list_of_year]
day_month_year = [i + j + k + l + m for i, j, k, l, m in zip(list_of_day, sep, list_of_month, sep, list_of_year )] 


for i in range(0, len(day_month_year)): 
    text = day_month_year[i]
    m = folium.Map(location=[input_data_animation_grpby['Lat'][i], input_data_animation_grpby['Lon'][i]],  zoom_start=4)
    folium.Circle([input_data_animation_grpby['Lat'][i], input_data_animation_grpby['Lon'][i]], 150000, fill=True).add_child(folium.Popup(text)).add_to(m)
    folium.map.Marker([input_data_animation_grpby['Lat'][i] + 0.5, input_data_animation_grpby['Lon'][i] - 1.6], icon=DivIcon(icon_size=(150,36), icon_anchor=(0,0), html='<div style="font-size: 24pt">%s</div>' % text,)).add_to(m)
    loc = [(input_data_animation_grpby['Lat'][i]-.3, input_data_animation_grpby['Lon'][i]-.3), (input_data_animation_grpby['Lat'][i]-.3, input_data_animation_grpby['Lon'][i]+.3), (input_data_animation_grpby['Lat'][i]+.3, input_data_animation_grpby['Lon'][i]), (input_data_animation_grpby['Lat'][i]-.3, input_data_animation_grpby['Lon'][i]-.3)]
    folium.PolyLine(loc, color='red',  dash_array='1').add_to(m)
    clear_output(wait = True)
    time.sleep(1)
    display(m)
    #m.save('fig.html')

# Plot Covid19 India initial day cases on map
# All points plotted without animation

m = folium.Map(location=[input_data_animation_grpby['Lat'][1], input_data_animation_grpby['Lon'][1]],  zoom_start=4)
for i in range(0, len(day_month_year)): 
    text = day_month_year[i]
    folium.CircleMarker(location=[input_data_animation_grpby['Lat'][i], input_data_animation_grpby['Lon'][i]],
                        radius=5,
                        color='red',
                        popup =('District: ' + input_data_animation_grpby['Detected district'][i] + day_month_year[i] + '<br>'),
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(m)

m
# India initial days Covid cases animation

fig = px.scatter_mapbox(input_data_animation_grpby, lat="Lat", lon="Lon",
                     size='CountOfPerson',hover_data=['Detected district'],
                     color_continuous_scale='burgyl', animation_frame=input_data_animation_grpby["Diagnosed date"].astype(str), 
                     title='Initial days Corona spread in India')
fig.update(layout_coloraxis_showscale=True)
fig.update_layout(mapbox_style="carto-positron", mapbox_zoom=4, mapbox_center = {"lat":20.5937,"lon":78.9629})
fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
fig.show()
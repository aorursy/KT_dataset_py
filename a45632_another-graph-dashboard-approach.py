%matplotlib inline
import pandas as pd
import numpy as np
import time
import datetime
import itertools
import networkx as nx 
import matplotlib.pyplot as plt
from IPython.display import Image
import warnings
warnings.filterwarnings("ignore") 
import os
data = pd.read_csv("../input/crisis-data.csv")
today = time.strftime("%Y-%m-%d")
today = '2018-12-16'
data = data.loc[(data['Reported Date'] == today)]
#print(today)
call_types = data['Call Type'].unique()
officers = data['Officer ID'].unique()
g = nx.DiGraph()
color_map = []
size = []
labels = []
for i in range (0,len(call_types)):
    g.add_node(call_types[i], type = 'call_type')
    color_map.append('lightskyblue')
    size.append(2000)
for i in range (0,len(officers)):
    g.add_node(officers[i], type = 'officer')
    color_map.append('blue')
    size.append(1000)
    
for i in range (0,len(data)):
    if data.iloc[i,5] == 'SUICIDE - IP/JO SUICIDAL PERSON AND ATTEMPTS':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'red', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'ASSIST OTHER AGENCY - ROUTINE SERVICE':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'green', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'DISTURBANCE, MISCELLANEOUS/OTHER':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'green', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'SFD - ASSIST ON FIRE OR MEDIC RESPONSE':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'orange', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'ASLT - WITH OR W/O WEAPONS (NO SHOOTINGS)':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'orange', weight=1)
            labels.append(data.iloc[i,5])       
    if data.iloc[i,5] == 'MISSING - CHILD':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'green', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'BOMB THREATS - IP/JO':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'red', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'PERSON IN BEHAVIORAL/EMOTIONAL CRISIS':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'orange', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'TRESPASS':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'black', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'DIST - DV - NO ASLT':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'green', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'FIGHT - IP - PHYSICAL (NO WEAPONS)':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'green', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'THREATS - DV - NO ASSAULT':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'orange', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'OVERDOSE - DRUG RELATED CASUALTY':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'orange', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'NUISANCE - MISCHIEF':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'green', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'UNKNOWN - COMPLAINT OF UNKNOWN NATURE':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'green', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'ROBBERY - IP/JO (INCLUDES STRONG ARM)':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'red', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'BURG - IP/JO - RES (INCL UNOCC STRUCTURES)':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'green', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'ALARM - COMM, HOLD-UP/PANIC (EXCEPT BANKS)':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'red', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'SEX IN PUBLIC PLACE/VIEW (INCL MASTURBATION)':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'pink', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'CHILD - ABAND, ABUSED, MOLESTED, NEGLECTED':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'orange', weight=1)
            labels.append(data.iloc[i,5])
    if data.iloc[i,5] == 'DOWN - CHECK FOR PERSON DOWN':
            g.add_edge(data.iloc[i,4], data.iloc[i,13], type = data.iloc[i,5], color = 'red', weight=1)
            labels.append(data.iloc[i,5])
# Drawing Graph
print(today)
plt.figure(3,figsize=(10,10)) 
edges = g.edges()
colors = [g[u][v]['color'] for u,v in edges]
nx.draw_circular(g, node_size = size, node_color = color_map,edge_color = colors, width=3.0, size=0.2, with_labels = True);
plt.show();
plt.savefig('graph.png') 
Image(filename='graph.png')
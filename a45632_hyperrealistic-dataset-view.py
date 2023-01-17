%matplotlib inline
import json
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
# Read dataset
train = pd.read_json('../input/train.json')
test = pd.read_json('../input/test.json') 
# Create feature number of ingredients
train['num_ingredients'] = train['ingredients'].apply(lambda x: len(x))
# create list of cooking styles
styles = train.cuisine.unique()
# create list of unique ingredients
def removeDuplicates(aList):
    newList = []
    for i in aList:
        if i not in newList:
            newList.append(i)
    return newList
# Extract all ingredients
ingredients = []
for i in range(0, len(train)):
    line = train.iloc[i,2]
    for j in range(0, len(line)):
        ingredients.append(line[j])      
# Extract unique ingredients list
clean = removeDuplicates(ingredients)
# Create a Graph 
g = nx.Graph()

for i in range(0,len(styles)):
    g.add_node(styles[i],type = 'style')

for i in range(0,len(clean)):
    g.add_node(clean[i],type = 'ingredient')
    
for i in range(0,len(train)):
    line = train.iloc[i,2]
    for j in range(0, len(line)):
        if train.iloc[i,0] == 'greek':
                g.add_edge(train.iloc[i,0],line[j],color='blue', weight=1)
        if train.iloc[i,0] == 'southern_us':
                g.add_edge(train.iloc[i,0],line[j],color='lime', weight=1)
        if train.iloc[i,0] == 'filipino':
                g.add_edge(train.iloc[i,0],line[j],color='cyan', weight=1)
        if train.iloc[i,0] == 'indian':
                g.add_edge(train.iloc[i,0],line[j],color='pink', weight=1)
        if train.iloc[i,0] == 'jamaican':
                g.add_edge(train.iloc[i,0],line[j],color='yellow', weight=1)
        if train.iloc[i,0] == 'spanish':
                g.add_edge(train.iloc[i,0],line[j],color='red', weight=1)
        if train.iloc[i,0] == 'italian':
                g.add_edge(train.iloc[i,0],line[j],color='green', weight=1)                
        if train.iloc[i,0] == 'mexican':
                g.add_edge(train.iloc[i,0],line[j],color='coral', weight=1)
        if train.iloc[i,0] == 'chinese':
                g.add_edge(train.iloc[i,0],line[j],color='orange', weight=1)
        if train.iloc[i,0] == 'british':
                g.add_edge(train.iloc[i,0],line[j],color='grey', weight=1)
        if train.iloc[i,0] == 'thai':
                g.add_edge(train.iloc[i,0],line[j],color='orchid', weight=1)
        if train.iloc[i,0] == 'vietnamese':
                g.add_edge(train.iloc[i,0],line[j],color='olive', weight=1)
        if train.iloc[i,0] == 'cajun_creole':
                g.add_edge(train.iloc[i,0],line[j],color='gold', weight=1)                
        if train.iloc[i,0] == 'brazilian':
                g.add_edge(train.iloc[i,0],line[j],color='blue', weight=1)        
        if train.iloc[i,0] == 'french':
                g.add_edge(train.iloc[i,0],line[j],color='black', weight=1)        
        if train.iloc[i,0] == 'japanese':
                g.add_edge(train.iloc[i,0],line[j],color='peru', weight=1)        
        if train.iloc[i,0] == 'irish':
                g.add_edge(train.iloc[i,0],line[j],color='maroon', weight=1)
        if train.iloc[i,0] == 'korean':
                g.add_edge(train.iloc[i,0],line[j],color='salmon', weight=1)
        if train.iloc[i,0] == 'moroccan':
                g.add_edge(train.iloc[i,0],line[j],color='violet', weight=1)
        if train.iloc[i,0] == 'russian':
                g.add_edge(train.iloc[i,0],line[j],color='purple', weight=1)

# Plot the graph
plt.figure(3,figsize=(90,90))  
edges = g.edges()
colors = [g[u][v]['color'] for u,v in edges]
nx.draw(g,node_color = 'lime', edge_color = colors, with_labels = True)
plt.show()
plt.savefig('graph_cooking.png')
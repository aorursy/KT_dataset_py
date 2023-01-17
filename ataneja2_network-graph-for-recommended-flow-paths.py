# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np
import networkx as nx
from plotly.graph_objs import *
import pandas as pd
import random
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
#import pip
#!pip uninstall -y python-igraph
#for package in sorted(pip.get_installed_distributions(), key=lambda x: x.project_name):
#print("{} ({})".format(package.project_name, package.version))
data = pd.read_csv("../input/graph.csv")
cleaned_data = data.drop(['Cluster','region'],axis=1)
columns = cleaned_data.columns
graph_dataframe = pd.DataFrame(columns=['Node1','Node2','rowNo'])
graph_row_count=1
list1=[]
for index,row in cleaned_data.iterrows(): 
    for col in columns:
        if cleaned_data.iloc[index,cleaned_data.columns.get_loc(col)]==1:
            list1.append(col)
            graph_row_count = graph_row_count + 1
            if graph_row_count == 3:
                list1.append(index)
                df = pd.DataFrame(data=[list1],  columns =  ['Node1','Node2','rowNo'])
                #df.head()
                graph_dataframe= graph_dataframe.append(df,ignore_index=True)
                #print(list1)
                list1.clear()
                list1.append(col)
                graph_row_count = 2

n = data.shape[0]
edgelist = graph_dataframe
edgelist.head()
edgeWeigtedList = edgelist.groupby(["Node1","Node2"]).size().reset_index(name='weight')
edgeWeigtedList.head(10)
nodelist = pd.DataFrame(columns =  ['Node'])
for col in columns:
    df_node = pd.DataFrame(data=[col], columns =  ['Node'])
    nodelist= nodelist.append(df_node,ignore_index=True)
nodelist.head()
nodeWeightedList = edgeWeigtedList.groupby(['Node1'])[['weight']].sum().reset_index()
nodeWeightedList.head()
matched=False
for i,node in nodelist.iterrows(): 
    matched=False
    for j,nodeWeight in nodeWeightedList.iterrows():
     if node['Node']==nodeWeight['Node1']:
             matched=True
            
    if matched==False:
           #print(node['Node'])
           df = pd.DataFrame([[node['Node'], 0]], columns = ['Node1','weight'])
           nodeWeightedList=nodeWeightedList.append(df)
           #df
    
    
#nodeWeightedList        
def get_cmap(n, name='hsv'):
 #   '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
  #  RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)    

cmap = get_cmap(n)
def colors(n):
  ret = []
  r = int(random.random() * 256)
  g = int(random.random() * 256)
  b = int(random.random() * 256)
  step = 256 / n
  for i in range(n):
    r += step
    g += step
    b += step
    r = int(r) % 256
    g = int(g) % 256
    b = int(b) % 256
    ret.append((r,g,b)) 
  return ret
G = nx.DiGraph()
for i,elrow in edgeWeigtedList.iterrows():
      G.add_edge(elrow[0], elrow[1],weight=elrow[2])

#print(edgeWeigtedList.iterrows()
for i, nlrow in nodelist.iterrows():
          #print(nlrow['Node'])
          G.add_node(nlrow['Node'],pos=((10 * (i+1)),  random.random()))
   
wts=[]
edge_trace = Scatter(
    x=[],
    y=[],
    line=dict(width=[],color='#888'),
              #olorscale='YIGnBu'),
    hoverinfo='none',
    mode='lines'
)
for node, adjdata in G.adjacency():
    for adj, diction in adjdata.items():
        wt = diction['weight']
        #print(node,adj,wt)
        wts.append(wt)
for edge,wt in zip(G.edges(),wts):
    x0, y0 = G.node[edge[0]]['pos']
    x1, y1 = G.node[edge[1]]['pos']
    edge_trace['x'] += [x0, x1, None]
    edge_trace['y'] += [y0, y1, None]
    #edge_trace['line']['color'].append(str(wt))
    #edge_trace['line']['width'].append(wt)
    #print(edge_trace['line']['width'])


#edge_trace
#fig = dict(data=edge_trace, layout=layout)
labels=[]
node_trace = Scatter(
    x=[],
    y=[],
    text=labels,
    mode='markers',
    hoverinfo='text',
    marker=Marker(
        showscale=True,
        # colorscale options
        # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
        # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
        colorscale='YIGnBu',
        reversescale=True,
        color=[],
        size=30,
        colorbar=dict(
            thickness=15,
            title='Node Connections',
            xanchor='left',
            titleside='right'
        )
        ,line=dict(width=2)
    
    ))

#node_trace
for node in G.nodes():
    x, y = G.node[node]['pos']
    node_trace['x'].append(x)
    node_trace['y'].append(y)
    labels.append(node)
for i,node in nodeWeightedList.iterrows():
     node_trace['marker']['color'].append(node['weight'])
     node_info = ' # of connections: '+str(node['weight'])
     labels[i]=labels[i]+node_info
     #print(node['weight'])
     #print(labels[i])
     #node_trace['text'].append(node_info)
#for node, adjdata in G.adjacency():
#    for adj, diction in adjdata.items():
#        wt = diction['weight']
#        wts.append(wt)
#        edge_trace['line']['width'].append(wt)
#        node_trace['marker']['color'].append(len(adjacencies))
#        node_info = '# of connections: '+str(len(adjacencies))
#        node_trace['text'].append(node_info)
#print(wt)
#print(node,adjdata)
#print(adjacency.index)
#print(node,adjdata)
#print(node)  
#print(edge_trace)
#print(adjacencies.count)
#print(adjacencies.count)
#print(edgeWeigtedList.count())
#adjlist = nx.generate_adjlist(G)
#for node, adjacencies in enumerate(adjlist):
#    node_trace['marker']['color'].append(len(adjacencies))
#    node_info = '# of connections: '+str(len(adjacencies))
#    node_trace['text'].append(node_info)
    #print(adjacencies)
#for node in G.nodes():
    #print(node)
    #x, y = G.node[node]['pos']
    #print(x)
fig = Figure(data=Data([node_trace,edge_trace]),
             layout=Layout(
                title='<br>Network graph for workflow nodes',
                titlefont=dict(size=16),
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[ dict(
                    text="Workflow Node Graph",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002 ) ],
                xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))


iplot(fig, filename='networkx')
rG = nx.DiGraph()
for i,elrow in edgeWeigtedList.iterrows():
      rG.add_edge(elrow[0], elrow[1],weight=(1000/elrow[2]))
path= nx.all_shortest_paths(rG,'label1','label10','weight')
for p in path:
    print(p)
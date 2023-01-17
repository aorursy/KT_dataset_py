import plotly.offline as pyo

import plotly.plotly as py

from plotly.graph_objs import *

import pandas as pd

import plotly

plotly.offline.init_notebook_mode()

from scipy import signal

pyo.offline.init_notebook_mode()

import plotly.plotly as py

from plotly.graph_objs import *

import plotly.plotly as py

from plotly.graph_objs import *
nb=pd.read_csv("../input/tianyi-datasets/neighborhood graph.csv")

nb.head()
boros=list(nb['boro'].unique())

boros
trace=[]

for boro in boros:

    trace.append({'type':'scatter',

                  'mode':'markers',

                  'y':nb.loc[nb['boro']==boro,'latitude'],

                  'x':nb.loc[nb['boro']==boro,'longitude'],

                  'name':boro,

                  'marker':{'size':10,'opacity':0.7,

                            'line':{'width':1.25,'color':'black'}}})

layout={'title':'NYC in an interesting view',

       'xaxis':{'title':'latitude'},

       'yaxis':{'title':'longitutde'}}

fig=Figure(data=trace,layout=layout)

pyo.iplot(fig)
data=pd.read_csv("../input/tianyi-datasets/nodes ignore direction.csv")

data.head()
def get_edge(nb,data,threshold):

    edge=[]

    for i in range(len(data)):

        if data['count'][i]>=threshold:

            edge.append((data['neighborhood1'][i],data['neighborhood2'][i]))      

    return edge
def get_numbers_of_adjcs(edge,nb):

    n=len(nb)

    num_of_adjacencies=[]

    for i in range(n):

        num_of_adjacencies.append(0)

    for d in edge:

        num_of_adjacencies[d[0]-1]+=1

        num_of_adjacencies[d[1]-1]+=1

    return num_of_adjacencies
def prep(edge,num_of_adjacencies,text,nb):

    edge_trace = Scatter(

    x=[],

    y=[],

    line=Line(width=0.5,color='#888'),

    hoverinfo='none',

    mode='lines')

    

    for i in range(len(edge)):

        e1=edge[i][0]-1

        e2=edge[i][1]-1

        x0, y0 = nb['longitude'][e1],nb['latitude'][e1]

        x1, y1 = nb['longitude'][e2],nb['latitude'][e2]

        edge_trace['x'] += [x0, x1, None]

        edge_trace['y'] += [y0, y1, None]



    node_trace = Scatter(

        x=[],

        y=[],

        text=[],

        mode='markers',

        hoverinfo='text',

        marker=Marker(

            showscale=True,

            colorscale='YIGnBu',

            reversescale=True,

            color=[],

            size=10,

             colorbar=dict(

                thickness=15,

                title='Node Connections',

                xanchor='left',

                titleside='right'

            ),

            line=dict(width=2)))

    

    for i in range(len(nb)):

        x, y = nb['longitude'][i],nb['latitude'][i]

        node_trace['x'].append(x)

        node_trace['y'].append(y)



    for i in range(len(nb)):

        node_info = text[i]

        node_trace['text'].append(node_info)

        node_trace['marker']['color'].append(num_of_adjacencies[i])



        

    fig = Figure(data=Data([edge_trace, node_trace]),

                 layout=Layout(

                    title='<br>NYC texi trip neighborhood interactions',

                    titlefont=dict(size=16),

                    showlegend=False,

                    hovermode='closest',

                    margin=dict(b=20,l=5,r=5,t=40),

                    annotations=[ dict(

                        showarrow=False,

                        xref="paper", yref="paper",

                        x=0.005, y=-0.002 ) ],

                    xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),

                    yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)))

    

    return fig
edge=get_edge(nb,data,500)

num_of_adjacencies=get_numbers_of_adjcs(edge,nb)

#prepare the hover text

text=[]

for i in range(len(nb)):

    t='neighborhood:'+'<b>'+str(nb['neighborhood_name'][i])+'</b>'+'<br>'+'boro:'+ '<b>'+str(nb['boro'][i])+'</b>'+'<br>'+'# of connections:'+"<b>"+str(num_of_adjacencies[i])

    text.append(t)

fig=prep(edge,num_of_adjacencies,text,nb)

pyo.iplot(fig)
edge=get_edge(nb,data,2000)

num_of_adjacencies=get_numbers_of_adjcs(edge,nb)

text=[]

for i in range(len(nb)):

    t='neighborhood:'+'<b>'+str(nb['neighborhood_name'][i])+'</b>'+'<br>'+'boro:'+ '<b>'+str(nb['boro'][i])+'</b>'+'<br>'+'# of connections:'+"<b>"+str(num_of_adjacencies[i])

    text.append(t)

fig=prep(edge,num_of_adjacencies,text,nb)

pyo.iplot(fig)
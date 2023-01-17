# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import csv

#

#

#with open('some.csv', newline='', encoding='utf-8') as f:

with open( "../input/" + 'edges_1.csv', newline='' ) as csv_File:

    #

    if 1 == 0:

        dialect = csv.Sniffer().sniff( csv_File.read(1024) )

        print( "sniff.dialect:", dialect )   

        # back to start

        csv_File.seek(0)

    #

    #reader = csv.reader( csvfile, dialect )

    # csv.reader(csvfile, dialect='excel', **fmtparams)

    csv_Reader = csv.reader( 

        csv_File, 

        #dialect

        delimiter=' ', 

        #quotechar='|' 

        #quoting=csv.QUOTE_NONE

    )

    if 1 == 1:

        print( "csv_Reader.dialect:", csv_Reader.dialect )   

    #    

    # TypeError: 'list' object

    edges_Total = int( next( csv_Reader )[0] )

    #edges_Total = next( csv_Reader )

    if 1 == 1:

        print( "edges_Total: {} {}".format( type( edges_Total ), edges_Total ) )  

        # edges_Total: <class 'list'> ['4']

    if 1 == 0:

        for row in csv_Reader:

            if csv_Reader.line_num < 3:

                print( ', '.join( row ) )

            else:

                break # for

    #  

    # csv.DictReader(f, fieldnames=None, restkey=None, restval=None, dialect='excel', 

    #   *args, **kwds)

    # DictReader objects have the following public attribute:

    #    csvreader.fieldnames

    #    If not passed as a parameter 

    #    when creating the object, 

    #    this attribute is initialized 

    #    upon first access 

    #    or when the first record is read from the file.

    #

    #csv_File.seek(0)

    # from current 'csv_File' position

    field_Names = ['node_V', 'node_U']

    dict_Reader = csv.DictReader( 

        csv_File, 

        fieldnames = field_Names, 

        delimiter=' ', 

    ) 

    #next( dict_Reader )

    #

    # assuming directed edges|arcs: 'node_V' -> 'node_U'| parent -> child

    nodes = set()

    root = set()

    leafs = set()

    edges_Map_List = list()

    edges_Tuple_List = list()

    #

    #for row in reader:

    #    print( row['first_name'], row['last_name'] )

    for r_i in range( 0, edges_Total, 1 ):

        # If a row has more fields than fieldnames, 

        # the remaining data is put in a list 

        # and stored with the fieldname 

        # specified by restkey (which defaults to None). 

        # If a non-blank row has fewer fields than fieldnames, 

        # the missing values are filled-in with None.

        row = next( dict_Reader )

        node_V = int( row['node_V'] ) 

        node_U = int( row['node_U'] ) 

        edges_Map_List.append( { 'node_V': node_V, 'node_U': node_U } )

        edges_Tuple_List.append( ( node_V, node_U ) )

        nodes.add( node_V )

        nodes.add( node_U )

        root.add( node_V )

        # root must be without parent

        root.discard( node_U )

        leafs.add( node_U )

        # leaf must be without children

        leafs.discard( node_V )

        print( "row:", row )

    #    

    tree_Radius = int( next( csv_Reader )[0] )

    if 1 == 1:

        print( "tree_Radius{}:{}".format( type( tree_Radius ), tree_Radius ) )   

    # difference(*others)

    inner_Nodes = nodes - root - leafs

    if 1 == 1:

        print( "nodes({}):{}".format( len( nodes ), nodes ) )   

        print( "root({}):{}".format( len( root ), root ) )   

        print( "inner_Nodes({}):{}".format( len( inner_Nodes ), inner_Nodes ) )   

        print( "leafs({}):{}".format( len( leafs ), leafs ) )   

        print( "edges({}):{}".format( len( edges_Map_List ), edges_Map_List ) )   
# evenly sampled time at 200ms intervals

t = np.arange(0., 5., 0.2)

t
from random import randrange

#

#

[ ( randrange( 1, 10, 1 ), randrange( 1, 10, 1 ) ) for ( v, u ) in edges_Tuple_List ]
nodes_Xs = [ randrange( 1, 10, 1 ) for node in nodes ]

nodes_Ys = [ randrange( 1, 10, 1 ) for node in nodes ]

( nodes_Xs, nodes_Ys )
nodes_Coords_List = [ 

    { 

        'node': node,

        'x': randrange( 1, 10, 1 ), 

        'y': randrange( 1, 10, 1 ) 

    } 

    for node in nodes 

]

nodes_Coords_List
nodes_Coords_Map = { 

    node: { 

        #'node': node,

        'x': randrange( 1, 10, 1 ), 

        'y': randrange( 1, 10, 1 ) 

    } 

    for node in nodes 

}

nodes_Coords_Map
#nodes_Coords_Map.items()

#zipped = [ ( node['x'], node['y'] ) for node in nodes_Coords_Map.values() ]

( edges_Xs, edges_Ys ) = zip( 

    *[ ( node['x'], node['y'] ) for node in nodes_Coords_Map.values()  ] )

( edges_Xs, edges_Ys )

#zipped
import matplotlib.pyplot as plt

# plot() is a versatile command, 

# and will take an arbitrary number of arguments. 

# plt.plot(x, y, linewidth=2.0)

#

if 1 == 0:

    x1 = 1.0

    y1 = 4.0

    x2 = 1 

    y2 = 1

    lines = plt.plot(

        ##x1, y1, x2, y2

        #[x1, x2], [y1, y2]

        #nodes_Xs,

        #nodes_Ys

        edges_Xs,

        edges_Ys

    )

    # use keyword args

    plt.setp( 

        lines, 

        color='r', 

        linestyle = ':',

        linewidth=2.0,

        marker = '.',#'+',

        markeredgecolor = 'b',

        # float value in points

        markeredgewidth = 3.0,

        zorder = 2

    )

#

# matplotlib.pyplot.figtext(*args, **kwargs)

# horizontalalignment or ha	[ ‘center’ | ‘right’ | ‘left’ ]

# label	string or anything printable with ‘%s’ conversion.

# picker	[None|float|boolean|callable]

# position	(x,y)

# text	string or anything printable with ‘%s’ conversion.

# x	float

# y	float

# text(x, y, s, fontdict=None, **kwargs)

#plt.text( x = 6.0, y = 6.0, s = '0' )

for ( 

        k, 

        #{'x': x, 'y': y} 

        coords

    ) in nodes_Coords_Map.items():

    x = coords['x']

    y = coords['y']

    plt.plot( x, y, 'ro')

    plt.text( x = x, y = y, s = str( k ) )

if 1 == 0:

    plt.figtext( 

        6,

        6,

        "zero",

        horizontalalignment = 'right',

        #position = ( 6, 6 ),

        label = '0',  

        #text = "zero"

    )

if 1 == 0:    

    plt.annotate(

        'local max', 

        xy=(2, 1), 

        xytext=(3, 1.5),

        arrowprops=dict(facecolor='black', shrink=0.05),

    )    

ax = plt.axes()

for ( node_V, node_U ) in edges_Tuple_List:  

    node_V_XY = nodes_Coords_Map[ node_V ]

    node_U_XY = nodes_Coords_Map[ node_U ]

    ax.arrow(

        node_V_XY['x'], node_U_XY['x'], 

        node_V_XY['y'], node_U_XY['y'], 

        #head_width=0.2, 

        #head_length=0.5, 

        #fc='k', ec='k'

    )    

    plt.plot(

        [ node_V_XY['x'], node_U_XY['x'] ],

        [ node_V_XY['y'], node_U_XY['y'] ]

    )

#

#plt.plot([2,3])

#plt.plot([3,4])

#plt.plot([1,4])

#plt.plot([1], [1], 'ro', zorder = 1)

#plt.plot([1], [4], 'ro')

#plt.plot([1,4, 1,1])

plt.ylabel('some numbers')

plt.show()
import igraph as ig
# see: "http://graphviz.readthedocs.io/en/stable/manual.html"

# and "http://www.graphviz.org/doc/info/lang.html"

from graphviz import Digraph, Source
src = Source('digraph "the holy hand grenade" { rankdir=LR; 1 -> 2 -> 3 -> lob }')

src
dot = Digraph(comment='The Round Table')

dot.node('A', 'King Arthur')

dot.node('B', 'Sir Bedevere the Wise')

dot.node('L', 'Sir Lancelot the Brave')

dot.edges(['AB', 'AL'])

dot.edge('B', 'L', constraint='false')

print(dot.source)
dot._repr_svg_()
#dot.render('test-output/round-table.gv', view=True) 

dot.format = 'svg'

dot.render()
from IPython.core.display import display, HTML

print( dot.pipe().decode('utf-8') )
help
import plotly.plotly as py

import numpy as np



data = [dict(

        visible = False,

        line=dict(color='00CED1', width=6),

        name = '𝜈 = '+str(step),

        x = np.arange(0,10,0.01),

        y = np.sin(step*np.arange(0,10,0.01))) for step in np.arange(0,5,0.1)]

data[10]['visible'] = True



steps = []

for i in range(len(data)):

    step = dict(

        method = 'restyle',

        args = ['visible', [False] * len(data)],

    )

    step['args'][1][i] = True # Toggle i'th trace to "visible"

    steps.append(step)



sliders = [dict(

    active = 10,

    currentvalue = {"prefix": "Frequency: "},

    pad = {"t": 50},

    steps = steps

)]



layout = dict(sliders=sliders)

fig = dict(data=data, layout=layout)



#py.iplot(fig, filename='Sine Wave Slider')
import plotly.plotly as py

from plotly.graph_objs import *

#

import networkx as nx

# from: "https://plot.ly/python/network-graphs/"

#

G=nx.random_geometric_graph(200,0.125)

pos=nx.get_node_attributes(G,'pos')



dmin=1

ncenter=0

for n in pos:

    x,y=pos[n]

    d=(x-0.5)**2+(y-0.5)**2

    if d<dmin:

        ncenter=n

        dmin=d



p=nx.single_source_shortest_path_length(G,ncenter)
edge_trace = Scatter(

    x=[],

    y=[],

    line=Line(width=0.5,color='#888'),

    hoverinfo='none',

    mode='lines')



for edge in G.edges():

    x0, y0 = G.node[edge[0]]['pos']

    x1, y1 = G.node[edge[1]]['pos']

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

        # colorscale options

        # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |

        # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'

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



for node in G.nodes():

    x, y = G.node[node]['pos']

    node_trace['x'].append(x)

    node_trace['y'].append(y)
for node, adjacencies in enumerate(G.adjacency_list()):

    node_trace['marker']['color'].append(len(adjacencies))

    node_info = '# of connections: '+str(len(adjacencies))

    node_trace['text'].append(node_info)
"""

"""

fig = Figure(

    data=Data([edge_trace, node_trace]),

    layout=Layout(

    title='<br>Network graph made with Python',

    titlefont=dict(size=16),

    showlegend=False,

    hovermode='closest',

    margin=dict(b=20,l=5,r=5,t=40),

    annotations=[ dict(

        text="Python code: ",

        #<a href='https://plot.ly/ipython-notebooks/network-graphs/'> 

        #https://plot.ly/ipython-notebooks/network-graphs/</a>

        #",

        showarrow=False,

        xref="paper", yref="paper",

        x=0.005, y=-0.002 ) 

    ],

        xaxis=XAxis(showgrid=False, zeroline=False, showticklabels=False),

        yaxis=YAxis(showgrid=False, zeroline=False, showticklabels=False)

    )

)

#

#py.iplot(fig, filename='networkx')
import numpy  as np

import pandas as pd

import geopandas, folium



from IPython.display import IFrame, display
df = pd.read_csv("../input/sao-paulo-metro/metrosp_stations.csv")



colors = { 'azul'    : 'darkblue',

           'verde'   : 'green',

           'vermelha': 'red',

           'amarela' : 'beige',

           'lilas'   : 'purple',

           'prata'   : 'lightgray'}

df.loc[:, 'cor'] = df['line'].apply( lambda x: colors[sorted(x.strip("[]").replace("'", '').replace(" ", "").split(","))[0]] )



df.sample(5)
centroid = (df['lat'].mean(), df['lon'].mean())

sp = folium.Map( location=(centroid[0], centroid[1]), zoom_start=12 )

for i, r in df.iterrows():

    folium.Marker( location = [r['lat'], r['lon']],

                   icon     = folium.Icon(color=r['cor']),

                   popup    = r['name'] ).add_to(sp)

# display( sp )
class Node:

    def __init__ (self, state, cost, parent=None, action=None):

        self.state  = state

        self.cost   = cost

        self.parent = parent

        self.action = action

        

        if self.parent:

            self.depth = self.parent.depth + 1

        else:

            self.depth = 0

    

    def __repr__ (self):

        return '<Node {}>'.format(self.state)
class Stack:

    def __init__ (self):

        self.items = []

    

    def push(self, item):

        self.items.append(item)

    

    def pop(self):

        return self.items.pop()

    

    def peek(self):

        return self.items[len(self.items)-1]

    

    def __len__(self):

        return len(self.items)
class Problem(object):

    def __init__ (self, s0='luz', s='luz'):

        self.states = GRAPH

        self.costs  = COSTS

        self.goal   = s

        self.start  = s0

    

    

    def start(self):

        return self.start

    

    def is_state(self, state):

        return state in self.states

    

    def actions(self, state):

        if state in self.states:

            return self.states[state]['neigh']

        else:

            return None

    

    def next_state(self, state, action):

        if action in self.actions(state):

            return action

        else:

            return None

    

    def is_goal_state(self, state):

        return state == self.goal

    

    def cost(self, state, action):

        return self.costs[(state, action)]
GRAPH = { r['station'] : {'neigh': r['neigh'].strip("[]").replace("'", '').replace(" ", "").split(","), 'pos': (r['lat'], r['lon'])} for i,r in df.iterrows() }
def costFunc(s, S):

    sy, sx = s['pos'][0], s['pos'][1]

    Sy, Sx = S['pos'][0], S['pos'][1]

    return ((sx-Sx)**2 + (sy-Sy)**2)**(0.5)



COSTS = {}

for s in GRAPH:

    for S in GRAPH[s]['neigh']:

        COSTS[(s, S)] = costFunc(GRAPH[s], GRAPH[S])
def depthFirstSearch(problem):

    node     = Node(problem.start, 0)

    frontier = Stack()

    frontier.push(node)

    explored = set()

    while len(frontier) > 0:

        node = frontier.pop()

        explored.add( node.state )

        

        if problem.is_goal_state( node.state ):

            return node

        

        for act in problem.actions( node.state ):

            next_state = problem.next_state( node.state, act )

            if next_state not in explored:

                cost = problem.cost( node.state, act ) + node.cost

                frontier.push( Node(next_state, cost, node, act) )

    return None

p = Problem('butanta', 'corinthians-itaquera')

a = depthFirstSearch(p)
def ans(problem, sol):

    sy, sx = GRAPH[problem.start]['pos'][0], GRAPH[problem.start]['pos'][1]

    gy, gx = GRAPH[problem.goal]['pos'][0], GRAPH[problem.goal]['pos'][1]

    

    centroid = (np.mean([sy,gy]), np.mean([sx,gx]))



    map = folium.Map( location=(centroid[0], centroid[1]), zoom_start=12 )

    for i, r in df.iterrows():

        folium.Marker( location = [r['lat'], r['lon']],

                   icon     = folium.Icon(color=r['cor']),

                   popup    = r['name'] ).add_to(map)

    

    parent = sol

    points = [(gy, gx), (GRAPH[parent.state]['pos'][0], GRAPH[parent.state]['pos'][1])]

    folium.CircleMarker(location=points[-1], radius='4').add_to(map)

    while parent != None:

        points.append( (GRAPH[parent.state]['pos'][0], GRAPH[parent.state]['pos'][1]) )

        folium.CircleMarker(location=points[-1], radius='4').add_to(map)

        parent = parent.parent

    

    folium.PolyLine(locations=points, color="black", weight=4).add_to(map)

    

    # display(map)



ans(p, a)
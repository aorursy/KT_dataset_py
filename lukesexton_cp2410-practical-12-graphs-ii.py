# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from IPython.display import Image

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
class Graph:

    """Representation of a simple graph using an adjacency map."""



    # ------------------------- nested Vertex class -------------------------

    class Vertex:

        """Lightweight vertex structure for a graph."""

        __slots__ = '_element'



        def __init__(self, x):

            """Do not call constructor directly. Use Graph's insert_vertex(x)."""

            self._element = x



        def element(self):

            """Return element associated with this vertex."""

            return self._element



        def __hash__(self):  # will allow vertex to be a map/set key

            return hash(id(self))



        def __str__(self):

            return str(self._element)



    # ------------------------- nested Edge class -------------------------

    class Edge:

        """Lightweight edge structure for a graph."""

        __slots__ = '_origin', '_destination', '_element'



        def __init__(self, u, v, x):

            """Do not call constructor directly. Use Graph's insert_edge(u,v,x)."""

            self._origin = u

            self._destination = v

            self._element = x



        def endpoints(self):

            """Return (u,v) tuple for vertices u and v."""

            return (self._origin, self._destination)



        def opposite(self, v):

            """Return the vertex that is opposite v on this edge."""

            if not isinstance(v, Graph.Vertex):

                raise TypeError('v must be a Vertex')

            return self._destination if v is self._origin else self._origin

            raise ValueError('v not incident to edge')



        def element(self):

            """Return element associated with this edge."""

            return self._element



        def __hash__(self):  # will allow edge to be a map/set key

            return hash((self._origin, self._destination))



        def __str__(self):

            return '({0},{1},{2})'.format(self._origin, self._destination, self._element)



    # ------------------------- Graph methods -------------------------

    def __init__(self, directed=False):

        """Create an empty graph (undirected, by default).



        Graph is directed if optional paramter is set to True.

        """

        self._outgoing = {}

        # only create second map for directed graph; use alias for undirected

        self._incoming = {} if directed else self._outgoing



    def _validate_vertex(self, v):

        """Verify that v is a Vertex of this graph."""

        if not isinstance(v, self.Vertex):

            raise TypeError('Vertex expected')

        if v not in self._outgoing:

            raise ValueError('Vertex does not belong to this graph.')



    def is_directed(self):

        """Return True if this is a directed graph; False if undirected.



        Property is based on the original declaration of the graph, not its contents.

        """

        return self._incoming is not self._outgoing  # directed if maps are distinct



    def vertex_count(self):

        """Return the number of vertices in the graph."""

        return len(self._outgoing)



    def vertices(self):

        """Return an iteration of all vertices of the graph."""

        return self._outgoing.keys()



    def edge_count(self):

        """Return the number of edges in the graph."""

        total = sum(len(self._outgoing[v]) for v in self._outgoing)

        # for undirected graphs, make sure not to double-count edges

        return total if self.is_directed() else total // 2



    def edges(self):

        """Return a set of all edges of the graph."""

        result = set()  # avoid double-reporting edges of undirected graph

        for secondary_map in self._outgoing.values():

            result.update(secondary_map.values())  # add edges to resulting set

        return result



    def get_edge(self, u, v):

        """Return the edge from u to v, or None if not adjacent."""

        self._validate_vertex(u)

        self._validate_vertex(v)

        return self._outgoing[u].get(v)  # returns None if v not adjacent



    def degree(self, v, outgoing=True):

        """Return number of (outgoing) edges incident to vertex v in the graph.



        If graph is directed, optional parameter used to count incoming edges.

        """

        self._validate_vertex(v)

        adj = self._outgoing if outgoing else self._incoming

        return len(adj[v])



    def incident_edges(self, v, outgoing=True):

        """Return all (outgoing) edges incident to vertex v in the graph.



        If graph is directed, optional parameter used to request incoming edges.

        """

        self._validate_vertex(v)

        adj = self._outgoing if outgoing else self._incoming

        for edge in adj[v].values():

            yield edge



    def insert_vertex(self, x=None):

        """Insert and return a new Vertex with element x."""

        v = self.Vertex(x)

        self._outgoing[v] = {}

        if self.is_directed():

            self._incoming[v] = {}  # need distinct map for incoming edges

        return v



    def insert_edge(self, u, v, x=None):

        """Insert and return a new Edge from u to v with auxiliary element x.



        Raise a ValueError if u and v are not vertices of the graph.

        Raise a ValueError if u and v are already adjacent.

        """

        if self.get_edge(u, v) is not None:  # includes error checking

            raise ValueError('u and v are already adjacent')

        e = self.Edge(u, v, x)

        self._outgoing[u][v] = e

        self._incoming[v][u] = e
def graph_from_edgelist(E, directed=False):

    """Make a graph instance based on a sequence of edge tuples.



    Edges can be either of from (origin,destination) or

    (origin,destination,element). Vertex set is presume to be those

    incident to at least one edge.



    vertex labels are assumed to be hashable.

    """

    g = Graph(directed)

    V = set()

    for e in E:

        V.add(e[0])

        V.add(e[1])



    verts = {}  # map from vertex label to Vertex instance

    for v in V:

        verts[v] = g.insert_vertex(v)



    for e in E:

        src = e[0]

        dest = e[1]

        element = e[2] if len(e) > 2 else None

        g.insert_edge(verts[src], verts[dest], element)



    return g
E = (

        ('A', 'B'), ('A', 'C'), ('A', 'D'),

        ('B', 'D'), ('C', 'D'), ('C', 'E'),

        ('D', 'E'), ('E', 'A')

    )



graph = graph_from_edgelist(E, True)



vertices = []

edges = []

print('Vertices (directed Graph):', '\n')

for vertice in graph._incoming:

    vertices.append(vertice)

    print(vertice)

print('\n')    



for vertice in graph._incoming:

    print('Vertice --', vertice)

    print('Adjacent Vertices (Origin, Desination, Path Number')

    incidents = graph.incident_edges(vertice)

    for incident in incidents:

        edges.append(incident)

        print(incident)
Image("../input/directed_matrix.JPG")
Image("../input/prac_12_q2.jpg")
E2 = (

        ('A', 'B', 4), ('A', 'C', 7), ('A', 'D', 6), ('A', 'F', 15),

        ('B', 'E', 3), ('C', 'F', 8), ('D', 'E', 2),

        ('E', 'F', 5)

    )



graph2 = graph_from_edgelist(E, False)



vertices = []

edges = []

print('Vertices (directed Graph):', '\n')

for vertice in graph2._outgoing:

    vertices.append(vertice)

    print(vertice)

print('\n')    



for vertice in graph2._outgoing:

    print('Vertice --', vertice)

    print('Adjacent Vertices (Origin, Desination, Path Number)')

    incidents = graph2.incident_edges(vertice)

    for incident in incidents:

        edges.append(incident)

        print(incident)
class PriorityQueueBase:

    """Abstract base class for a priority queue."""



    # ------------------------------ nested _Item class ------------------------------

    class _Item:

        """Lightweight composite to store priority queue items."""

        __slots__ = '_key', '_value'



        def __init__(self, k, v):

            self._key = k

            self._value = v



        def __lt__(self, other):

            return self._key < other._key  # compare items based on their keys



        def __repr__(self):

            return '({0},{1})'.format(self._key, self._value)



    # ------------------------------ public behaviors ------------------------------

    def is_empty(self):  # concrete method assuming abstract len

        """Return True if the priority queue is empty."""

        return len(self) == 0



    def __len__(self):

        """Return the number of items in the priority queue."""

        raise NotImplementedError('must be implemented by subclass')



    def add(self, key, value):

        """Add a key-value pair."""

        raise NotImplementedError('must be implemented by subclass')



    def min(self):

        """Return but do not remove (k,v) tuple with minimum key.

    

        Raise Empty exception if empty.

        """

        raise NotImplementedError('must be implemented by subclass')



    def remove_min(self):

        """Remove and return (k,v) tuple with minimum key.

    

        Raise Empty exception if empty.

        """

        raise NotImplementedError('must be implemented by subclass')
class HeapPriorityQueue(PriorityQueueBase):  # base class defines _Item

    """A min-oriented priority queue implemented with a binary heap."""



    # ------------------------------ nonpublic behaviors ------------------------------

    def _parent(self, j):

        return (j - 1) // 2



    def _left(self, j):

        return 2 * j + 1



    def _right(self, j):

        return 2 * j + 2



    def _has_left(self, j):

        return self._left(j) < len(self._data)  # index beyond end of list?



    def _has_right(self, j):

        return self._right(j) < len(self._data)  # index beyond end of list?



    def _swap(self, i, j):

        """Swap the elements at indices i and j of array."""

        self._data[i], self._data[j] = self._data[j], self._data[i]



    def _upheap(self, j):

        parent = self._parent(j)

        if j > 0 and self._data[j] < self._data[parent]:

            self._swap(j, parent)

            self._upheap(parent)  # recur at position of parent



    def _downheap(self, j):

        if self._has_left(j):

            left = self._left(j)

            small_child = left  # although right may be smaller

            if self._has_right(j):

                right = self._right(j)

                if self._data[right] < self._data[left]:

                    small_child = right

            if self._data[small_child] < self._data[j]:

                self._swap(j, small_child)

                self._downheap(small_child)  # recur at position of small child



    # ------------------------------ public behaviors ------------------------------

    def __init__(self):

        """Create a new empty Priority Queue."""

        self._data = []



    def __len__(self):

        """Return the number of items in the priority queue."""

        return len(self._data)



    def add(self, key, value):

        """Add a key-value pair to the priority queue."""

        self._data.append(self._Item(key, value))

        self._upheap(len(self._data) - 1)  # upheap newly added position



    def min(self):

        """Return but do not remove (k,v) tuple with minimum key.

    

        Raise Empty exception if empty.

        """

        if self.is_empty():

            raise Empty('Priority queue is empty.')

        item = self._data[0]

        return (item._key, item._value)



    def remove_min(self):

        """Remove and return (k,v) tuple with minimum key.

    

        Raise Empty exception if empty.

        """

        if self.is_empty():

            raise Empty('Priority queue is empty.')

        self._swap(0, len(self._data) - 1)  # put minimum item at the end

        item = self._data.pop()  # and remove it from the list;

        self._downheap(0)  # then fix new root

        return (item._key, item._value)
class AdaptableHeapPriorityQueue(HeapPriorityQueue):

    """A locator-based priority queue implemented with a binary heap."""



    # ------------------------------ nested Locator class ------------------------------

    class Locator(HeapPriorityQueue._Item):

        """Token for locating an entry of the priority queue."""

        __slots__ = '_index'  # add index as additional field



        def __init__(self, k, v, j):

            super().__init__(k, v)

            self._index = j



    # ------------------------------ nonpublic behaviors ------------------------------

    # override swap to record new indices

    def _swap(self, i, j):

        super()._swap(i, j)  # perform the swap

        self._data[i]._index = i  # reset locator index (post-swap)

        self._data[j]._index = j  # reset locator index (post-swap)



    def _bubble(self, j):

        if j > 0 and self._data[j] < self._data[self._parent(j)]:

            self._upheap(j)

        else:

            self._downheap(j)



    # ------------------------------ public behaviors ------------------------------

    def add(self, key, value):

        """Add a key-value pair."""

        token = self.Locator(key, value, len(self._data))  # initiaize locator index

        self._data.append(token)

        self._upheap(len(self._data) - 1)

        return token



    def update(self, loc, newkey, newval):

        """Update the key and value for the entry identified by Locator loc."""

        j = loc._index

        if not (0 <= j < len(self) and self._data[j] is loc):

            raise ValueError('Invalid locator')

        loc._key = newkey

        loc._value = newval

        self._bubble(j)



    def remove(self, loc):

        """Remove and return the (k,v) pair identified by Locator loc."""

        j = loc._index

        if not (0 <= j < len(self) and self._data[j] is loc):

            raise ValueError('Invalid locator')

        if j == len(self) - 1:  # item at last position

            self._data.pop()  # just remove it

        else:

            self._swap(j, len(self) - 1)  # swap item to the last position

            self._data.pop()  # remove it from the list

            self._bubble(j)  # fix item displaced by the swap

        return (loc._key, loc._value)
def shortest_path_lengths(g, src):

    """Compute shortest-path distances from src to reachable vertices of g.



    Graph g can be undirected or directed, but must be weighted such that

    e.element() returns a numeric weight for each edge e.



    Return dictionary mapping each reachable vertex to its distance from src.

    """

    d = {}  # d[v] is upper bound from s to v

    cloud = {}  # map reachable v to its d[v] value

    pq = AdaptableHeapPriorityQueue()  # vertex v will have key d[v]

    pqlocator = {}  # map from vertex to its pq locator



    # for each vertex v of the graph, add an entry to the priority queue, with

    # the source having distance 0 and all others having infinite distance

    for v in g.vertices():

        if v is src:

            d[v] = 0

        else:

            d[v] = float('inf')  # syntax for positive infinity

        pqlocator[v] = pq.add(d[v], v)  # save locator for future updates



    while not pq.is_empty():

        key, u = pq.remove_min()

        cloud[u] = key  # its correct d[u] value

        del pqlocator[u]  # u is no longer in pq

        for e in g.incident_edges(u):  # outgoing edges (u,v)

            v = e.opposite(u)

            if v not in cloud:

                # perform relaxation step on edge (u,v)

                wgt = e.element()

                if d[u] + wgt < d[v]:  # better path to v?

                    d[v] = d[u] + wgt  # update the distance

                    pq.update(pqlocator[v], d[v], v)  # update the pq entry



    return cloud  # only includes reachable vertices





def shortest_path_tree(g, s, d):

    """Reconstruct shortest-path tree rooted at vertex s, given distance map d.



    Return tree as a map from each reachable vertex v (other than s) to the

    edge e=(u,v) that is used to reach v from its parent u in the tree.

    """

    tree = {}

    for v in d:

        if v is not s:

            for e in g.incident_edges(v, False):  # consider INCOMING edges

                u = e.opposite(v)

                wgt = e.element()

                if d[v] == d[u] + wgt:

                    tree[v] = e  # edge e is used to reach v

    return tree

E = (

        ('A', 'B', 4), ('A', 'C', 9), ('A', 'D', 6),

        ('B', 'D', 3), ('B', 'F', 7), ('C', 'D', 2), ('C', 'E', 9),

        ('D', 'E', 5), ('D', 'F', 3), ('F', 'E', 1) 

    )



graph = graph_from_edgelist(E, False)



vertices = []

edges = []

print('Vertices (directed Graph):', '\n')

for vertice in graph._outgoing:

    vertices.append(vertice)

    print(vertice)

print('\n')    



for vertice in graph._outgoing:

    print('Vertice --', vertice)

    print('Adjacent Vertices (Origin, Desination, Path Number')

    incidents = graph.incident_edges(vertice)

    for incident in incidents:

        edges.append(incident)

        print(incident)
class Partition:

    """Union-find structure for maintaining disjoint sets."""



    # ------------------------- nested Position class -------------------------

    class Position:

        __slots__ = '_container', '_element', '_size', '_parent'



        def __init__(self, container, e):

            """Create a new position that is the leader of its own group."""

            self._container = container  # reference to Partition instance

            self._element = e

            self._size = 1

            self._parent = self  # convention for a group leader



        def element(self):

            """Return element stored at this position."""

            return self._element



    # ------------------------- nonpublic utility -------------------------

    def _validate(self, p):

        if not isinstance(p, self.Position):

            raise TypeError('p must be proper Position type')

        if p._container is not self:

            raise ValueError('p does not belong to this container')



    # ------------------------- public Partition methods -------------------------

    def make_group(self, e):

        """Makes a new group containing element e, and returns its Position."""

        return self.Position(self, e)



    def find(self, p):

        """Finds the group containging p and return the position of its leader."""

        self._validate(p)

        if p._parent != p:

            p._parent = self.find(p._parent)  # overwrite p._parent after recursion

        return p._parent



    def union(self, p, q):

        """Merges the groups containg elements p and q (if distinct)."""

        a = self.find(p)

        b = self.find(q)

        if a is not b:  # only merge if different groups

            if a._size > b._size:

                b._parent = a

                a._size += b._size

            else:

                a._parent = b

                b._size += a._size

def MST_Kruskal(g):

    """Compute a minimum spanning tree of a graph using Kruskal's algorithm.



    Return a list of edges that comprise the MST.



    The elements of the graph's edges are assumed to be weights.

    """

    tree = []  # list of edges in spanning tree

    pq = HeapPriorityQueue()  # entries are edges in G, with weights as key

    forest = Partition()  # keeps track of forest clusters

    position = {}  # map each node to its Partition entry



    for v in g.vertices():

        position[v] = forest.make_group(v)



    for e in g.edges():

        pq.add(e.element(), e)  # edge's element is assumed to be its weight



    size = g.vertex_count()

    while len(tree) != size - 1 and not pq.is_empty():

        # tree not spanning and unprocessed edges remain

        weight, edge = pq.remove_min()

        u, v = edge.endpoints()

        a = forest.find(position[u])

        b = forest.find(position[v])

        if a != b:

            tree.append(edge)

            forest.union(a, b)



    return tree
mst = MST_Kruskal(graph)

for i in mst:

    print(i)
def MST_PrimJarnik(g):

    """Compute a minimum spanning tree of weighted graph g.



    Return a list of edges that comprise the MST (in arbitrary order).

    """

    d = {}  # d[v] is bound on distance to tree

    tree = []  # list of edges in spanning tree

    pq = AdaptableHeapPriorityQueue()  # d[v] maps to value (v, e=(u,v))

    pqlocator = {}  # map from vertex to its pq locator



    # for each vertex v of the graph, add an entry to the priority queue, with

    # the source having distance 0 and all others having infinite distance

    for v in g.vertices():

        if len(d) == 0:  # this is the first node

            d[v] = 0  # make it the root

        else:

            d[v] = float('inf')  # positive infinity

        pqlocator[v] = pq.add(d[v], (v, None))



    while not pq.is_empty():

        key, value = pq.remove_min()

        u, edge = value  # unpack tuple from pq

        del pqlocator[u]  # u is no longer in pq

        if edge is not None:

            tree.append(edge)  # add edge to tree

        for link in g.incident_edges(u):

            v = link.opposite(u)

            if v in pqlocator:  # thus v not yet in tree

                # see if edge (u,v) better connects v to the growing tree

                wgt = link.element()

                if wgt < d[v]:  # better edge to v?

                    d[v] = wgt  # update the distance

                    pq.update(pqlocator[v], d[v], (v, link))  # update the pq entry



    return tree
mst2 = MST_PrimJarnik(graph)

for i in mst2:

    print(i)
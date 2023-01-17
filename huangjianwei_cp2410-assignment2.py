import csv

import math

import sys

import time

import matplotlib.pyplot as plt

sys.setrecursionlimit(40000)
class Graph_v1:

    class Vertex:

        def __init__(self, data):

            self._data = data



        def element(self):

            return self._data



    class Edge:

        def __init__(self, data, start, end):

            self._data = data

            self._start = start

            self._end = end

        

        def element(self):

            return self._data

        

        def endpoints(self):

            return (self._start, self._end)



    

    def __init__(self, size=0, directed=False):

        """

        Initialise lists to store

        """

        self._vertex_list = []

        self._edge_list = []

        self._adj_matrix = []

        self._size = size



    def insert_vertex(self,val):

        """

        Create Vertex nodes and append it to vertices list

        """

        v = self.Vertex(val)

        self._vertex_list.append(v)

        self._size += 1



    def insert_edge(self,val,start_pointer,end_pointer):

        """

        Create Edges nodes and append it to edges list

        Insert value into adjacency matrix

        """

        e = self.Edge(val,start_pointer,end_pointer)

        self._edge_list.append(e)

        # Getting index of start and end pointer

        x = self._vertex_list.index(start_pointer)

        y = self._vertex_list.index(end_pointer)

        self._adj_matrix[x][y] = val

        self._adj_matrix[y][x] = val

    

    def constr_adj_matrix(self):

        """

        Initialise adjacency matrix with all 0s

        """

        for i in range(self._size):

            self._adj_matrix.append([0 for j in range(self._size)])





    def display(self):

        """

        Display values in each row of adjacency matrix

        """

        for row in self._adj_matrix:

            for val in row:

                print("{:18}".format(val), end=" ")

            print()
def two_points_dist(p1, p2):

    """

    Calculate the shortest distance between the p1 and p2

    Return: euclid distance

    """

    return math.sqrt(((p1[1]-p2[1])*(p1[1]-p2[1]))+((p1[2]-p2[2])*(p1[2]-p2[2])))
graph = Graph_v1()
with open("../input/cities-2000/cities_2000.csv", "r") as cities_file:

        cities_data = csv.reader(cities_file)

        for row in cities_data:

            graph.insert_vertex([int(row[0]), float(row[1]), float(row[2])])
print("Number of Vertices: ",graph._size)
# Create the adj_matrix

graph.constr_adj_matrix()
for i in range(0,graph._size):

    if i%500 == 0:

        print(i)

    for j in range(0,graph._size):

        # ensure that the same point is not used

        if i != j:

            # find distance and add it as an edge into graph

            dist = two_points_dist(graph._vertex_list[i]._data,graph._vertex_list[j]._data)

            graph.insert_edge(dist,graph._vertex_list[i],graph._vertex_list[j])
# graph.display()
def DFS_v1():

    """

    Depth-First Search

    Return: order or list in which it traversed through the graph

    """

    # Mark all vertices as unvisited

    visited = [False] * graph._size

    visit_order = []



    # Traverse through the graph

    return DFS_traversal_v1(0, visited, visit_order)





def DFS_traversal_v1(index, visit_list, visit_order):

    """

    Recursive

    """

    # Exit condition, when left with the last unvisited vertex node

    if sum(visit_list) == (len(graph._vertex_list)-1):

        # Mark last vertex node as visited

        visit_list[index] = True

        visit_order.append(index)

        return visit_order

    else:

        if len(visit_order)%500 == 0:

            print(len(visit_order))

        # Mark vertex as visited

        visit_list[index] = True

        # Append index of vertex to record visit order

        visit_order.append(index)

        cur_vertex = graph._vertex_list[index]

        # Check for adjacent vertex

        for i in range(len(graph._edge_list)):

            (st_pt, en_pt) = graph._edge_list[i].endpoints()

            # If current point is the same as the starting point

            if st_pt == cur_vertex:

                en_pt_index = graph._vertex_list.index(en_pt)

                # if adjacent node is not visited yet

                if visit_list[en_pt_index] == False:

                    return DFS_traversal_v1(en_pt_index,visit_list,visit_order)
visit_order_dfs_v1 = DFS_v1()

# print(visit_order)
print("Calculating distance")

total_dist_dfs_v1 = 0

for i in range(len(visit_order_dfs_v1)-1):

    if i%500 == 0:

        print(i)

    # Taking distances(weight) from adjacency matrix

    total_dist_dfs_v1 += graph._adj_matrix[visit_order_dfs_v1[i]][visit_order_dfs_v1[i+1]]

total_dist_dfs_v1 += graph._adj_matrix[visit_order_dfs_v1[-1]][0]

print("Total Distance: ", total_dist_dfs_v1)
# Plot the visit order of the cities

visit_order_dfs_v1.append(0)



origin_point = visit_order_dfs_v1.pop(0)



prev_point = graph._vertex_list[origin_point]._data



fig, ax = plt.subplots(figsize=(15,10))

for item in visit_order_dfs_v1:



    x = graph._vertex_list[item]._data[1]

    y = graph._vertex_list[item]._data[2]

    ax.plot([prev_point[1], x], [prev_point[2], y], c="blue", linewidth=0.5)



    prev_point = graph._vertex_list[item]._data



plt.show()
print("Number of Vertices: ",graph._size)
def dijkstra(st_pt, en_pt):

    """

    Dijkstra

    Return:

        sum(d_list): float (total distance),

        visited: list (order of visit)

    """

    # Get index of current start point

    st_pt_index = graph._vertex_list.index(st_pt)



    # Create a list of indexes based on the graph size

    queue = [i for i in range(graph._size)]



    # Create distance list

    d_list = [math.inf] * graph._size

    # Create previous list

    prev_list = [st_pt_index] * graph._size

    # Set start point distance to 0 and previous to -1

    d_list[st_pt_index] = 0

    prev_list[st_pt_index] = -1



    

    visited = []

    # Add start point to visited 

    visited.append(st_pt_index)



    # Set current point

    cur_vertex = st_pt_index



    # Run as long as there is an index in queue list

    while len(queue) > 1:

        if len(queue) %500 == 0:

            print("Queue length:",len(queue))

        # Add distance of unvisited cities to d_list

        for i in range(len(graph._adj_matrix[cur_vertex])):

            if i not in visited: 

                d_list[i] = graph._adj_matrix[cur_vertex][i]

        # Use min function to get the smallest distance in the d_list

        smallest_dist = min(i for i in d_list if (d_list.index(i) not in visited))

        smallest_dist_index = d_list.index(smallest_dist)



        # Remove current index from queue list

        queue.remove(cur_vertex)

        prev_list[smallest_dist_index] = cur_vertex

        cur_vertex = smallest_dist_index



        visited.append(smallest_dist_index)



    # Add point of origin distance

    end_point_dist = graph._adj_matrix[st_pt_index][cur_vertex]

    d_list.append(end_point_dist)



    return sum(d_list), visited
# Get distance and visited list order

total_dist_dijstra, visited_list_dijkstra = dijkstra(graph._vertex_list[0],graph._vertex_list[0])

print("Total Distance: ", total_dist_dijstra)
# Plot the visit order of the cities

visited_list_dijkstra.append(0)



origin_point = visited_list_dijkstra.pop(0)



prev_point = graph._vertex_list[origin_point]._data



fig, ax = plt.subplots(figsize=(15,10))

for item in visited_list_dijkstra:



    x = graph._vertex_list[item]._data[1]

    y = graph._vertex_list[item]._data[2]

    ax.plot([prev_point[1], x], [prev_point[2], y], c="blue", linewidth=0.5)



    prev_point = graph._vertex_list[item]._data



plt.show()
class Graph_v2:

    class Vertex:

        def __init__(self, data, visited=False):

            self._data = data

            self._visited = visited

        

        def element(self):

            return self._data



        

    class Edge:

        def __init__(self, data, start, end):

            self._data = data

            self._start = start

            self._end = end

        

        def element(self):

            return self._data

        

        def endpoints(self):

            return (self._start, self._end)



    

    def __init__(self, size=0, directed=False):

        """

        Initialise lists to store

        """

        self._vertex_list = []

        self._edge_list = []

        self._adj_matrix = []

        self._size = size



    def insert_vertex(self,val):

        """

        Create Vertex nodes and append it to vertices list

        """

        v = self.Vertex(val)

        self._vertex_list.append(v)

        self._size += 1

    

    def insert_edge(self,val,start_pointer,end_pointer):

        """

        Create Edges nodes and append it to edges list

        """

        e = self.Edge(val,start_pointer,end_pointer)

        self._edge_list.append(e)

    

    def constr_adj_matrix(self,size):

        """

        Initialise adjacency matrix with all 0s

        """

        for i in range(size):

            self._adj_matrix.append([0 for j in range(size)])

    

    def adjust_matrix(self,x,y,weight):

        """

        Add edges weight - calculated distances to adjacency matrix

        """

        self._adj_matrix[x][y] = weight

        self._adj_matrix[y][x] = weight

   

    def dijkstra(self, st_pt, en_pt):

        """

        Dijkstra

        """

        # Get index of current start point

        st_pt_index = self._vertex_list.index(st_pt)

        

        queue = [i for i in range(self._size)]

        # Create distance list

        d_list = [math.inf] * self._size

        # Create previous list

        prev_list = [st_pt_index] * self._size

        # Set start point distance to 0 and previous to -1

        d_list[st_pt_index] = 0

        prev_list[st_pt_index] = -1



        visited = []

        # Add start point to visited 

        visited.append(st_pt_index)

        

        # Set current point

        cur_vertex = st_pt_index

        

        while len(queue) > 1:

            if len(queue) %500 == 0:

                print("Queue length:",len(queue))

            for i in range(len(self._adj_matrix[cur_vertex])):

                if i not in visited:

                    d_list[i] = self._adj_matrix[cur_vertex][i]

            smallest_dist = min(i for i in d_list if (d_list.index(i) not in visited))

            smallest_dist_index = d_list.index(smallest_dist)

            queue.remove(cur_vertex)

            prev_list[smallest_dist_index] = cur_vertex

            cur_vertex = smallest_dist_index



            visited.append(smallest_dist_index)

        

        end_point_dist = self._adj_matrix[st_pt_index][cur_vertex]

        d_list.append(end_point_dist)

        return sum(d_list)



    def DFS_v2(self):

        """

        Depth-First Search

        Return: order or list in which it traversed through the graph

        """

        # Mark all vertices as unvisited

        visit_list = [False] * self._size

        visit_order = []



        # Traverse through the graph

        return self.DFS_traversal_v2(0, visit_list, visit_order)

         



    def DFS_traversal_v2(self, index, visit_list, visit_order):

        """

        Recursive

        """

        # Exit condition, size of visit_order is same as visit_list

        if len(visit_order) == len(visit_list):

            return visit_order

        else:

            if visit_list[index] == False:

                # Mark vertex as visited

                visit_list[index] = True

                # Append index of vertex to record visit order

                visit_order.append(index)



            connected_points = []



            # Check for adjacent vertex

            for edge in self._edge_list:

                (st_pt, en_pt) = edge.endpoints()



                # If current point is the same as the starting point

                if st_pt == index:

                    connected_points.append(en_pt)

                    # if adjacent node is not visited yet

                    if visit_list[en_pt] == False:

                        return self.DFS_traversal_v2(en_pt,visit_list,visit_order)

                # If current point is the same as the end point

                if en_pt == index:

                    connected_points.append(st_pt)

                    # if adjacent node is not visited yet

                    if visit_list[st_pt] == False:

                        return self.DFS_traversal_v2(st_pt,visit_list,visit_order)



            # no more unvisited -> go back one vertex and check

            return self.DFS_traversal_v2(connected_points[0],visit_list,visit_order)

                        



    def display(self):

        """

        Display values in each row of adjacency matrix

        """

        for row in self._adj_matrix:

            for val in row:

                print("{:18}".format(val), end=" ")

            print()
def two_points_dist_dict(p1, p2):

    """

    Calculate the shortest distance between the p1 and p2

    Return: euclid distance

    """

    return math.sqrt(((p1[0]-p2[0])*(p1[0]-p2[0]))+((p1[1]-p2[1])*(p1[1]-p2[1])))
def isprime(n):

    """

    Check if integer n is a prime

    """

    # make sure n is a positive integer

    n = abs(int(n))

    # 0 and 1 are not primes

    if n < 2:

        return False

    # 2 is the only even prime number

    if n == 2: 

        return True    

    # all other even numbers are not primes

    if not n & 1: 

        return False

    # range starts with 3 and only needs to go up the squareroot of n

    # for all odd numbers

    for x in range(3, int(n**0.5)+1, 2):

        if n % x == 0:

            return False

    return True
# Initialise

cities_dict = {}

adj_matrix = []



mst_graph = Graph_v2()
# Load csv file into linked list, storing a list for each row of record

with open("../input/cities-2000/cities_2000.csv", "r") as cities_file:

    cities_data = csv.reader(cities_file)

    for row in cities_data:

        cities_dict[int(row[0])] = [float(row[1]), float(row[2])]

        mst_graph.insert_vertex([int(row[0]), float(row[1]), float(row[2])])
mst_graph.constr_adj_matrix(len(cities_dict))
for i in range(len(cities_dict)):

    adj_matrix.append([0 for j in range(len(cities_dict))])



for i in range(0,len(cities_dict)):

    if i%500 == 0:

        print("City:",i)

    for j in range(0,len(cities_dict)):

        if i != j:

            dist = two_points_dist_dict(cities_dict[i],cities_dict[j])

            adj_matrix[i][j] = dist

            adj_matrix[i][j] = dist
def MST(adj_matrix, visited, visit_list, step_counter):

    """

    Minimum Spanning Tree, using Prim's algorithm

    Takes the current cluster of visited indexes, search for 

    the next shortest weight in the adjacency matrix.

    Return: 

        visited: list,

        visit_list: list,

        frm_pt: int (start point),

        min_pos: int (end point),

        min_dist: float (weight),

        step_counter: int

    """

    frm_pt = 0

    min_pos = 0

    min_dist = math.inf

    for index in visited:

        # Look at every index of visited

        for pos, dist in enumerate(adj_matrix[index]):

            # Check: Every 10th step (stepNumber % 10 == 0) is 10% more lengthy

            # unless coming from a prime CityId.

            if (step_counter % 10 == 0) and not isprime(pos):

                dist = dist*1.1

            # Only replace if distance is shortest, not equal to 0 and index not yet visited

            if dist < min_dist and dist != 0 and visit_list[pos] == False:

                min_dist = dist

                min_pos = pos

                frm_pt = index



    visited.append(min_pos)

    visit_list[min_pos] = True

    step_counter += 1



    return visited, visit_list, frm_pt, min_pos, min_dist, step_counter
# Initialise

visited = []

visit_list = [False] * len(cities_dict)

step_counter = 0



# start point    

visited.append(0)

visit_list[0] = True

step_counter += 1



print("Calculating MST")

for i in range(len(cities_dict)-1):

    if i%500 == 0:

        print("City:",i)

    # Calculate next shortest weight

    (visited, visit_list, st_pt, en_pt, weight, step_counter) = MST(adj_matrix,visited, visit_list, step_counter)

    # Add it to adjacency matrix in Graph

    mst_graph.adjust_matrix(st_pt,en_pt,weight)

    # Insert edge into Graph

    mst_graph.insert_edge(weight,st_pt,en_pt)
# mst_graph.display()
visit_order_mst_dfs = mst_graph.DFS_v2()

# print(visit_order_mst_dfs)
start_time = time.time()



print("Calculating distance1")

total_dist = 0

for i in range(len(visit_order_mst_dfs)-1):

    if i%500 == 0:

        print(i)

    # Calculating the 2 point distances

    total_dist += two_points_dist_dict(cities_dict[visit_order_mst_dfs[i]], cities_dict[visit_order_mst_dfs[i+1]])

total_dist += two_points_dist_dict(cities_dict[visit_order_mst_dfs[-1]],cities_dict[0])



end_time = time.time()

time_taken = end_time - start_time

print("Time Taken:",time_taken)

print("Total Distance: ", total_dist)



start_time2 = time.time()



print("Calculating distance2")

total_dist2 = 0

for i in range(len(visit_order_mst_dfs)-1):

    if i%500 == 0:

        print(i)

    # Taking distances(weight) from adjacency matrix

    total_dist2 += adj_matrix[visit_order_mst_dfs[i]][visit_order_mst_dfs[i+1]]

total_dist2 += adj_matrix[visit_order_mst_dfs[-1]][0]



end_time2 = time.time()

time_taken2 = end_time2 - start_time2

print("Time Taken:",time_taken2)

print("Total Distance: ", total_dist2)
# Plot the visit order of the cities

visit_order_mst_dfs.append(0)



origin_point = visit_order_mst_dfs.pop(0)



prev_point = cities_dict[origin_point]



fig, ax = plt.subplots(figsize=(15,10))

for item in visit_order_mst_dfs:



    x = cities_dict[item][0]

    y = cities_dict[item][1]

    ax.plot([prev_point[0], x], [prev_point[1], y], c="blue", linewidth=0.5)



    prev_point = cities_dict[item]



plt.show()
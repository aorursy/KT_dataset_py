from collections import deque # importing deque to use to store final order of the cities

import csv  # imported to open and read the .csv file

import math # imported to use the math.sqrt function 

import sympy # imported to use sympy.isprime function to check for prime numbers

import time # imported to print time taken to run 

import heapq #imported heap

from collections import defaultdict 



# Open cities.csv file

with open('../input/cities/cities.csv', 'r') as file:

    reader = csv.reader(file)

    cities = list(reader)

    # Splits the length of cities list to the first 10%

    subindex = len(cities)//1000

    # First data structure small_cities list

    small_cities = cities[1:subindex]

print(len(small_cities))



# Calculating the euclidean distance between the cities

# Algorithm 1

def calculate_distance(x1,y1,x2,y2):  

    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)  

    return distance



current_city = small_cities[0]

distances = []

distances_row = []

x = 0

#algorithm 2

#this calculates the distance from each city to every other city to create a Graph.

while (len(distances) < len(small_cities)):

    current_city = small_cities[x]

    for i in range (len(small_cities)):

        current_distance = calculate_distance(float(current_city[1]),float(current_city[2]),float(small_cities[i][1]),float(small_cities[i][2]))

        distances_row.append(current_distance)

    distances.append(distances_row)

    distances_row = []

    #print(x)

    x += 1

    #print(current_city)

    

#print(distances)
# Python program for Kruskal's algorithm to find 

# Minimum Spanning Tree of a given connected, 

# undirected and weighted graph 

#Class to represent a graph 

class Graph: 



    def __init__(self,vertices): 

        self.V= vertices #No. of vertices 

        self.graph = [] # default dictionary 

            # to store graph 



    # function to add an edge to graph 

    def addEdge(self,u,v,w): 

        self.graph.append([u,v,w]) 



    # A utility function to find set of an element i 

    # (uses path compression technique) 

    def find(self, parent, i): 

        if parent[i] == i: 

            return i 

        return self.find(parent, parent[i]) 



    # A function that does union of two sets of x and y 

    # (uses union by rank) 

    def union(self, parent, rank, x, y): 

        xroot = self.find(parent, x) 

        yroot = self.find(parent, y) 



        # Attach smaller rank tree under root of 

        # high rank tree (Union by Rank) 

        if rank[xroot] < rank[yroot]: 

            parent[xroot] = yroot 

        elif rank[xroot] > rank[yroot]: 

            parent[yroot] = xroot 



        # If ranks are same, then make one as root 

        # and increment its rank by one 

        else : 

            parent[yroot] = xroot 

            rank[xroot] += 1



        # The main function to construct MST using Kruskal's 

        # algorithm 

    def KruskalMST(self): 



        result =[] #This will store the resultant MST 



        i = 0 # An index variable, used for sorted edges 

        e = 0 # An index variable, used for result[] 



            # Step 1: Sort all the edges in non-decreasing 

                # order of their 

                # weight. If we are not allowed to change the 

                # given graph, we can create a copy of graph 

        self.graph = sorted(self.graph,key=lambda item: item[2]) 



        parent = [] ; rank = [] 



        # Create V subsets with single elements 

        for node in range(self.V): 

            parent.append(node) 

            rank.append(0) 

    

        # Number of edges to be taken is equal to V-1 

        while e < self.V -1 : 



            # Step 2: Pick the smallest edge and increment 

                    # the index for next iteration 

            u,v,w = self.graph[i] 

            i = i + 1

            x = self.find(parent, u) 

            y = self.find(parent ,v) 



            # If including this edge does't cause cycle, 

                        # include it in result and increment the index 

                        # of result for next edge 

            if x != y: 

                e = e + 1

                result.append([u,v,w]) 

                self.union(parent, rank, x, y)

            # Else discard the edge 



        # print the contents of result[] to display the built MST 

        print ("Following are the edges in the constructed MST")

        totalDistance = 0

        for u,v,weight in result:

            totalDistance += weight

            #print str(u) + " -- " + str(v) + " == " + str(weight) 

            print ("%d -- %d == %d" % (u,v,weight)) 

        print("Total Distance is ",totalDistance)    

# Driver code 



q = 0

w = 0

e = 0

g = Graph(len(distances)) 

while q < (len(distances)):

    for item in distances[e]:

        #print(q, w, item)

        g.addEdge(q, w , item)

        w += 1

    e += 1   

    w = 0    

    q += 1

    

g.KruskalMST() 



#This code is contributed by Neelam Yadav 



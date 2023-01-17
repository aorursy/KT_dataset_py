import matplotlib.pyplot as plt

import math

import sys

import itertools

INFINITY = math.inf



class Graph_Test(object):

    

    def __init__(self, matrix):

        self.adjMatrix = matrix

        



        

    def remove_edge(self, v1, v2):

        if self.adjMatrix[v1][v2] == 0:

            print("No edge between %d and %d" % (v1, v2))

            return

        self.adjMatrix[v1][v2] = INFINITY

        self.adjMatrix[v2][v1] = INFINITY

        

    def get_edge(self, v1, v2):

        return self.adjMatrix[v1][v2]

    

    def contains_edge(self, v1, v2):

        return True if self.adjMatrix[v1][v2] > 0 else False

    

    def __len__(self):

        return self.size

        

    def toString(self):

        for row in self.adjMatrix:

            for val in row:

                print('{:9.2f}  '.format(val), end=""),

            print()

            

    def get_minimum(self):

        min = INFINITY

        count_row = 0

        count_col = 0

        min_row = 0

        min_col = 0

        for row in self.adjMatrix:

            for val in row:

                if val < min and val != 0:

                    min = val

                    min_row = count_row

                    min_col = count_col

                count_col += 1

            count_row += 1

            count_col = 0  # reset after finish each row

        return min_row, min_col, min

    

    def mst_Kruskal(self):

        #     Kruskal'S Algorithm

        #     Algorithm Generic-MST(G, w):

        #         A <- 0

        #         while A does not form a spanning tree

        #             do find an edge (u, v) that is safe for A

        #                 A   <-A U {(u,v)}

        #         return A

        

        spt_vertices = set()  # track of vertices in spanning tree

        set_of_vertices = set()  # pairs of vertices or edges

        

        #for i in range(0, len(self.adjMatrix)-1):

        counter = len(self.adjMatrix)-1

        while(counter > 0):

            min_row, min_col, min = self.get_minimum()

            edge = (min_row, min_col)

            print("Proposed edge ", edge)

            if not(self.is_cycle(min_row, min_col, set_of_vertices)): # do not have circularities   

                set_of_vertices.add(edge)

                print("Add edge", edge)

                counter -= 1

            self.remove_edge(min_row, min_col)

        print(set_of_vertices)

                

    def is_cycle(self,walk, search, vertices):

        count = 0

        for vertex in vertices:

            copy_vertices = set((copy_vertex for copy_vertex in vertices if copy_vertex!=vertex ))

            if walk in vertex:

                if search in vertex:

                    print("Cycle found ", walk, search)                    

                    return True

               

                walk = vertex[0]                

                if self.is_cycle(walk, search, copy_vertices) == True:

                    return True

               

                walk = vertex[1]

                if self.is_cycle(walk, search, copy_vertices) == True:

                    return True

                



matrix = [[0,4,9,6,INFINITY,INFINITY],

         [4,0,2,3,INFINITY,7],

         [9,2,0,2,9,INFINITY],

         [6,3,2,0,5,3],

         [INFINITY,INFINITY,9,5,0,1],

         [INFINITY,7,INFINITY,3,1,0]]

g = Graph_Test(matrix)     

g.mst_Kruskal()

g.toString()



# {(0, 1), (1, 2), (4, 5), (2, 3), (3, 5)}           

def is_cycle(walk, search, vertices):

    print(walk)

    print("orginal ", vertices)

    

    count = 0

    print("******************************************************")

    for vertex in vertices:

        copy_vertices = set((copy_vertex for copy_vertex in vertices if copy_vertex!=vertex ))



        print("\tcopy vertices ", copy_vertices)

        print("\tvertex",vertex)

        

        if walk in vertex:

            print("\t\t walk in vertex", vertex)

            if search in vertex:

                print("found")

                return True



            walk = vertex[0]      

            print("Left Walk")

            if is_cycle(walk, search, copy_vertices) == True:

                return True



            walk = vertex[1]

            print("Right Walk")

            if is_cycle(walk, search, copy_vertices) == True:

                return True

        else:

            print("\tno walk")

        count += 1

        

            



vertices = set()

vertices.add((1,2))

vertices.add((2,3))

vertices.add((5,1))

# vertices.add((2,3))

# vertices.add((3,5))







print(is_cycle(1, 7, vertices))
# Python Program to detect cycle in an undirected graph 



from collections import defaultdict 



#This class represents a undirected graph using adjacency list representation 

class Graph_: 



    def __init__(self, vertices = 0): 

        self.V= vertices #No. of vertices 

        self.graph = defaultdict(list) # default dictionary to store graph 





    # function to add an edge to graph 

    def addEdge(self,v,w): 

        self.graph[v].append(w) #Add w to v_s list 

        self.graph[w].append(v) #Add v to w_s list 

        self.V = len(self.graph)



    # A recursive function that uses visited[] and parent to detect 

    # cycle in subgraph reachable from vertex v. 

    def isCyclicUtil(self,v,visited,parent): 



        #Mark the current node as visited 

        visited[v]= True



        #Recur for all the vertices adjacent to this vertex 

        for i in self.graph[v]: 

            # If the node is not visited then recurse on it 

            if visited[i]==False : 

                if(self.isCyclicUtil(i,visited,v)): 

                    return True

            # If an adjacent vertex is visited and not parent of current vertex, 

            # then there is a cycle 

            elif parent!=i: 

                return True



        return False





    #Returns true if the graph contains a cycle, else false. 

    def isCyclic(self): 

        # Mark all the vertices as not visited 

        visited =[False]*(self.V) 

        print("**************************", len(self.graph))

        # Call the recursive helper function to detect cycle in different 

        #DFS trees 

        for i in range(self.V): 

            if visited[i] ==False: #Don't recur for u if it is already visited 

                if(self.isCyclicUtil(i,visited,-1))== True: 

                    return True



        return False



# Create a graph given in the above diagram 

g = Graph2() 

g.addEdge(1, 0) 

g.addEdge(0, 2) 

g.addEdge(2, 0) 

g.addEdge(0, 3) 

g.addEdge(3, 4) 





if g.isCyclic(): 

    print ("Graph contains cycle")

else : 

    print ("Graph does not contain cycle ")

g1 = Graph2() 

g1.addEdge(0,1) 

g1.addEdge(1,2) 





if g1.isCyclic(): 

    print("Graph contains cycle")

else : 

    print("Graph does not contain cycle ")



#This code is contributed by Neelam Yadav 

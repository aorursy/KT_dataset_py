from collections import defaultdict

from heapq import *



# edges is a list of edge and wight, f is start route node, f is end route node

def dijkstra(edges, input_start_node, input_end_node):

    

    # Define a dictionary of g to store graph structure

    ###START CODE HERE###

    g = defaultdict(list) 

    # l, r and c represent start node,end node and wight.

    for start_node, end_node, wight in edges: 

        g[start_node].append((wight, end_node)) 

    ###END CODE HERE### 

    

    # q represent the strat node information, which zero,start and () represently means cost,node and path.

    q, visited = [(0,input_start_node,())], set()

    

    

    #Find the shortest path between that node and every other

    #v1, v2 represent edge

    ### START CODE HERET ###

    while q:

        

        #set it as the new "current node"

        (cost,v1,path) = heappop(q)

        if v1 not in visited:

            visited.add(v1)

            path = (v1, path)

            

            # Step5: If the destination node has been marked visited, then stop. The algorithm has finish.

            if v1 == input_end_node : return (cost, path)

            

            # Step6: Otherwise, select the unvisited node that is marked with the smallest tentative distance,

            #mark the current node as visited and push it into the visited set.

            for c, v2 in g.get(v1, ()):

                if v2 not in visited:

                    heappush(q, (cost+c, v2, path))

        print (q)   

    return float("inf")

    ### END CODE HERE ###





if __name__ == "__main__":

    

    edges = [

        ("A", "B", 7),

        ("A", "D", 5),

        ("B", "C", 8),

        ("B", "D", 9),

        ("B", "E", 7),

        ("C", "E", 5),

        ("D", "E", 7),

        ("D", "F", 6),

        ("E", "F", 8),

        ("E", "G", 9),

        ("F", "G", 11)

    ]

    

    print ("=== Dijkstra ===")

   # print (edges)

    print ("D-> G:")

    print (dijkstra(edges, "D", "G"))
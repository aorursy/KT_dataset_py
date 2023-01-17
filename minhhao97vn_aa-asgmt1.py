# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import time

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import random as rand

import math
def mergeSort(A,inversion):

    n = len(A)

    if (n==1):

        return A,inversion

    else:

        q = math.ceil(n/2)

        L = A[0:q]

        R = A[q:n]

        LS,inversion = mergeSort(L,inversion)

        RS,inversion = mergeSort(R,inversion)

        return merge(LS,RS,inversion)

    

def merge(L,R,inversion):

    i = 0

    j = 0

    n1 = len(L)

    n2 = len(R)

    B = np.empty([n1+n2]).tolist()

    count = 0

    for k in range(n1+n2):

        if j>n2-1 or (i<n1 and L[i][2] <= R[j][2]):

            B[k] = L[i]

            i += 1

            inversion += count

        else:

            B[k] = R[j]

            count = count+1

            j += 1

    return B,inversion

def mergeSort_inversion(arr):

    inversion = 0

    sortedarr,inversion = mergeSort(arr,inversion)

    return sortedarr, inversion



def compareSortedArray(oldArray, newArray, end):

    for idx in range(0, end):

        if oldArray[idx][0] != newArray[idx][0] or oldArray[idx][1] != newArray[idx][1]:

            return True

    return False
def find(parent, i): 

    if parent[i] == i: 

        return i 

    return find(parent, parent[i])

  

def union(parent, rank, x, y): 

    xroot = find(parent, x) 

    yroot = find(parent, y) 

  

    if rank[xroot] < rank[yroot]: 

        parent[xroot] = yroot 

    elif rank[xroot] > rank[yroot]: 

        parent[yroot] = xroot 

    else : 

        parent[yroot] = xroot 

        rank[xroot] += 1

            

def KruskalMST(graph): 

    result =[] 

  

    parent = [] ; rank = [] 

    

    begin_index = -1

    end_index = -1



    i = 0 

    e = 0 

        

    for node in graph['vertices']:

        parent.append(node) 

        rank.append(0)

      

    while e < len(graph['vertices'])-1 :

        u,v,w = graph['edges'][i] 

        i = i + 1

        x = find(parent, u) 

        y = find(parent ,v)

            

        if x != y:

            e = e + 1     

            result.append([u,v,w]) 

            union(parent, rank, x, y)

            end_index = i-1

                

#     print ("Following are the edges in the constructed MST")

#     for u,v,weight  in result: 

#         print ("%d -- %d == %d" % (u,v,weight))

        

    return result, end_index
# graph = {

#     'vertices': [0,1,2,3,4,5,6,7,8], 

#     'edges': []

#         }



# graph['edges'].append([0,1,1])

# graph['edges'].append([2,4,2])

# graph['edges'].append([6,7,2])

# graph['edges'].append([6,8,3])

# graph['edges'].append([1,2,4])

# graph['edges'].append([2,3,5])

# graph['edges'].append([2,5,6])

# graph['edges'].append([0,2,7])

# graph['edges'].append([5,6,8])

# graph['edges'].append([7,8,9])



# graph = {

#     'vertices': [0,1,2,3,4,5], 

#     'edges': []

#         }



# graph['edges'].append([1,0,1])

# graph['edges'].append([1,2,2])

# graph['edges'].append([0,2,3])

# graph['edges'].append([3,0,4])

# graph['edges'].append([0,4,5])

# graph['edges'].append([3,4,6])

# graph['edges'].append([4,5,7])



# KruskalMST(graph)
def generate_spanning_tree(edges_matrix, spanning_tree):

    count = 1

    for edge in spanning_tree:

        weight = np.random.randint(1, max_weight)

        edges_matrix[edge[0]][edge[1]] = count

        edges_matrix[edge[1]][edge[0]] = count

        count+=1

    return edges_matrix



def generate_edges(edges_matrix, num_of_edges):

    n = num_of_edges



    while n > 0:

        from_vertex = np.random.randint(0, num_of_vertices)

        to_vertex = np.random.randint(0, num_of_vertices)

        if from_vertex == to_vertex or edges_matrix[from_vertex][to_vertex] != 0:

            continue

        weight = np.random.randint(1, max_weight)

        edges_matrix[from_vertex][to_vertex] = weight

        edges_matrix[to_vertex][from_vertex] = weight

        n -= 1

    return edges_matrix
num_of_vertices = 6

max_num_of_egdes = num_of_vertices*(num_of_vertices-1)/2

max_weight = 20

num_of_steps = 100



spanning_tree = [[1,0],[1,2],[3,0],[0,4],[4,5]]



def generate_graph():

    num_of_edges = np.random.randint(num_of_vertices-1, max_num_of_egdes)

    

    edges_matrix = np.zeros(num_of_vertices*num_of_vertices).reshape(num_of_vertices,num_of_vertices)

    edges_matrix = generate_spanning_tree(edges_matrix, spanning_tree)

    edges_matrix = generate_edges(edges_matrix, num_of_edges-num_of_vertices+1)



    graph = {'vertices': [i for i in range(0, num_of_vertices)], 'edges': []}

    

    for i in range(0, num_of_vertices):

        for j in range(i, num_of_vertices):

            if edges_matrix[i][j] != 0:

                graph['edges'].append([i,j, int(edges_matrix[i][j])])

    

    return graph



def generate_time_varying_network(num_of_steps):

    network = []

    while num_of_steps > 0:

        graph = generate_graph()

        network.append(graph)

        num_of_steps -= 1

        

    return network
network = generate_time_varying_network(num_of_steps)



network
count = 0

sucess = 0

result = []

old_sorted_edges = []

while count < num_of_steps:

    graph = network[count]

    sorted_edges, inversion = mergeSort_inversion(graph['edges'])



    graph['edges'] = sorted_edges

    isMSTChanged = True



    print ("core edges: ", old_sorted_edges)

    print ("sorted edges: ", graph['edges'])

    if len(old_sorted_edges) > 0:

        isMSTChanged = compareSortedArray(old_sorted_edges ,sorted_edges, end_index+1)

    



    if isMSTChanged:

        result, end_index = KruskalMST(graph)

        old_sorted_edges = graph['edges']

        print("New index: ", 0, end_index)

    else:

        result, end_index = KruskalMST(graph)

        print("MST not change")

        sucess+=1

        

    count+=1



print("Save MST computation times: ",sucess);


count = 0

sucess = 0

result = []

old_sorted_edges = []

start_time = time.time()

while count < num_of_steps:

    graph = network[count]

    sorted_edges, inversion = mergeSort_inversion(graph['edges'])

    graph['edges'] = sorted_edges

    KruskalMST(graph)    

    count+=1

end_time = time.time()

orginal_time = end_time - start_time
count = 0

sucess = 0

result = []

old_sorted_edges = []

start_time = time.time()

while count < num_of_steps:

    graph = network[count]

    sorted_edges, inversion = mergeSort_inversion(graph['edges'])

    

    graph['edges'] = sorted_edges

    isMSTChanged = True



    

    if len(old_sorted_edges) > 0:

        isMSTChanged = compareSortedArray(old_sorted_edges ,sorted_edges, end_index+1)

    



    if isMSTChanged:

        result, end_index = KruskalMST(graph)

        old_sorted_edges = graph['edges']

    else:

        sucess += 1

#         print("New index: ", 0, end_index)

    count+=1

end_time = time.time()

new_time = end_time - start_time
sucess
print('Compute MST evrey timestep: ',orginal_time)

print('Our algorithm: ',new_time)
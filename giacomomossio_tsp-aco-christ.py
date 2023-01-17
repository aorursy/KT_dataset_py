# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
import itertools

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from IPython.display import FileLink
import csv
def submission_generation(filename, str_output):
    os.chdir(r'/kaggle/working')
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in str_output:
            writer.writerow(item)
    return  FileLink(filename)
import math
from collections import namedtuple
Point = namedtuple("Point", ['n','x', 'y'])

def length(point1, point2):
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

class Min(object):
    def __init__(self, n, d):
      self.n = n
      self.d = d
def solve_it(input_data):
    # Modify this code to run your optimization algorithm

    # parse the input
    lines = input_data.split('\n')

    nodeCount = int(lines[0])

    #when creting the list of points I added a 'n' value that represent the node number, later will be essential
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(i-1,float(parts[0]), float(parts[1])))
        
        
    print(nodeCount)    
    
    
    if nodeCount < 2000:
        ###################christofide's algorithm###################
        #we use this only if the nodes are less that 1000
        #firstly we create the graph with the nodes and the weigthed edges
        G = nx.Graph()
        for i in points:
            G.add_node(i)
        for a in points:
            for b in points:
                G.add_edge(a,b,weight=length(a,b))
                
        #then we obtain the Minimum Spanning Tree
        T = nx.minimum_spanning_tree(G)
        
        #now we find the odds vertexes
        W = [0 for i in range(nodeCount)]
        for u,v in T.edges:
            W[u.n] = W[u.n] + 1
            W[v.n] = W[v.n] + 1
        W = [vertex for vertex, degree in enumerate(W) if degree % 2 == 1]
                
        #we create another graph with the odds vertex
        G2 = nx.Graph()
        for i in W:
            G2.add_node(i)
        for a in W:
            for b in W:
                G2.add_edge(a,b,weight=length(points[a],points[b]))
            
        #we calculate the perfect matching of the odds vertexes                
        M = nx.maximal_matching(G2)
        #and we add the found edges to the MST
        for a,b in M:
            T.add_edge(points[a],points[b],weight=length(points[a],points[b]))
            
            
        #now we check if the graph is eulerian and if so we obtain the eulerian circuit
        if nx.is_eulerian(T):
            print("executed with Christ")
            C = nx.eulerian_circuit(T)
            #now we just iterate the list and save all the nodes that we visit in solution[]
            solution = []
            for u,v in C:
                if(not u.n in solution):
                    solution.append(u.n)
                    
        ###################greedy###################
        #if the graph is not eulerian we use a greedy method
        #where we select the nearest nodes and create a path like that
        #actually this algorith is a bit slow and to execute all the files is gonna take like 1 hour
        else:
            print("not eulerian... executing greedy")
            #we need a copy of points for not modify it
            found = points.copy()
            solution = []
            count = 1
            #we start from the node 0
            a = found[0]
            solution.append(0)
            #as long as there are nodes in found we keep searching for the nearest node
            while(count < nodeCount):
                #we create an object where we save the index of the nearest node and its distance
                minimum = (Min(0,1000000000))
                for b in found:
                    if a != b and length(a,b) < minimum.d:
                        minimum.n, minimum.d = b.n, length(a,b)
                #now we remove the node from the found list
                found.remove(a)
                count += 1
                #we put the node founded in solution and update the node a
                solution.append(minimum.n)
                a = points[minimum.n]
                
    #if the node number is too large we use the same greedy algorithm as before
    else:
        print("too big... executing greedy")
        found = points.copy()
        solution = []
        count = 1
        a = found[0]
        solution.append(0)
        
        while(count < nodeCount):
            minimum = (Min(0,1000000000))
            
            for b in found:
                if a != b and length(a,b) < minimum.d:
                    minimum.n, minimum.d = b.n, length(a,b)
        
            found.remove(a)
            count += 1
            solution.append(minimum.n)
            a = points[minimum.n]
            
            
    ################### 2-OPT ###################
    #now, starting from the obtained solution we try make it a little better
    #the times value is how many times we iterate the list
    times = 100
    while times > 0:
        for i in range (0, len(solution)-3):
            actual = length(points[i],points[i+1])
            actual += length(points[i+1],points[i+2])
            #we calculate the length of the sequence points[i], points[i+2], points[i+1]
            changed = length(points[i],points[i+2])
            changed += length(points[i+2],points[i+1])

            if changed < actual:

                solution[i+1],solution[i+2] = solution[i+2], solution[i+1]

        times -= 1
    
    
            
    
    if len(solution) != nodeCount:
        print("ERRORRRRRRRRR")
    # calculate the length of the tour
    obj = length(points[solution[-1]], points[solution[0]])
    for index in range(0, nodeCount-1):
        obj += length(points[solution[index]], points[solution[index+1]])

    # prepare the solution in the specified output format
    output_data = '%.2f' % obj + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data, obj
str_output = [["Filename","Value"]]
for dirname, _, filenames in os.walk('/kaggle/input/tsp-aco'):
    for filename in filenames:
        full_name = dirname+'/'+filename
        with open(full_name, 'r') as input_data_file:
            input_data = input_data_file.read()
            output, value = solve_it(input_data)
            str_output.append([filename,str(value)])
str_output
submission_generation('sample_submission_non_sorted.csv', str_output)
reader = csv.reader(open("sample_submission_non_sorted.csv"))
sortedlist = sorted(reader, key=lambda row: row[0], reverse=False)
submission_generation('sample_submission.csv', sortedlist)
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Adding imports that will be used
from collections import namedtuple
import networkx as nx
import math
# Definition of the tuples that will be read from the input files. These tuples of two positions represent the coordinates of the cities
Point = namedtuple("Point", ['x', 'y'])

def distance(point1, point2):
    """ Function to compute the euclidean distance between two points (two cities) """
    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
def readPoints(input_data):
    """ This function takes all lines in a input data (coming from a file) and transforms it into a points array """
    lines = input_data.split('\n')
    nodeCount = int(lines[0])
    points = []
    for i in range(1, nodeCount+1):
        line = lines[i]
        parts = line.split()
        points.append(Point(float(parts[0]), float(parts[1])))
    return nodeCount, points
def complete_graph(points):
    """ This code must be completed to finish the notebook. In this code, using the points given in the argument a graph must be created using NetworkX. As hint, it is recommended to first create all nodes and then define all edges between nodes, using the distance function. Once it is finished, change this documentation to document the implementation. """
    
    # Create graph
    graph = None
    # Define nodes in the graph
    # Define edges in the graph
    return graph
with open('/kaggle/input/ap-algoritmos-y-programacin-2020-v2/', 'r') as input_data_file:
    input_data = input_data_file.read()
    nodeCount, points = readPoints(input_data)
    graph = complete_graph(points)
    if graph != None:
        print("Nodes in graph: ", graph.nodes(data=True))
        print("Edges in graph: ", graph.edges(data=True))
    
# Expected answers for this example:
# Nodes in graph:  [(0, {}), (1, {}), (2, {}), (3, {}), (4, {})]
# Edges in graph:  [(0, 1, {'weight': 0.5}), (0, 2, {'weight': 1.0}), (0, 3, {'weight': 1.4142135623730951}), (0, 4, {'weight': 1.0}), (1, 2, {'weight': 0.5}), (1, 3, {'weight': 1.118033988749895}), (1, 4, {'weight': 1.118033988749895}), (2, 3, {'weight': 1.0}), (2, 4, {'weight': 1.4142135623730951}), (3, 4, {'weight': 1.0})]
def calculate_distance_tour(solution, points):
    """ Given the solution and the points, here it is computed the cost by summing all distances between the cities considering the order of the solution """
    cost = distance(points[solution[-1]], points[solution[0]])
    for index in range(0, len(solution) - 1):
        cost += distance(points[solution[index]], points[solution[index+1]])
    return cost

def check_solution(solution, points):
    """ This method checks that the solution is a valid solution. It is checked that a city is not visited twice. If it's correct, it returns the distance of the journey """
    if len(solution) != len(points) or len(set(solution)) != len(points):
        print('Solución Incorrecta: No pasa por todos los nodos una única vez')
        return 0
    
    return calculate_distance_tour(solution, points)
def solve_2_approx(points, flag_nx_MST= False):
    """ This function must be defined to complete the practice. To fulfill all steps, NetworkX must be used. Modify this documentation to describe the implementation you did. """
    
    solution = None
    # 1.Create graph using the previous implemented function complete graph
    # 2.Find minimun spanning tree, T
    # 3.Duplicate T edges, G*
    # 4.Find C, Eulerian chain of G*
    # 5.Skip duplicates of C to find the Hamiltonian Path 
    return solution
def solve_it(input_data):
    """ This function takes input data that describes a specific problem of TSP and solve it. This is how it does:
        1 - Parse the input and transform the content of each lines into Points
        2 - If the list of cities is very big (> 1000)
        2a - It is not solved
        2b - Otherwise the previous implemented method is called.
        3 - Prepare the solution in the specified output format
        
    """
    nodeCount, points = readPoints(input_data)
    
    if nodeCount>1000:
        solution = range(0, nodeCount)
    else:
        solution = solve_2_approx(points, flag_nx_MST= True)
        
    cost = calculate_distance_tour(solution, points)
    output_data = '%.2f' % cost + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, solution))

    return output_data, check_solution(solution, points)
# Let's check the solution of the same example
with open('/kaggle/input/ap-algoritmos-y-programacin-2020-v2/', 'r') as input_data_file:
    input_data = input_data_file.read()
    example_output, example_distance = solve_it(input_data)
    print(example_distance)
    
# Expected distance for this example: 5.03224755112299
# This code iterates over all files in put directory and send the problem definitions to the solve it function
import time

total_time = 0
import gc
str_output_kaggle = [["Filename","Value"]]
str_output_moodle = [["Filename","Value", "Solution"]]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        full_name = dirname+'/'+filename
        with open(full_name, 'r') as input_data_file:
            input_data = input_data_file.read()
            
            start = time.time()
                
            output, value = solve_it(input_data)
            gc.collect()
            str_output_kaggle.append([filename,str(value)])
            str_output_moodle.append([filename,str(value), output.split('\n')[1]])
                
            end = time.time()
                
            print(filename, '(', value, ') time:', end - start)
            total_time += end-start
                
print('total time: ', total_time)
# This function generates the submission file that must be upload to the competition
from IPython.display import FileLink
import sys
import csv
csv.field_size_limit(sys.maxsize)
def submission_generation(filename, str_output):
    os.chdir(r'/kaggle/working')
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for item in str_output:
            writer.writerow(item)
    return  FileLink(filename)
submission_generation('algorithm_2_approx_unsorted_kaggle.csv', str_output_kaggle)
# The file generated by this method must be uploaded in the task of the "campus virtual". The file to upload in the "campus virtual" must be the one related to one submitted to Kaggle. That is, both submitted files must be generated in the same run
submission_generation('algorithm_2_approx_unsorted_moodle.csv', str_output_moodle)
reader = csv.reader(open("algorithm_2_approx_unsorted.csv"))
sortedlist = sorted(reader, key=lambda row: row[0], reverse=False)
# This file must be submitted in the competition of Kaggle
submission_generation('algorithm_2_approx_kaggle.csv', sortedlist)
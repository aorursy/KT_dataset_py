# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
def check_solution(node_count, edges, solution):

    for edge in edges:

        if solution[edge[0]]==solution[edge[1]]:

            print("solución inválida, dos nodos adyacentes tienen el mismo color")

            return 0

    value = max(solution)+1 #add one because minimum color is 0

    

    return value
def solve_it(input_data):

    # Modify this code to run your optimization algorithm



    # parse the input

    lines = input_data.split('\n')



    first_line = lines[0].split()

    node_count = int(first_line[0])

    edge_count = int(first_line[1])



    edges = []

    for i in range(1, edge_count + 1):

        line = lines[i]

        parts = line.split()

        edges.append((int(parts[0]), int(parts[1])))



    # build a trivial solution

    # every node has its own color

    solution = range(0, node_count)



    # prepare the solution in the specified output format

    output_data = str(node_count) + ' ' + str(0) + '\n'

    output_data += ' '.join(map(str, solution))

    

    return output_data, check_solution(node_count, edges, solution)
str_output = [["Filename","Min_value"]]



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        full_name = dirname+'/'+filename

        with open(full_name, 'r') as input_data_file:

            input_data = input_data_file.read()

            output, value = solve_it(input_data)

            str_output.append([filename,str(value)])
import csv

from IPython.display import FileLink

def submission_generation(filename, str_output):

    os.chdir(r'/kaggle/working')

    with open(filename, 'w', newline='') as file:

        writer = csv.writer(file)

        for item in str_output:

            writer.writerow(item)

    return  FileLink(filename)
submission_generation('NAME_sample.csv', str_output)
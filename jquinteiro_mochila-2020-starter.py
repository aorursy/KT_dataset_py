# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import csv

from collections import namedtuple

Item = namedtuple("Item", ['index', 'value', 'weight'])
from IPython.display import FileLink

def submission_generation(filename, str_output):

    os.chdir(r'/kaggle/working')

    with open(filename, 'w', newline='') as file:

        writer = csv.writer(file)

        for item in str_output:

            writer.writerow(item)

    return  FileLink(filename)
def check_solution(capacity, items, taken):

    weight = 0

    value = 0

    for item in items:

        if taken[item.index]== 1:

            weight += item.weight

            value += item.value

    if weight> capacity:

        print("solución incorrecta, se supera la capacidad de la mochila (capacity, weight):", capacity, weight)

        return 0

    return value
def solve_it(input_data):

    # Modify this code to run your optimization algorithm



    # parse the input

    lines = input_data.split('\n')



    firstLine = lines[0].split()

    item_count = int(firstLine[0])

    capacity = int(firstLine[1])



    items = []



    for i in range(1, item_count+1):

        line = lines[i]

        parts = line.split()

        items.append(Item(i-1, int(parts[0]), int(parts[1])))



    # a trivial greedy algorithm for filling the knapsack

    # it takes items in-order until the knapsack is full

    value = 0

    weight = 0

    taken = [0]*len(items)



    for item in items:

        if weight + item.weight <= capacity:

            taken[item.index] = 1

            value += item.value

            weight += item.weight

            

    # prepare the solution in the specified output format

    output_data = str(value) + ' ' + str(0) + '\n'

    output_data += ' '.join(map(str, taken))

    return output_data, check_solution(capacity, items, taken)
str_output = [["Filename","Max_value"]]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        full_name = dirname+'/'+filename

        with open(full_name, 'r') as input_data_file:

            input_data = input_data_file.read()

            output, value = solve_it(input_data)

            str_output.append([filename,str(value)])
submission_generation('prueba.csv', str_output)
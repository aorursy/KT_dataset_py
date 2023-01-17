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
Point = namedtuple("Point", ['x', 'y'])



def length(point1, point2):

    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
def solve_it(input_data):

    # Modify this code to run your optimization algorithm



    # parse the input

    lines = input_data.split('\n')



    nodeCount = int(lines[0])



    points = []

    for i in range(1, nodeCount+1):

        line = lines[i]

        parts = line.split()

        points.append(Point(float(parts[0]), float(parts[1])))



    # build a trivial solution

    # visit the nodes in the order they appear in the file

    solution = range(0, nodeCount)



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
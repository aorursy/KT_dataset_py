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
import math

from collections import namedtuple

Point = namedtuple("Point", ['x', 'y'])



def length(point1, point2):

    return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)
def solve_it(input_data):

    # Modify this code to run your algorithm



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
str_output_kaggle = [["Filename","Value"]]

str_output_moodle = [["Filename","Value", "Solution"]]

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        full_name = dirname+'/'+filename

        with open(full_name, 'r') as input_data_file:

            input_data = input_data_file.read()

            output, value = solve_it(input_data)

            str_output_kaggle.append([filename,str(value)])

            str_output_moodle.append([filename,str(value), output.split('\n')[1]])
str_output_moodle[0:2]
submission_generation('L1_G1_Nombres_Starter.csv', str_output_kaggle)
from IPython.display import FileLink

import csv

def submission_generation(filename, str_output):

    os.chdir(r'/kaggle/working')

    with open(filename, 'w', newline='') as file:

        writer = csv.writer(file)

        for item in str_output:

            writer.writerow(item)

    return  FileLink(filename)
submission_generation('L1_G1_Nombres_Starter_solution.csv', str_output_moodle)
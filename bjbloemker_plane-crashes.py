import pandas as pd 

import csv as csv 

import numpy as np



test_file = open('../input/Airplane_Crashes_and_Fatalities_Since_1908.csv', 'r')

test_file_object = csv.reader(test_file)

next(test_file_object)
total = 0

morethan75 = 0

for row in test_file_object:

    if row[9] == '' or row[10] == '' or float(row[9]) == 0:

        total = total + 1

    else:

        total = total + 1

        if float(row[10])/float(row[9]) > .75:

            morethan75 = morethan75 + 1

            

print(morethan75/total)
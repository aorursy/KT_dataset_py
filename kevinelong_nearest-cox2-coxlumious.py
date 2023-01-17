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
import math


def distance(p1, p2):
    d1 = p1[0] - p2[0]
    print(d1)

    d2 = p1[1] - p2[1]
    print(d1)

    s1 = d1 ** 2
    print(s1)

    s2 = d2 ** 2
    print(s2)

    t = s1 + s2
    print(t)

    d = math.sqrt(t)

    return d


def distance2(p1, p2):
    return math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


print(distance([3.5, 3.5], [5.5, 5.5]))
print(distance2([3.5, 3.5], [5.5, 5.5]))

locations = [
    [5, 5],
    [2, 5],
    [2, 6],
    [1, 4],
    [7, 7],
]
current = [3, 2]

def find_nearest(current, locations):
    results = []

    for index in range(0, len(locations)):
        item = locations[index]

        d = distance2(item, current)
        results.append( { "index" : index, "distance" : d } )
        
    return results


result = find_nearest(current, locations)

# TODO SORT OUTPUT
print(result)

# what is the shortest path from the current location to visit all,
#  and then return to the start.
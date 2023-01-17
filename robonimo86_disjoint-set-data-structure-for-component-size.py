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



data = open(os.path.join(dirname, filename))
n = int(data.readline())

edges = [[int(num) for num in data.readline().split()] for _ in range(n)]
'''

This is a mostly functional implementation of the disjoint set data structure. 

Near verbatim translation of the pseudocode at 

https://en.wikipedia.org/wiki/Disjoint-set_data_structure

'''

class Node:

    def __init__ (self, label):

        self.label = label

    def __str__(self):

        return self.label



def make_set(x):

    x.parent = x

    x.rank = 0

    x.size = 1



def union_by_size(x, y):

    '''

    Merges the sets by joining the smaller set to the larger set.

    '''

    x_root = find(x)

    y_root = find(y)

    # If roots coincide, nothing to do.

    if x_root == y_root:

        return

    # If x is smaller, swap x and y.

    elif x_root.size < y_root.size:

        x_root, y_root = y_root, x_root

    # Merge the trees.

    y_root.parent = x_root

    # Update the parent's size.

    x_root.size += y_root.size





def find(x):

    '''

    Utilizes path compression.

    '''

    if x.parent != x:

        x.parent = find(x.parent)

    return x.parent





def componentsInGraph(gb):

    # Initialize each node as a disconnected component.

    components = [Node(i) for i in range(1, 2*n+1)]

    component_sizes = []



    [make_set(v) for v in components]



    for e in gb:

        v1, v2 = e

        union_by_size(components[v1-1], components[v2-1])

    for v in components:

        v.parent = find(v)

        component_sizes.append(v.parent.size)



    return min(cpnt for cpnt in component_sizes if cpnt > 1), max(component_sizes)



sizes = componentsInGraph(edges)
print(sizes)
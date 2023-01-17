# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
from random import shuffle

filename = 'data1.csv'
#hyperparameters of the algorithm
n_folds =20
max_depth = 10
min_size = 5


with open('../input/data1.csv','r') as f:
    contents = f.readlines()

print  (contents)

contents = [i.strip("\n").split(",") for i in contents]

train_data = [[float(val) for val in row] for row in contents]

train_data
with open('../input/test1.csv','r') as f:
    contents = f.readlines()

print  (contents)

contents = [i.strip("\n").split(",") for i in contents]

test_data = [[float(val) for val in row] for row in contents]
print (test_data)

from math import log
def information_gain(groups,class_values):
    value = 0.0
    for class_value in class_values: #for each class in {0, 1} 
        for group in groups:
            number = float(0)
            if len(group) == 0:
                continue
            for row in group:
                if(row[-1] == class_value):
                    number += 1.0
            prob = number / float(len(group))
            # this is the probability for that class
            if (prob == 0):
                continue
            else:
                value += -1 * (prob * log(prob)) #formula to calculate entropy
    return value

# Based on an attribute index from dset and a given value, break the dataset into two parts
def test_split(ind, val, dset):
    group1, group2 = [], []
    for r in dset:
        if r[ind] < val:
            group1.append(r)
        else:
            group2.append(r)
    return group1, group2

# Calculate the best point for a split in the dataset
def splitting(dataset):
    class_values = [0, 1] # these are the class vals for binary data
    node_index=1000 #a high val
    node_value=1000 #a high val
    node_score=1000 #a high val
    node_groups = None
    for index in range(len(dataset[0])-1):
        count= float(0)
        count_row = float(0)
        for row in dataset: #here we take the average of the values in a attribute to split the data 
            count += 1
            count_row = row[index] + count_row
        count_row = float(count_row)/float(count)
        groups = test_split(index,count_row,dataset)
        value = information_gain(groups,class_values)
        if value < node_score:#store the values based on which the information gain is lowest
        # on the basis of that attribute 
        # The less than sign ensures the condition given from question
                node_index, node_value, node_score, node_groups = index, row[index], value, groups
    return {'i':node_index, 'value':node_value, 'div':node_groups}#return the dictionary containing the details 

def terminal_node(group):
    outcomes = [row[-1] for row in group]#this returns all the labels for the data and stores in the outcome list
    outc=[0,0]
    for out in outcomes:
        outc[(int(out))]+=1
    if(outc[0]>outc[1]):
        return 0
    else:
        return 1

# Create child splits for a node or make terminal
def node_from_tree_split(node_from_tree,depth):
    left, right = node_from_tree['div']
    del(node_from_tree['div'])
    # check if there is no split and the data is coherent. If all the values in the attribute is greater or less that the particular avg value of the attribute
    if not left or not right:
        node_from_tree['left'] = node_from_tree['right'] = terminal_node(left + right)
        return
    # check for max depth is achieved
    if depth >= max_depth:
        node_from_tree['left'], node_from_tree['right'] = terminal_node(left), terminal_node_from_tree(right)
        return
    # process left child
    if len(left) <= min_size:#if the mininmum records in the data is less than or equal to the min_size. No splitting
        node_from_tree['left'] = terminal_node(left)
    else:
        node_from_tree['left'] = splitting(left)#otherwise split the data further keeping in mind the min_size and the max_depth
        node_from_tree_split(node_from_tree['left'], depth+1)
    # process right child same as above
    if len(right) <= min_size:
        node_from_tree['right'] = terminal_node(right)
    else:
        node_from_tree['right'] = splitting(right)
        node_from_tree_split(node_from_tree['right'], depth+1)

 
# Make a prediction from the tree
def output_from_tree(node_from_tree, row):
    if row[node_from_tree['i']] < node_from_tree['value']:
        if (type(node_from_tree['left'])== dict):#if there exists a node_from_tree['left'] of type dict
            return output_from_tree(node_from_tree['left'], row)
        else:
            return node_from_tree['left']#otherwise return the label depicted by the terminal node_from_tree
    else:#same for the right side of the subtree
        if (type(node_from_tree['right'])== dict):
            return output_from_tree(node_from_tree['right'], row)
        else:
            return node_from_tree['right']

 
[test_data[i].append(None) for i in range(len(test_data))]

test_data
root= splitting(train_data)
node_from_tree_split(root,1)
predictions = list()
for row in test_data:
    prediction = output_from_tree(root, row)
    predictions.append(prediction)

print (predictions)
f = open('out.csv', 'w')
f.write("Id,Class\n")
f.write("1,"+str(predictions[0])+"\n")
f.write("2,"+str(predictions[1])+"\n")
f.write("3,"+str(predictions[2])+"\n")
f.write("4,"+str(predictions[3])+"\n")
f.close()

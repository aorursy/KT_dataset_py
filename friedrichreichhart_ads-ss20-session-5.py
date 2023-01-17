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
class Node:
    def __init__(self, dataval=None):
        self.dataval = dataval
        self.nextval = None
class LinkedList:
    def __init__(self):
        self.headval = None
    
    def printlist(self):
        #print(self.headval.dataval)
        printval = self.headval
        while printval is not None:
            print(printval.dataval)
            printval = printval.nextval
    
    def insertAtBegin(self, newval):
        newNode = Node(newval)
        newNode.nextval = self.headval
        self.headval = newNode
        
    def insertAtEnd(self, newval):
        newNode = Node(newval)
        ## headval is none // lsit is empty
        if self.headval is None:
            self.headval = newNode
            return
        ## at least one or more nodes
        lastNode = self.headval
        while(lastNode.nextval):
            lastNode = lastNode.nextval
        lastNode.nextval = newNode
        
    def insertAfterNode(self, node, newval):
        # node is before newNode
        # check if Node is None -> print "error" and return
        # use as node nodeB
        if node is None:
            print("Node does not exist - Error")
            return
        
        newNode = Node(newval)
        newNode.nextval = node.nextval
        node.nextval = newNode
        
    def removeNode(self, node2Remove):
        if node is None:
            print("Node does not exist - Error")
            return
 
        node = self.headval
        while(node.nextval):
            if node.nextval == node2Remove:
                node.nextval = node2Remove.nextval
                return

mylist = LinkedList()
mylist.headval = Node("A")
mylist.printlist()
nodeB = Node("B")

nodeC = Node("C")
mylist.headval.nextval = nodeB
nodeB.nextval =  nodeC
mylist.printlist()
mylist.insertAtBegin("NewNode")
mylist.printlist()
mylist.insertAtEnd("NewEndNode")
mylist.printlist()
mylist.insertAfterNode(nodeB, "NewInsertedNode")
mylist.printlist()

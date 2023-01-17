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
        self.prevval = None

class LinkedList:
    def __init__(self):
        self.headval = None
    
    def printlist(self):
        node = self.headval
        while node is not None:
            print(f"Node: {node.dataval}")
            if node.nextval is not None:
                print(f"Nextval: {node.nextval.dataval}")
            if node != self.headval:
                print(f"Prevval: {node.prevval.dataval}")
            print("-------")
            node = node.nextval

    def insertAtBegin(self,newval):
        newnode = Node(newval)
        newnode.nextval = self.headval
        self.headval.prevval = newnode
        self.headval = newnode
        
    def insertAtEnd(self,newval):
        newnode = Node(newval)
        if self.headval is None:
            self.headval = newnode
            return
        node = self.headval
        while node.nextval:
            node = node.nextval
        node.nextval = newnode
        newnode.prevval = node
    
    def insertAfterNode(self,node, newval):
        # node is before newnode
        # check if Node is None -> print "error" and return
        if node is None:
            print("Node does not exist - Error")
            return
        
        newnode = Node(newval)
        newnode.nextval = node.nextval
        newnode.prevval = node
        node.nextval.prevval = newnode
        node.nextval = newnode
    
    def removeNode(self,node2Remove):
        if node2Remove is None:
            print("Node does not exist - Error")
            return
        node = self.headval
        
        # Special case if node2Remove is headnode
        if self.headval == node2Remove:
            node2Remove.nextval.prevval = None
            self.headval = self.headval.nextval
            node2Remove.nextval = None
            
        
        while node.nextval:
            if node.nextval == node2Remove:
                node.nextval = node2Remove.nextval
                if node.nextval is not None:
                    node2Remove.nextval.prevval = node
                node2Remove.nextval = None
                #node2Remove.dataval = None
                node2Remove.prevval = None
                return
            node = node.nextval
mylist = LinkedList()
mylist.headval = Node("1")
mylist.insertAtEnd("5")
mylist.insertAtEnd("10")
mylist.insertAfterNode(mylist.headval.nextval, "7")
mylist.insertAtBegin("0")
mylist.printlist()

mylist.removeNode(mylist.headval)
mylist.printlist()
mylist.removeNode(mylist.headval.nextval)
mylist.printlist()
mylist.removeNode(mylist.headval.nextval.nextval)
mylist.printlist()
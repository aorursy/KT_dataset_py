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
        node = self.headval
        while node is not None:
            if node.nextval is None:
                print(f"{node.dataval}, ")
                break
            print(f"{node.dataval}, ", end="")
            node = node.nextval

    def insertAtBegin(self,newval):
        newnode = Node(newval)
        newnode.nextval = self.headval
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
    
    def insertAfterNode(self,node, newval):
        # node is before newnode
        # check if Node is None -> print "error" and return
        if node is None:
            print("Node does not exist - Error")
            return
        
        newnode = Node(newval)
        newnode.nextval = node.nextval
        node.nextval = newnode
    
    def removeNode(self,node2Remove):
        if node2Remove is None:
            print("Node does not exist - Error")
            return
        node = self.headval
        while node.nextval:
            if node.nextval == node2Remove:
                node.nextval = node2Remove.nextval
                node2Remove.nextval = None
                #node2Remove.dataval = None
                return
            node = node.nextval
            
    # Function to insert an array
    def insertValuesAtEnd(self, data):
        for i in data:
            self.insertAtEnd(i)
    
    # Function to get length of the Linked List
    def getLength(self):
        node = self.headval
        length = 0
        if node is not None:
            length += 1
        while node.nextval is not None:
            node = node.nextval
            length += 1
        return length
mylist = LinkedList()
mylist.insertValuesAtEnd([35,20,15,16,96,45,81,88,2,86,2,4,10,19,60,84,87,3,34,70])
print(mylist.getLength())
mylist.printlist()
def bubblesortLL(linkedlist):
    # get length
    length = linkedlist.getLength()
    
    if length <= 1:
        return linkedlist

    for i in range(length-1):
        node = linkedlist.headval
        while node.nextval is not None:
            # Conditions if the headval is in the comparison
            if node == linkedlist.headval:
                if node.dataval > node.nextval.dataval:
                    linkedlist.headval = node.nextval
                    helper1 = node.nextval.nextval
                    node.nextval.nextval = node
                    node.nextval = helper1
              
            else:
                if node.dataval > node.nextval.dataval:
                    helper1 = node.nextval.nextval
                    node.nextval.nextval = node
                    helper2 = node.nextval
                    node.nextval = helper1
                    
                    prevNode = linkedlist.headval
                    while prevNode.nextval is not None:
                        if prevNode.nextval == node:
                            prevNode.nextval = helper2
                            break
                        prevNode = prevNode.nextval
                        
            # Check if nextval is None, if it is then the loops ends
            if node.nextval is not None:
                node = node.nextval
    return linkedlist
                
                
bubblesortLL(mylist)
mylist.printlist()
print(sorted([35,20,15,16,96,45,81,88,2,86,2,4,10,19,60,84,87,3,34,70]))
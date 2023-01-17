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
class Stack:
    
    def __init__(self):
        self.stack = []
        
    def push(self, value):
        self.stack.append(value)
        
    def printTop(self):
        print(self.stack[-1])
        
    def pop(self):
        if (len(self.stack) == 0):
            print("No Items left on stack")
            return ""
        else:
            return self.stack.pop()
TestStack = Stack()

TestStack.push("Plate1")
TestStack.push("Plate2")
TestStack.push("Plate3")
TestStack.printTop()
TestStack.push("Plate4")
TestStack.printTop()
print(TestStack.pop())
StackLeft = Stack()
StackRight = Stack()

StackLeft.push("Plate1")
StackLeft.push("Plate2")
StackLeft.push("Plate3")
StackLeft.push("Plate4")
StackLeft.push("Plate5")
StackLeft.push("Plate6")


StackElement = StackLeft.pop()
while (StackElement != ""):
    print(StackElement)
    StackRight.push(StackElement)
    StackElement = StackLeft.pop()

StackElement = StackRight.pop()
while (StackElement != ""):
    print(StackElement)
    StackLeft.push(StackElement)
    StackElement = StackRight.pop()
class Queue:
    
    def __init__(self):
        self.queue = list()
        
    def add2Q(self, value):
        self.queue.insert(0, value)
        
    def size(self):
        return len(self.queue)
    
    def getFQ(self):
        if (len(self.queue)>0):
            return self.queue.pop()
        else:
            print("No Elements in Queue")
            return ""
    
MyQueue = Queue()

MyQueue.add2Q("Plate1")
MyQueue.add2Q("Plate2")
MyQueue.add2Q("Plate3")
MyQueue.add2Q("Plate4")
MyQueue.add2Q("Plate5")
MyQueue.add2Q("Plate6")

print(MyQueue.size())
print(MyQueue.getFQ())
class Node:
    
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data
        
    def insert(self, data):
        if (self.data):
            if (data < self.data):
                if (self.left is None):
                    self.left = Node(data)
                else:
                    self.left.insert(data)
            elif  (data > self.data):
                if (self.right is None):
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
           self.data = data 
                
                    
    
    def printTree(self):
        if (self.left):
            self.left.printTree()
            
        print(self.data)
        
        if(self.right):
            self.right.printTree()
        
rootNode = Node(5)
rootNode.printTree()
rootNode.insert(4)
rootNode.insert(3)
rootNode.insert(7)
rootNode.insert(6)
rootNode.insert(8)
rootNode.insert(8)

rootNode.printTree()
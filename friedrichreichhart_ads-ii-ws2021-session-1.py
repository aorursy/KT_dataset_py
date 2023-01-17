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
from queue import *

class Node:
    
    def __init__(self,data):
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
            elif (data > self.data):
                if (self.right is None):
                    self.right = Node(data)
                else:
                    self.right.insert(data)
        else:
            self.data = data
            
    def dfs_inorder_traversal(self):
        elements = []
        
        # visit left-tree-side
        if self.left:
            elements += self.left.dfs_inorder_traversal()
        
        # visit root-node
        elements.append(self.data)
        
        
        #visit right-tree-side
        if (self.right):
            elements += self.right.dfs_inorder_traversal()
            
        # return
        return elements

    def dfs_preorder_traversal(self):
        elements = []
        
        # visit root-node
        elements.append(self.data)
        
        # visit left-tree-side
        if self.left:
            elements += self.left.dfs_preorder_traversal()
        
        
        #visit right-tree-side
        if (self.right):
            elements += self.right.dfs_preorder_traversal()
            
        # return
        return elements

    def dfs_postorder_traversal(self):
        elements = []
        
        # visit left-tree-side
        if self.left:
            elements += self.left.dfs_postorder_traversal()
        
        #visit right-tree-side
        if (self.right):
            elements += self.right.dfs_postorder_traversal()

        # visit root-node
        elements.append(self.data)
            
        # return
        return elements
    
    def bfs(self):
        elements = []
        queue = Queue()
        queue.put(self)
        
        while not queue.empty():
            self = queue.get()
            elements.append(self.data)
            
            if self.left:
                queue.put(self.left)
                
            if self.right:
                queue.put(self.right)
        return elements
    
    def minval(node):
        current = node
        
        while(current.left is not None):
            current = current.left
            
        return current.data
        

    
def build_tree(elements):
    root = Node(elements[0]);
    for i in range(1, len(elements)):
        root.insert(elements[i])

    return root 
numbers = [15,12,7,14,27,20,23,89]
numbers_tree = build_tree(numbers)

numbers_dfs_inorder = numbers_tree.dfs_inorder_traversal()
print('Inorder')
print(numbers_dfs_inorder)

numbers_dfs_preorder = numbers_tree.dfs_preorder_traversal()
print('Preorder')
print(numbers_dfs_preorder)

numbers_dfs_postorder = numbers_tree.dfs_postorder_traversal()
print('Postorder')
print(numbers_dfs_postorder)

print('BFS')
print(numbers_tree.bfs())

print('minValue')
print(numbers_tree.minval())
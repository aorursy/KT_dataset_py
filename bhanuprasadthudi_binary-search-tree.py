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

    def __init__(self,data):

        self.right=self.left=None

        self.data = data

class bhanu:

    def insert(self,root,data):

        if root==None:

            return Node(data)

        else:

            if data<=root.data:

                cur=self.insert(root.left,data)

                root.left=cur

            else:

                cur=self.insert(root.right,data)

                root.right=cur

        return root



    def getHeight(self,root):

        if root==None:

            return -1

        lft=self.getHeight(root.left)

        rht=self.getHeight(root.right)



        return max(lft,rht)+1



print("how many values you want in binary tree:")    

t=int(input())

tree=bhanu()

root=None

for i in range(t):

    print("enter data:")

    data=int(input())

    root=tree.insert(root,data)

h=tree.getHeight(root)

print(tree)

print("the longet path=",h)

    

    

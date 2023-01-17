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
#program to append,delete,display elements in a class

class check():

    def __init__(self):

        self.n=[]

    def add(self,a):

        return self.n.append(a)

    def rem(self,b):

        return self.n.remove(b)

    def dis(self):

        return self.n

obj=check()

choice=1

while(choice!=0):

    print("0.Exit")

    print("1.Add")

    print("2.Delete")

    print("3.Display")

    choice=int(input("enter the choice:"))

    if(choice==1):

        n=int(input("enter the number to be added:"))

        obj.add(n)

        print("list:",obj.dis())

    elif(choice==2):

        n=int(input("enter the number to be deleted:"))

        obj.rem(n)

        print("list:",obj.dis())

    elif(choice==3):

        print("list",obj.dis())

    elif(choice==0):

        print("exiting!")

    else:

        print("invalid")

print()
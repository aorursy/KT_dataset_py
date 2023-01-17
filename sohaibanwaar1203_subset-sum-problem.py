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
def subset(set):

    #Base case: if set is empty set,return himself.

    if len(set) == 0:

        return[set]

    #if set is only has one element, return

    elif len(set) == 1:

        return [[]] + [set]

    #recursive thinking

    else:

        #split the set into the first and rest and get the rest element of set

        rest = subset(set[1:])

        #the list of all subset

        alist = []

        # for every element in the the rest part

        for item in rest:

            #first element in subset add every element in the rest

            blist = [set[0]]

            blist += item

            alist.append(blist)

        #rest is the subset of rest (last subset call)

        return rest + alist
set1=[1,2,3]

a=subset(set1)

list_of_goal_subsets=[]

goal=6

for i in a:

    sum1=0

    print(i)

    for j in i:

        

        

        sum1=j+sum1

        

    if(sum1==goal):

        print(sum1)

        list_of_goal_subsets.append(i)

            

        
list_of_goal_subsets
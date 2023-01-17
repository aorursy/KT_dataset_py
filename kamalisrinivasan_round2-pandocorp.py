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
# Defining variables for big cuboidal container into which the small cuboids are gonna placed

# Big cuboidal container dimensions

L = 700 # Length

B = 500 # Breadth

H = 500 # Height



# Small cuboidal boxes dimensions

range_of_small_box_dim = [1,3]



# Number of small cuboids

N = 500
a = list(range(range_of_small_box_dim[0],range_of_small_box_dim[1]+1))

print(a)
# Creating an numpy array with its respective dimensions and population value as zero indicating that the sapce is empty

Big_Container = np.zeros((L,B,H))
for box in range(N):

    # Randomly formulating the dimensions for each small cuboid with respect to the range defined

    temp = list((np.random.choice(a,3,replace=True)))

    while(len(list(set(temp))) > 1):

        l,b,h = temp

        temp = list((np.random.choice(a,3,replace=True)))

    temp = [l,b,h]

    if len(list(set(temp))) == 3:

        temp = sorted(temp)

        l = temp[2]

        b = temp[1]

        h = temp[0]

    else:

        if temp.count(list(set(temp))[0]) == 1:

            l = list(set(temp))[0]

            b = h = list(set(temp))[1]

        else:

            l = list(set(temp))[1]

            b = h = list(set(temp))[0]

            

    Small_cuboids_dim = np.ones((l,b,h))

    flag = 0

    for length in range(L-l+1):

        for breadth in range(B-b+1):

            for height in range(H-h+1):

                val = np.multiply(Big_Container[length:length+l,breadth:breadth+b,height:height+h],Small_cuboids_dim)

                if np.sum(val) == 0:

                    print("Placed box no.",box+1, "Position:","length =",length,'breadth =',breadth,'height =',height)

                    Big_Container[length:length+l,breadth:breadth+b,height:height+h] = box + 1

                    flag=1

                    break

            if flag == 1:

                break

        if flag == 1:

            break

    if(flag==0):

        print("unable to fit the small box",box,"in big box")
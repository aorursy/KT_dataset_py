# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#--------Warm up: Question 1----------------



# Gives substrings of s in decending order.

def substrings(s):



    # Declare local variable for the length of s.

    l = len(s)



    # Here I chose range over xrange for python version compatibility.

    for end in range(l, 0, -1):

        for i in range(l-end+1):

            yield s[i: i+end]



# Define palindrome.

def palindrome(s):

    return s == s[::-1]



# Main function.

def Question1(a):

    for l in substrings(a):

        if palindrome(l):

            return l



# Simple test case.

print (Question1("A rotating part of a mechanical device called rotor is made of steel stresseddesserts"))

# stresseddesserts
#--- Statistics: Question 1 ----------- 

import random

random.seed(3)

datalength=100

data = np.zeros(datalength)

#print(len(data))

for i in range(datalength-1):

    data[i]=random.randint(0,9)

    

my_histogram=np.zeros(len(bin_array)-1)

for i in range(len(bin_array)-1):

    #print(i)

    mask = (data>=bin_array[i])&(data<bin_array[i+1])

    #print(mask)

    my_histogram[i]=(len(data[mask]))



import matplotlib.pyplot as plt

%matplotlib inline

plt.plot ((bin_array[1:]+bin_array[:-1])/2.,my_histogram)
#----SQL Question 1



PowerConsumption = pd.DataFrame({ 'Device' : ['A', 'A', 'B', 'A', 'C', 'A', 'C', 'C', 'B', 'D', 'A', 'B', 'B', 'C'],

                                 'Power' : [2, 1, 3, 2, 5, 1, 1, 2, 4, 6, 2, 1, 1, 2]})

print(PowerConsumption.groupby(['Device']).sum().idxmax())
PowerConsumption.groupby(['Device']).sum()
import matplotlib.pyplot as plt

%matplotlib inline

plt.plot ((bin_array[1:]+bin_array[:-1])/2.,my_histogram)
my_histogram
data[mask]
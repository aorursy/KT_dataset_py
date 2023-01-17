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


def searchString(s):

    A=0

    C=0

    G=0

    T=0    

    for x in s:

        if x == "A":

            A += 1

        elif x == "C":

            C += 1

        elif x == "G":

            G += 1

        elif x == "T":

            T += 1

    myList = [A,C,G,T]

    print (myList)
# TEST NO1

s = "ATGCTTCAGAAAGGTCTTACG."

searchString(s)
def isIn(dictionary):

    for x in dictionary:

        if dictionary[x] in x:

            print ("State ", x, "contains", dictionary[x])

        else:

            print("No substring in", x)
# TEST NO2

myObj = {

    "ABUJA": "BUJ",

    "NASARAWA": "SARA",

    "LAGOS": "COTONOU",

    "BAUCHI": "JALINGO",

    "KATSINA": "TSI"

}



isIn(myObj)

def isInRec(arg, arg2, arg3):

    if type(arg) is tuple and len(arg) == 2 and type(arg2) is tuple and len(arg2) == 2 and type(arg3) is tuple and len(arg3) == 2:

        if arg3[0] >= arg[0] and arg3[0] <= arg2[0]:

            if arg3[1] >= arg[1] and arg3[1] <= arg2[1]:

                return True

            else:

                return False

        else:

            return False

    

    else:

        print("All co-ordinate arguments must be tuples of length 2")
# TEST NO3

isInRec((4,19), (13,27), (8,28))
def allIsInRec(arg, arg2, arg3):

    if type(arg) is tuple and len(arg) == 2 and type(arg2) is tuple and len(arg2) == 2 and type(arg3) is list and len(arg3) != 0:

        a = []

        countFalse = 0

        answer = True

        for x in arg3:

            if x[0] >= arg[0] and x[0] <= arg2[0]:

                a += ["True"]

            else:

                a += ["False"]

            if x[1] >= arg[1] and x[1] <= arg2[1]:

                a += ["True"]

            else:

                a += ["False"]

        for y in a:

            if y == "False":

                countFalse += 1

        if countFalse != 0:

            answer = False

        else:

            answer = True

        return answer

    else:

        return False
# TEST NO4

allIsInRec((4,19), (13,27), [(5,20), (8,25), (12,27)])
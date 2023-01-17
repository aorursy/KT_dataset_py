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
#Q1

def is_multiple(n, m):

    i = 4

    if n == m*i:

        return True

    elif ():

        return False



#Q1 Answer

def is_multiple(n, m):

    return n % m == 0

#Q2

x = [1]



for index in range(0,9):

    new_sum = x[index]*2

    x.append(new_sum)

print(x)
#Q2 Answer

[2 ** i for i in range(0,9)]
#Q3

def input_check():

    try:

        new_input = int(input("Please add new number: "))

        return new_input

    except ValueError:

        print("Input must be an integer")

        input_check()





def sequence_check(sequence):

    new_number=input_check()

    if sequence.count(new_number)<1:

        sequence.append(new_number)

        print(sequence)

        sequence_check(sequence)

    else:

        print("{} is repeated in this sequence".format(new_number))

        empty_sequence=[]

        sequence_check(empty_sequence)





sequence = []

sequence_check(sequence)
#Q3 Answer

def all_distinct(seq):

    for i in range( 0 , len(seq) - 1 ):

        for j in range(i + 1 , len(seq)):

            if seq[i] == seq[j]:

                return False

    return True
#Q4

def harmonic_list(n):

    result = []

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

        result.append(h)

    return result



print(harmonic_list(4))





def harmonic_gen(n):

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

    return h





print(harmonic_gen(4))
#Q4 Answer

def harmonic_gen(n):

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

        yield h
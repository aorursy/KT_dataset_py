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
# Practical 1





def main():

    # Task 1

    is_multiple(10, 5)



    # Task 2

    task2 = list_comprehension()



    # Task 3

    all_distinct(task2)



    # Task 4

    harmonic_gen(5)





def is_multiple(n, m):

    print('Is multiple:')

    if n % m == 0:

        print("true")

    else:

        print("false")





def list_comprehension():

    the_list = [2 ** i for i in range(0, 9)]

    print('List Comprehension:\n{}'.format(the_list))

    return the_list





def all_distinct(seq):

    for i in range(0, len(seq) - 1):

        for j in range(i + 1, len(seq)):

            if seq[i] == seq[j]:

                print('All distinct:\nFalse')

                return False

    print('All distinct:\nTrue')

    return True





def harmonic_gen(n):

    h = 0

    for i in range(1, n + 1):

        h += 1 / i

    yield h

    print('The first {} natural numbers: {}'.format(n, h))





main()

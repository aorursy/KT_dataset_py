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
def is_multiple(n,m):

    

    if n%m==0:

        return True

    else:

        return False

print(is_multiple(12,6)) 
num=1

s=[2**num for num in range(9)]

print(s)
def differentnums(list):

    if len(list)>len(set(list)):

        return False

    else:

        return True

print(differentnums([3,4,5,6,6,7,8,1,2]))
def harmonic_gen(n):

    h=0

    for i in range(1,n+1):

        h+=1/i

        yield h



for i in harmonic_gen(3):

    print(i)
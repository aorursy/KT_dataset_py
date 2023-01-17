# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import ggplot



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import random

from math import sqrt



def mean(data):

    return float(sum(data)/len(data))



def variance(data):

    mu = mean(data)

    return sum([(float(x)-mu)**2 for x in data])/len(data)



def stddev(data):

    return sqrt(variance(data))



print(stddev([1,2,3,4,5]))
def flip(count):

    l = []

    for i in range(count):

            l.append(random.random()>0.5)

            

    return l                  

N = 1000

f=flip(N)

print(mean(f))

print(stddev(f))
def sample(N):

    return [mean(flip(N)) for x in range(N)]
import matplotlib.pyplot as py

N = 1000

outcomes = sample(N)

py.hist(outcomes,bins=30)
### plot of m with differewnt k and p 

### k with m and n 

### will be done using 

### https://www.kaggle.com/romankovalenko/data-distribution-3d-scatter-plots
# lets see the space taken by a bloom filter compared to a hashmap or set as a nive example. 
# Awesome implementation from geeksandgeeks

# Python 3 program to build Bloom Filter 

# Install mmh3 and bitarray 3rd party module first 

# pip install mmh3 

# pip install bitarray 

import math 

import mmh3 

from bitarray import bitarray 

  

class BloomFilter(object): 

  

    ''' 

    Class for Bloom filter, using murmur3 hash function 

    '''

  

    def __init__(self, items_count,fp_prob): 

        ''' 

        items_count : int 

            Number of items expected to be stored in bloom filter 

        fp_prob : float 

            False Positive probability in decimal 

        '''

        # False posible probability in decimal 

        self.fp_prob = fp_prob 

  

        # Size of bit array to use 

        self.size = self.get_size(items_count,fp_prob) 

  

        # number of hash functions to use 

        self.hash_count = self.get_hash_count(self.size,items_count) 

  

        # Bit array of given size 

        self.bit_array = bitarray(self.size) 

  

        # initialize all bits as 0 

        self.bit_array.setall(0) 

  

    def add(self, item): 

        ''' 

        Add an item in the filter 

        '''

        digests = [] 

        for i in range(self.hash_count): 

  

            # create digest for given item. 

            # i work as seed to mmh3.hash() function 

            # With different seed, digest created is different 

            digest = mmh3.hash(item,i) % self.size 

            digests.append(digest) 

  

            # set the bit True in bit_array 

            self.bit_array[digest] = True

  

    def check(self, item): 

        ''' 

        Check for existence of an item in filter 

        '''

        for i in range(self.hash_count): 

            digest = mmh3.hash(item,i) % self.size 

            if self.bit_array[digest] == False: 

  

                # if any of bit is False then,its not present 

                # in filter 

                # else there is probability that it exist 

                return False

        return True

  

    @classmethod

    def get_size(self,n,p): 

        ''' 

        Return the size of bit array(m) to used using 

        following formula 

        m = -(n * lg(p)) / (lg(2)^2) 

        n : int 

            number of items expected to be stored in filter 

        p : float 

            False Positive probability in decimal 

        '''

        m = -(n * math.log(p))/(math.log(2)**2) 

        return int(m) 

  

    @classmethod

    def get_hash_count(self, m, n): 

        ''' 

        Return the hash function(k) to be used using 

        following formula 

        k = (m/n) * lg(2) 

  

        m : int 

            size of bit array 

        n : int 

            number of items expected to be stored in filter 

        '''

        k = (m/n) * math.log(2) 

        return int(k) 

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
a = set(['apple', 'boy', 'cat', 'dog'])

b = set(['apple', 'boy', 'cat', 'dog', 'cow'])
import sys
sys.getsizeof(a), sys.getsizeof(b)
n = 4 #no of items to add 

p = 0.01 #false positive probability 

  

bloomf = BloomFilter(n,p) 

print("Size of bit array:{}".format(bloomf.size)) 

print("False positive Probability:{}".format(bloomf.fp_prob)) 

print("Number of hash functions:{}".format(bloomf.hash_count)) 

!ls "../input"
mmh3.hash("bloom filter",1)
import numpy as np

import pandas as pd

import math

import matplotlib.pyplot as plt

from plotly import tools

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

from IPython.display import HTML, Image
ns = [np.power(10,i)  for i in range(5,13)]

powers = [-7, -6, -5.6, -5.2, -5, -4.8, -4.6, -4.4, -4.2, -4.1 -4.05, -4, -3.9, -3.8, -3.7, -3.6, -3.5, -3.4, -3.3, -3.2,-3.1, -3.06, -3.04, 3.02, -3.,   

         -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2,-2.1, -2.06, -2.04, 2.02, -2.,

          -1.9, -1.8, -1.7, -1.6, -1.5, -1.4, -1.3, -1.2,-1.1, -1.06, -1.04, -1.02, -1. ]

ps = [1/np.power(10,-p, dtype=np.float64) for p in powers]

ps = [p for p in ps if p< 1]

x = []

y = []

z = []

x1 = []

y1 = []

for ln_n in range(5,13):

    n = np.power(10,ln_n)

    for ln_p in powers:

        p = 1/np.power(10,-ln_p, dtype=np.float64)

        if p < 1:

            pass

        k = -(n * math.log(p))/(math.log(2)**2) 

        x.append(n)

        y.append(p)

        x1.append(ln_n)

        y1.append(ln_p)

        z.append(k)

        print("%10d %20.10f %20.10f" %(n,p,k))

len(x), len(y), len(ns), len(ps)
t = go.Scatter3d(

    x=x1,

    y=y1,

    z=z,

    mode='markers',

    marker=dict(

        size=4,

        line=dict(

            color='rgba(217, 217, 217, 0.14)',

            width=0.5

        ),

        opacity=1

    )

)

data = [t]

layout = go.Layout(

    margin=dict(

        l=0,

        r=0,

        b=0,

        t=0

    ),

    xaxis=dict(title="Set Size"),

    yaxis=dict(title="False Positive Rate"),

    title = "Blooms Filter Size"

)

fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='3d-scatter')
data = [

    go.Contour(

        x=x1,

        y=y1,

        z=z,

        colorscale='Jet',

    )

]



layout = go.Layout(

    title = "Distribution of Bug pokemon",

    width = 1000,

    height = 800

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename='bug-contour')
%matplotlib inline

import numpy as np

np.random.seed(123456789)



import matplotlib.pyplot as plt

import seaborn as sns



amount = 1000



def horse(): return max(0, np.random.normal(5,2,1)[0])

def ball(): return max(0, 1 + np.random.normal(1,0.3,1)[0])

def bike(): return max(0, np.random.normal(20,10,1)[0])

def train(): return max(0, np.random.normal(10,5,1)[0])

def coal(): return 47 * np.random.beta(0.5,0.5,1)[0]

def book(): return np.random.chisquare(2,1)[0]

def doll(): return np.random.gamma(5,1,1)[0]

def block(): return np.random.triangular(5,10,20,1)[0]

def gloves(): return  3.0 + np.random.rand(1)[0] if np.random.rand(1) < 0.3 else np.random.rand(1)[0]



item_list = [horse, ball, bike, train, coal, book, doll, block, gloves]

items_dist = [[i() for x in range(amount)] for i in item_list]



for idx, dist in enumerate(items_dist):

  plt.figure()

  plt.hist(dist, histtype='bar', bins=100)

  plt.suptitle(item_list[idx].__name__)

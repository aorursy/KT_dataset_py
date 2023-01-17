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
from __future__ import print_function, division

from builtins import range

# Note: you may need to update your version of future

# sudo pip install -U future





import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import beta





NUM_TRIALS = 2000

BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]





class Bandit(object):

  def __init__(self, p):

    self.p = p

    self.a = 1

    self.b = 1



  def pull(self):

    return np.random.random() < self.p



  def sample(self):

    return np.random.beta(self.a, self.b)



  def update(self, x):

    self.a += x

    self.b += 1 - x





def plot(bandits, trial):

  x = np.linspace(0, 1, 200)

  for b in bandits:

    y = beta.pdf(x, b.a, b.b)

    plt.plot(x, y, label="real p: %.4f" % b.p)

  plt.title("Bandit distributions after %s trials" % trial)

  plt.legend()

  plt.show()





def experiment():

  bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]



  sample_points = [5,10,20,50,100,200,500,1000,1500,1999]

  for i in range(NUM_TRIALS):



    # take a sample from each bandit

    bestb = None

    maxsample = -1

    allsamples = [] # let's collect these just to print for debugging

    for b in bandits:

      sample = b.sample()

      allsamples.append("%.4f" % sample)

      if sample > maxsample:

        maxsample = sample

        bestb = b

    if i in sample_points:

      print("current samples: %s" % allsamples)

      plot(bandits, i)



    # pull the arm for the bandit with the largest sample

    x = bestb.pull()



    # update the distribution for the bandit whose arm we just pulled

    bestb.update(x)





if __name__ == "__main__":

  experiment()

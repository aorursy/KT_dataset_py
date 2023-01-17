# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import random
def random_kid():

    return random.choice(["boy", "girl"])

both_girls = 0

older_girl = 0

either_girl = 0

random.seed(0)

for _ in range(10000):

    younger = random_kid()

    older = random_kid()

    if older == "girl":

        older_girl += 1

    if older == "girl" and younger == "girl":

         both_girls += 1

    if older == "girl" or younger == "girl":

         either_girl += 1

            

print("P(both | older):", both_girls / older_girl) # 0.514 ~ 1/2

print("P(both | either): ", both_girls / either_girl) # 0.342 ~ 1/3
def uniform_pdf(x):

    return 1 if x >= 0 and x < 1 else 0



def uniform_cdf(x):

    "returns the probability that a uniform random variable is <= x"

    if x < 0: return 0 # uniform random is never less than 0

    elif x < 1: return x # e.g. P(X <= 0.4) = 0.4

    else: return 1 # uniform random is always less than 1
for _ in np.arange(-1,5,0.1):

    print(uniform_pdf(_))
for _ in np.arange(-1,5,0.1):

    print(uniform_cdf(_))
import matplotlib.pyplot as plt
#Plotting Uniform PDF

xs = np.arange(-1,5,0.1)

plt.plot(xs,[uniform_pdf(x) for x in xs],'-',label='Uniform PDF')

plt.legend()

plt.title("Uniform Probability Distribution Function")

plt.show()
#Plotting Uniform CDF

xs = np.arange(-1,5,0.1)

plt.plot(xs,[uniform_cdf(x) for x in xs],'-',label='Uniform CDF')

plt.legend()

plt.title("Uniform Cumulative Distribution Function")

plt.show()
import math
def normal_pdf(x, mu=0, sigma=1):

    sqrt_two_pi = math.sqrt(2 * math.pi)

    return (math.exp(-(x-mu) ** 2 / 2 / sigma ** 2) / (sqrt_two_pi * sigma))

xs = [x / 10.0 for x in range(-50, 50)]

plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')

plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')

plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')

plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')

plt.legend()

plt.title("Various Normal pdfs")

plt.show()

def normal_cdf(x, mu=0,sigma=1):

    return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2
xs = [x / 10.0 for x in range(-50, 50)]

plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')

plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')

plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs],':',label='mu=0,sigma=0.5')

plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs],'-.',label='mu=-1,sigma=1')

plt.legend(loc=4) # bottom right

plt.title("Various Normal cdfs")

plt.show()

#Using Python 3.8 Features xD

def bernoulli_trial(p: float) -> int:

    """Returns 1 with probability p and 0 with probability 1-p"""

    return 1 if random.random() < p else 0



def binomial(n: int, p: float) -> int:

    """Returns the sum of n bernoulli(p) trials"""

    return sum(bernoulli_trial(p) for _ in range(n))
from collections import Counter
def binomial_histogram(p: float, n: int, num_points: int) -> None:

    """Picks points from a Binomial(n, p) and plots their histogram"""

    data = [binomial(n, p) for _ in range(num_points)]



    # use a bar chart to show the actual binomial samples

    histogram = Counter(data)

    plt.bar([x - 0.4 for x in histogram.keys()],

            [v / num_points for v in histogram.values()],

            0.8,

            color='0.75')



    mu = p * n

    sigma = math.sqrt(n * p * (1 - p))



    # use a line chart to show the normal approximation

    xs = range(min(data), max(data) + 1)

    ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)

          for i in xs]

    plt.plot(xs,ys)

    plt.title("Binomial Distribution vs. Normal Approximation")
binomial_histogram(0.75,100,10000)
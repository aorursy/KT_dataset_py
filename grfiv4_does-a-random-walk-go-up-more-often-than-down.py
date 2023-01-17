import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



d = pd.read_hdf('../input/train.h5')



a=plt.hist(d.y, bins=100)

plt.title("Two Sigma y's are symmetrical (looking) with spikey tails")

plt.show()
a=plt.hist(d.y, bins=8)

plt.title("The Two Sigma y-duck looks to the right")

plt.show()
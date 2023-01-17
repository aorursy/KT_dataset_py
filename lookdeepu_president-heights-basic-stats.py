from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
import pandas as pd

import numpy as np

import re

import sklearn

import xgboost as xgb

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

train = pd.read_csv('../input/president_heights_new.csv')
train.head()
heights = np.array(train['height(cm)'])
print(heights)
print("mean height:",heights.mean())

print("standard deviation:",heights.std())

print("Minimum height: ",heights.min())

print("Maximum height: ",heights.max())
print("25th percentile:",np.percentile(heights, 25))

print("median: ", np.median(heights))

print("75th percentile:",np.percentile(heights, 75))
import matplotlib.pyplot as plt

import seaborn; seaborn.set()
plt.hist(heights)

plt.title("Height dist of US presidents")

plt.xlabel('height(cm)')

plt.ylabel('number');

          
%pylab inline
# in Xkcd format of US presidential heights

plt.xkcd()

plt.hist(heights)

plt.title("Height dist of US presidents")

plt.xlabel('height(cm)')

plt.ylabel('number');
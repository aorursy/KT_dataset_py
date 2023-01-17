# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
ufo = pd.read_csv("../input/scrubbed.csv")

ufo.head()
countryFreq = ufo["country"].value_counts()



labels = list(countryFreq.index)

positionsForBars = list(range(len(labels)))



plt.bar(positionsForBars, countryFreq.values) # plot our bars

plt.xticks(positionsForBars, labels) # add lables

plt.title("countries found ufo")
stateFreq = ufo["state"][(ufo.country == 'us')].value_counts()



labels = list(stateFreq.index)

positionsForBars = list(range(len(labels)))



plt.figure(figsize=(18,8))

plt.bar(positionsForBars, stateFreq.values) # plot our bars

plt.xticks(positionsForBars, labels) # add lables

plt.title("state found ufo")
sns.countplot(ufo["country"]).set_title("Country found ufo")
plt.figure(figsize=(18,8))

ax = sns.countplot(ufo["state"][(ufo.country == 'us')]).set_title("State found ufo")
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# Load in data

boulder = pd.read_csv('../input/ifsc-sport-climbing-competition-results/boulder_results.csv')

lead = pd.read_csv('../input/ifsc-sport-climbing-competition-results/lead_results.csv')

speed = pd.read_csv('../input/ifsc-sport-climbing-competition-results/lead_results.csv')

combined = pd.read_csv('../input/ifsc-sport-climbing-competition-results/combined_results.csv')
avg_rank = boulder.groupby(['FIRST', 'LAST'])['Rank'].agg(['mean', 'count']).reset_index().sort_values('mean')

avg_rank
avg_rank = avg_rank[avg_rank['count'] > 5]
avg_rank = avg_rank[0:10]

names = avg_rank['FIRST'] +  ' ' + avg_rank['LAST'].str.lower().str.capitalize()

firsts = []

seconds = []

thirds = []

for _, climber in avg_rank.iterrows():

    placings = boulder[(boulder['FIRST'] == climber['FIRST']) & (boulder['LAST'] == climber['LAST'])]['Rank']

    firsts.append(len(placings[placings == 1]))

    seconds.append(len(placings[placings == 2]))

    thirds.append(len(placings[placings == 3]))
ax = plt.subplot(111)

X = np.arange(len(names))

wid = 0.2

ax.bar(X, firsts, width=wid, color='y')

ax.bar(X-wid, seconds, width=wid, color='silver')

ax.bar(X+wid, thirds, width=wid, color='brown')

plt.xticks(X, names, rotation='vertical')

plt.show()
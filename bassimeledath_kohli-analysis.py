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
# dependencies



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# read csv



df = pd.read_csv("../input/kohli_stats.csv")
# view df



df['Runs'] = [int(x) for x in df['Runs'].str.replace(',','')]

df['Balls'] = [int(x) for x in df['Balls'].str.replace(',','')]

df
df['avg_runs_per_innings'] = df['Runs']/df['Innings']

mean = np.mean(df['avg_runs_per_innings'])

median = np.median(df['avg_runs_per_innings'])

plt.figure(figsize=(8,10))

sns.distplot(df['avg_runs_per_innings'],bins=6,rug=True)

plt.axvline(x=mean,label='mean = {}'.format(round(mean,2)),c='red')

plt.axvline(x=median,label='median = {}'.format(round(median,2)),c='green')

plt.title("Virat Kohli's Average Run Per Innings Over 13 years")

plt.legend()
# Other descriptive statistics



df['avg_runs_per_innings'].describe()
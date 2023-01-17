# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/covid19-sao-paulo-state/covid-19 sp.csv')

df = df.fillna(0)

df.head()
fig, ax = plt.subplots(2, 1, figsize=(20, 10))

fig.subplots_adjust(hspace = .5)

ax[0].plot(df['total de casos'])

ax[0].set_xlabel('Time')

ax[0].set_ylabel('Cases')

ax[0].set_title('Cases Over Time')



ax[1].plot(df['obitos por dia'], color='r')

ax[1].set_xlabel('Time')

ax[1].set_ylabel('Deaths')

ax[1].set_title('Deaths Over Time')
plt.subplots(figsize=(20, 10))

plt.bar(np.arange(len(df['casos por dia'])), height=df['casos por dia'])

plt.xlabel('Time', size=16)

plt.ylabel('Deaths', size=16)

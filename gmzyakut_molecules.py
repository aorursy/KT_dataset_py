# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

%matplotlib inline



import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go



# Any results you write to the current directory are saved as output.
struct = pd.read_csv("../input/structures.csv")

struct.head(10)
len(atoms)

print(atoms[0])

print(atoms)

print(struct.loc[struct.atom==atoms[n]].x.values[0:M])
M = 10000

fig, ax = plt.subplots(1,3,figsize=(20,5))



colors = ["blue", "red", "green", "yellow", "purple"]

atoms = struct.atom.unique()



for n in range(len(atoms)):



    ax[0].scatter(struct.loc[struct.atom==atoms[n]].x.values[0:M],

                  struct.loc[struct.atom==atoms[n]].y.values[0:M],

                  color=colors[n], s=2, alpha=0.5, label=atoms[n])

    ax[0].legend()

    ax[0].set_xlabel("x")

    ax[0].set_ylabel("y")

    

    ax[1].scatter(struct.loc[struct.atom==atoms[n]].x.values[0:M],

                  struct.loc[struct.atom==atoms[n]].z.values[0:M],

                  color=colors[n], s=2, alpha=0.5, label=atoms[n])

    ax[1].legend()

    ax[1].set_xlabel("x")

    ax[1].set_ylabel("z")

    

    ax[2].scatter(struct.loc[struct.atom==atoms[n]].y.values[0:M],

                  struct.loc[struct.atom==atoms[n]].z.values[0:M],

                  color=colors[n], s=2, alpha=0.5, label=atoms[n])

    ax[2].legend()

    ax[2].set_xlabel("y")

    ax[2].set_ylabel("z")
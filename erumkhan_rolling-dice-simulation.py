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
means = []

for i in range(1,1000):

    rolls = np.random.choice(a=[1,2,3,4,5,6], size=i, replace=True, p=[1/6,1/6,1/6,1/6,1/6,1/6])

    means.append(rolls.mean())

fig, ax = plt.subplots(figsize=(12,8))

ax.set_ylim((1,6))

ax.plot(means)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as plt

import seaborn as sns #visualization tool

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

combats = pd.read_csv("../input/pokemon-challenge/combats.csv")

data = pd.read_csv("../input/pokemon-challenge/pokemon.csv")

tests = pd.read_csv("../input/pokemon-challenge/tests.csv")
data.info()
data.corr()
#corelation map

f,ax = plt.pyplot.subplots(figsize=(16,16))

sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt=".1f", ax=ax)

plt.pyplot.show()
data.head(15)
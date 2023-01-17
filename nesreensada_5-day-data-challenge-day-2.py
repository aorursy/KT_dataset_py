# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# read in our data
data = pd.read_csv("../input/database.csv")
data.describe()
#data.head()

# list column names
print(data.columns)

# get Engine1 Damage
eng1_damage = data['Windshield Damage']
print(eng1_damage.head())
plt.hist(eng1_damage)
plt.title("Engine 1 damange in aircrafts")

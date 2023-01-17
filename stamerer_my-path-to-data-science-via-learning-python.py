# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



print("Welcome")

# Any results you write to the current directory are saved as output.
data = pd.read_csv("../input/data.csv")

print("Lets start")
data.info()
data.describe()
data.corr()
data.head()
data.columns
data.plot(kind='scatter', x='Agility', y='Finishing',alpha = 0.5, color = 'green')

plt.xlabel('Agility')        

plt.ylabel('Finishing')

plt.title('Agility - GOAL Scatter Plot')  
x = data[(data["Finishing"] >85) & (data["Penalties"] > 85)]

print(x.Name)
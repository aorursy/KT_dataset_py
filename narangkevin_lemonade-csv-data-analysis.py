# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv ('../input/Lemonade.csv', header=0)

print (df)

df.columns
mean1 = df['Sales'].mean()

min1 = df['Sales'].min()

max1 = df['Sales'].max()

print('Mean: ', mean1)

print('Min: ', min1)

print('Max: ', max1)
#Plotting Daily Sales



df['Sales'].value_counts().sort_index().plot.line()
#Plotting ScatterPlot 

import seaborn as sns

sns.scatterplot(x='Flyers', 

           y='Sales',

          data = df)
import matplotlib.pyplot as plt

label = df['Day']

mean = df['Sales'].mean(axis=1)

plt.pie(mean, labels=label)

plt.title("Average Sales per Day")

plt.show()
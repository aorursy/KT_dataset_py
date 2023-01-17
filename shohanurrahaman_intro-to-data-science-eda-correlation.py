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
df = pd.read_csv('/kaggle/input/default-of-credit-card-clients-dataset/UCI_Credit_Card.csv')

correlation = df.corr()

print(correlation)
import matplotlib.pyplot as plt 

import seaborn as sns





# correlation table

corr = df.corr()

#plot heatmap 

#vmin = minimum number of color

#vmax = maximum number of color 

plt.figure(figsize=(10,10))

sns.heatmap(data=corr, vmin=-1, vmax=1, cmap='coolwarm')
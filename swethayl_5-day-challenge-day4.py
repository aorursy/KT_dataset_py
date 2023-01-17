# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
# read in our data
cereals = pd.read_csv("../input/cereal.csv")

# Any results you write to the current directory are saved as output.
# What are the different manufacturers present in the data?
Manufacturer = cereals.mfr.unique()
mfr_count = []
for mnfc in Manufacturer:
    mfr_count.append(cereals[cereals['mfr'] == mnfc]['mfr'].count())
# Copy the manufacturer names from the data info
# A = American Home Food Products;
# G = General Mills
# K = Kelloggs
# N = Nabisco
# P = Post
# Q = Quaker Oats
# R = Ralston Purina
import seaborn as sns
sns.countplot(cereals.mfr).set_title("Cereal counts by Manufacturer")

x = np.arange(len(Manufacturer))
plt.figure(figsize=(10,8))
plt.bar(x, mfr_count, align='center', alpha=0.5, edgecolor = 'orange')
plt.xticks(x, Manufacturer, fontsize=18 )
plt.ylabel('Number of cereals a manufacturer produces', fontsize=18 )
plt.title('Manufacturer of cereals', fontsize=18)
plt.show()
explode = (0, 0, 0.1, 0, 0, 0, 0)  # explode 1st slice
plt.figure(figsize=(8,8))
plt.rcParams['font.size'] = 16
plt.pie(mfr_count, 
        explode=explode, 
        labels=Manufacturer, 
        autopct='%1.1f%%', 
        shadow=True, 
        startangle = 140 )
 
plt.axis('equal')
plt.tight_layout()
plt.show()
           
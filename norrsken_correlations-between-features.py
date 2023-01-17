# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/200-financial-indicators-of-us-stocks-20142018/2018_Financial_Data.csv')
plt.figure(figsize = (20,10))

plot = sns.countplot(df['Sector'])



plot.set_xticklabels(plot.get_xticklabels(), rotation=30)

df.Sector.value_counts()
corr = df.corr().round(2)



# Hide the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True



# Set figure size

f, ax = plt.subplots(figsize=(200, 200))



# Drawing the heatmap

plot = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)



plt.tight_layout()
plot.figure.savefig("output.png")
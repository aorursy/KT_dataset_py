# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # for visualization

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/heart-disease-uci/heart.csv')

df.describe()
f, ax = plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(), annot=True, lw=.5, fmt=".1f", ax=ax)

plt.show()
df.head()
df.columns
df.chol.plot(kind='line', c='purple', lw=.8, label="serum cholestoral", alpha=.7, ls="--", figsize=(15,5))

df.thalach.plot(kind='line', c='b', lw=.8, label="max heart rate achieved", alpha=.7, ls="-")

plt.xlabel("id")

plt.ylabel("mg/dl")

plt.title("Line Plots")

plt.legend()

plt.show()
df.plot(kind='scatter', x="thalach", y="age", c='r', lw=3, edgecolor="b", s=100, alpha=.7, figsize=(15,5))

plt.title('Scatter Plot',size=25, loc='left', alpha=.25)

plt.ylabel('Age', size=15, alpha=.7, color="g")              # label = name of label

plt.xlabel('Max Heart Rate',size=15, alpha=.7, color="g")

plt.gca().invert_yaxis()
df.columns
filter1 = df['age'] < 35

df[filter1]
filter2 = df['thalach'] > 192

df[filter2]
df[filter1 & filter2]
df[filter1 | filter2]
for index, value in df[['age']][0:3].iterrows():

    print(index, "::", value.age)



### or
for index, value in enumerate(df['age'][0:3]):

    print(index, "::", value)
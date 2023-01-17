# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
df.head(100)
df.columns
df.info()
df.corr()
f,xx=plt.subplots(figsize=(10,10))

sns.heatmap(df.corr(), annot=True, linewidth=.5, fmt='.2f', ax=xx)



plt.show()
correlated =[]

corr = df.corr()

for i in df:

    for j in df:

        if corr[i][j]>0.40 and i!=j:

            correlated.append(i)

            correlated.append(j)

correlated = list(set(correlated))
print('correlated columns:{}'.format(correlated))
plt.figure(figsize=(5,5))

sns.scatterplot(x = df['cp'],y = df['thalach'],hue =df['sex'])

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from PIL import Image

import matplotlib.pyplot as plt

im = Image.open("/kaggle/input/structure-image/structure.png")

plt.imshow(im);
df = pd.read_csv("/kaggle/input/cee-498-project9-structural-damage-detection/train.csv")

df.head()
df.dtypes
df.shape
df.drop(['Condition'], axis=1, inplace=False).hist(figsize=(20,20));



df_damaged = df[df.Condition == 1]

var1 = df_damaged.drop(['Condition'], axis=1, inplace=False).var(axis=1)

var1 = var1.reset_index(drop=True)

plt.plot(var1.index/200,var1.values)

plt.xlabel('Time(sec)');

plt.ylabel("Acceleration(g)");
df_undamaged = df[df.Condition == 0]

var2 = df_undamaged.drop(['Condition'], axis=1, inplace=False).var(axis=1)

var2 = var2.to_frame()

var2 = var2.dropna()

var2 = var2.reset_index(drop=True)

plt.plot(var2.index/200,var2.values)

plt.xlabel('Time(sec)');

plt.ylabel("Acceleration(g)");
sq = df.drop(['Condition'], axis=1, inplace=False)**2

mean = sq.mean()

np.sqrt(mean)
df_damaged.drop(['Condition'], axis=1, inplace=False).plot.kde();
df_undamaged = df_undamaged.reset_index(drop=True)

df_undamaged.drop(['Condition'], axis=1, inplace=False).plot.kde();
import seaborn as sns

sns.set()

corr_matrix = np.triu(df.drop(['Condition'], axis=1, inplace=False).corr())

sns.heatmap(df.drop(['Condition'], axis=1, inplace=False).corr(),annot = True,annot_kws = {"size": 8},fmt='.1g',mask=corr_matrix);
df_damaged.drop(['Condition'], axis=1, inplace=False).skew(axis=0,skipna=True)
df_undamaged.drop(['Condition'], axis=1, inplace=False).skew(axis=0,skipna=True)
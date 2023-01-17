# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/iris/Iris.csv')

df.head()
df.drop('Id',axis=1,inplace=True)
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=df, height=5)
sns.scatterplot(x="SepalLengthCm", y="SepalWidthCm", data=df,hue='Species')

sns.despine()
sns.boxplot(x="Species", y="PetalLengthCm", data=df)
sns.stripplot(x="Species", y="SepalLengthCm", data=df, jitter=True)
sns.boxplot(x="Species", y="SepalLengthCm", data=df)

sns.stripplot(x="Species", y="SepalLengthCm", data=df, jitter=True)
sns.violinplot(x="Species", y="SepalLengthCm", data=df)
sns.pairplot(df)
sns.pairplot(df,hue='Species',diag_kind='hist')
sns.heatmap(df.corr(),annot=True,cmap='terrain')
df.hist(edgecolor='black', linewidth=1.2)

plt.gcf().set_size_inches(12,6)

plt.show()
plt.gcf().set_size_inches(10,7)

sns.swarmplot(x="Species", y="PetalLengthCm", data=df)
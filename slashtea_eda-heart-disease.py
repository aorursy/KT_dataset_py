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
data = pd.read_csv('../input/heart.csv')

data.head()
y = data['target']

data.drop('target', axis=1, inplace=True)
y.head()
data.isnull().sum()
data.describe()
data.shape
import seaborn as sns

from matplotlib.pyplot import rcParams

import matplotlib.pyplot as plt





sns.set()

plt.figure(figsize=(10, 8))

plt.title("Distribution of age between male and female")

sns.boxplot(data['sex'].map(lambda x: "female" if x==0 else "male"), data['age'])
data.groupby(data['age'])['trestbps'].mean().sort_values(ascending=False).head()
data.groupby(data['age'])['cp'].mean().sort_values(ascending=False).head()
resting_blood_pressure = data.groupby(['sex', 'age'])['trestbps'].mean().sort_values(ascending=False)

resting_blood_pressure.head(20)
import seaborn as sns

from matplotlib.pyplot import rcParams

import matplotlib.pyplot as plt





plt.rcParams['figure.figsize'] = [45, 15]

sns.set()

sns.factorplot(x="age", y="trestbps", hue="sex", data=data,

                   size=26, kind="bar", palette="muted")



# sns.catplot(x="age", y="trestbps", hue="sex", kind="swarm", data=data)
sns.violinplot(x="age", y="cp", hue="sex", data=data, split=True,

               inner="quart", palette="Set2")
sns.violinplot(x="age", y="trestbps", hue="sex", data=data, split=True,

               inner="quart", palette="Set2")
data.groupby(['sex', 'age'])['cp'].mean().sort_values(ascending=False).head(20)
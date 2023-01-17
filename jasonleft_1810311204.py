# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from math import ceil
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.model_selection import validation_curve, learning_curve
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
test = pd.read_csv('/kaggle/input/titanic/test.csv', dtype={"Age": np.float64})
test.info()
train = pd.read_csv('/kaggle/input/titanic/train.csv', dtype={"Age": np.float64})
train.info()
train.sample(10)
all_data = pd.concat([train, test],ignore_index=True)
all_data
all_data['Surname']=all_data['Name'].apply(lambda x: x.split(',')[0].strip())
train=all_data[all_data['Surname'].notnull()]
train.sample(20)
pal = sns.cubehelix_palette(n_colors=3, rot=-.5, dark=.3)
sns.violinplot(x='Pclass', y='Survived', hue='Sex', data=train, split=True, palette=pal, bw=.2, cut=1, linewidth=1)
plt.show()
sns.violinplot(x='Embarked', y='Survived', hue='Sex', data=train, split=True, palette=pal, bw=.2, cut=1, linewidth=1)
plt.show()

g = sns.lmplot(x="Age", y="Survived", col="Sex", hue="Sex", data=train[train["Age"].notnull()], palette="Set1",
               y_jitter=.02, logistic=True, truncate=False)
g.set(xlim=(0, 80), ylim=(-.05, 1.05))
plt.show()


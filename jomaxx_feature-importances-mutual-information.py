# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_selection import mutual_info_classif, f_classif



print(os.listdir("../input"))



%matplotlib inline

# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")
train.head()
feature_columns = train.columns[2:]

target_column = train.columns[1:2]
X = train[feature_columns]

y = train[target_column]
plt.figure(figsize=(16,8))

ax = sns.countplot(x='target', data=train)
mic = mutual_info_classif(X.values, y.values.ravel())
feature_importances = {}

for i,f in enumerate(feature_columns):

    feature_importances[f] = mic[i] 
sorted(feature_importances.items(), key=lambda kv: kv[1], reverse=True)
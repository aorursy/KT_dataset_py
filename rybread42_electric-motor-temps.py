# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



dataset = []



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        dataset = pd.read_csv(os.path.join(dirname, filename))



X = dataset.drop(['stator_tooth'], axis=1)

y = dataset.drop(['stator_winding'], axis=1)
sns.scatterplot(x='stator_tooth', y='stator_winding', hue='stator_tooth', palette='RdBu', alpha=0.8, data=dataset[['stator_tooth', 'stator_winding']])
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



#X_train.reshape(-1, 1)

#y_train.reshape(-1, 1)



reg = LinearRegression()

reg.fit(X_train, y_train)



print(r2_score(X_test, y_test))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
# Any results you write to the current directory are saved as output.
X = pd.read_csv("../input/all_train.csv")
y = X["has_parkinson"]
X.drop(["has_parkinson","id"], axis=1,inplace=True)
skb = SelectKBest(chi2, k=350)
X_new = skb.fit_transform(X, y)
print(X_new.shape)
mask = skb.get_support() #list of booleans
new_features = [] # The list of your K best features
feature_names = list(X.columns.values) #all features
for bool, feature in zip(mask, feature_names):
    if bool:
        new_features.append(feature)
print(new_features)

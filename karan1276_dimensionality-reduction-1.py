# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
from __future__ import print_function, division

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import Binarizer
# Any results you write to the current directory are saved as output.
X = pd.read_csv("../input/all_train.csv")
y = X["has_parkinson"]
X.drop(["has_parkinson","id"], axis=1,inplace=True)
#score before k-best
clf = ExtraTreesClassifier()
clf = clf.fit(X, y)
features_imp = clf.feature_importances_
features_imp = np.sort(features_imp)[::-1]
#Sum the imporantance with previous values
features_imp_sum = []
features_imp_sum.append(features_imp[0])
for i in range(1,len(features_imp)-1):
    features_imp_sum.append(features_imp_sum[i-1]+features_imp[i])
plt.plot(features_imp_sum)
plt.grid()
plt.plot(features_imp_sum[0:350])

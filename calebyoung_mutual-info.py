# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#import data
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

# map diagnosis
diagnosis_data = data['diagnosis'].map({'M':1,'B':0})

#convert into numpy array
diagnosis_data = diagnosis_data.values

#convert into numpy array, reshape to 2D array
id_data = data['id'].values.reshape(-1,1)
# Change discrete_freatures = "True" to discrete_features = "False" and get a different value
mutual_info = mutual_info_classif(id_data,diagnosis_data,discrete_features=True)
#sklearn.metrics.mutual_info_score(labels_true, labels_pred, contingency=None)

print(mutual_info)
# Sandbox Example (change the numbers)
from sklearn.feature_selection import mutual_info_classif
X = np.array([[1, 2, 0],
              [2, 4, 1],
              [0, 0, 0],
              [8, 16, 0],
              [16, 32, 0]])
y = np.array([0, 1,2, 3, 4])
mutual_info_classif(X, y, discrete_features=True)
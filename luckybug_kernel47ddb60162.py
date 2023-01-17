# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/iris/Iris.csv')
data['PetalArea'] = data['PetalLengthCm'] * data['PetalWidthCm']
data['SepalArea'] = data['SepalLengthCm'] * data['SepalWidthCm']
data['SepalAreaToPetalArea'] = data['SepalArea'] / data['PetalArea']
data['SepalLengthToWidth'] =  data['SepalLengthCm'] / data['SepalWidthCm']
data['PetalLengthToWidth'] =  data['PetalLengthCm'] / data['PetalWidthCm']
data.head()
all_features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", 'PetalArea', 'SepalArea', 'SepalAreaToPetalArea', 'SepalLengthToWidth', 'PetalLengthToWidth']
pairs = [(all_features[i], all_features[j]) for i in range(len(all_features)-1) for j in range(i+1, len(all_features))]
pairs
species = ('Iris-setosa','Iris-versicolor','Iris-virginica')
colors = ('blue', 'green','red')
dt = []
for specie in species:
    dt.append(data[data['Species'] == specie])

for f1, f2 in pairs:
    plt.scatter(dt[0][[f1]], dt[0][[f2]], color=colors[0])
    plt.scatter(dt[1][[f1]], dt[1][[f2]], color=colors[1])
    plt.scatter(dt[2][[f1]], dt[2][[f2]], color=colors[2])
    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.show()
best_features = ['PetalWidthCm', 'PetalLengthCm']
y = data["Species"].to_numpy()
clf = RandomForestClassifier(n_estimators=10, min_samples_split=5, min_samples_leaf=1, max_depth=100, bootstrap=False, n_jobs=-1, random_state=12)

for use_features in (all_features, best_features):
    print('Using features', use_features)
    X = data[use_features].to_numpy()
    
    start = time.time()
    scores = cross_val_score(clf, X, y, cv=10, scoring='accuracy')
    end = time.time()
    mean, std = scores.mean(), scores.std()
    print(f'Accuracy: {round(mean*100, 2)}% (+/- {round(std*100, 2)})%')
    print(f'Time: {round(end - start, 4)}s')
    print()
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
filepath =[]
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        filepath.append(os.path.join(dirname, filename))
        print(filepath)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = filepath[1]
test = filepath[0]
trainData = pd.read_csv(train)
testData = pd.read_csv(test)

cdf = trainData[['battery_power','clock_speed','dual_sim','four_g','int_memory','n_cores','ram','touch_screen','wifi','price_range']]
# cdf.head()
# trainData.head()
train_X = cdf[['battery_power','clock_speed','dual_sim','four_g','int_memory','n_cores','ram','touch_screen','wifi']].values
train_y = cdf[['price_range']].values

test_X = testData[['battery_power','clock_speed','dual_sim','four_g','int_memory','n_cores','ram','touch_screen','wifi']].values

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# X_trainset, X_testset, y_trainset, y_testset = train_test_split(train_X, train_y, test_size=0.3, random_state=3)

# drugTree = DecisionTreeClassifier(criterion="entropy", random_state = 0, max_depth=6)
# drugTree.fit(X_trainset,y_trainset)

# yPred = drugTree.predict(X_testset)
# accuracy_score(yPred,y_testset)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion="entropy", random_state = 0, max_depth=8)
clf.fit(train_X,train_y)

yPred =  clf.predict(test_X)
import graphviz 
from sklearn import tree
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris")
features = ['battery_power','clock_speed','dual_sim','four_g','int_memory','n_cores','ram','touch_screen','wifi']
target = ['price_range']
dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=features,
                                class_names=target[0],
                                filled=True, rounded=True, 
                                special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

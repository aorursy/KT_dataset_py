# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
df=pd.read_csv("/kaggle/input/winedata/winequality_red.csv")
df.head()
df.groupby('quality').mean()
df.boxplot(column='fixed acidity', by='quality', grid=False)
df.boxplot(column='volatile acidity', by='quality', grid=False)
y = df.iloc[:,11]
x = df.iloc[:,0:11]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
from sklearn.feature_selection import SelectKBest, f_classif
selector = SelectKBest(f_classif, k=4)
selector.fit(x_train, y_train)
cols = selector.get_support(indices=True)
cols
x_train_s = x_train.iloc[:,cols]
x_test_s = x_test.iloc[:,cols]
from sklearn.tree import DecisionTreeClassifier
#Initalize the classifier
clf = DecisionTreeClassifier()
#Fitting the training data
clf.fit(x_train, y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test)
# Model Accuracy, how often is the classifier correct?
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Fitting the training data
clf.fit(x_train_s, y_train)
#Predict the response for test dataset
y_pred = clf.predict(x_test_s)
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
#Accuracy
print('Accuracy = ', knn.score(x_test_s, y_test))
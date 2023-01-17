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
import numpy as np
from collections import Counter
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X,y=iris.data,iris.target
X_train,X_test,y_train,y_test = train_test_split(X,y)
def euclidean_distance(x1,x2):
    return np.sqrt(np.sum(x1-x2)**2)

class myKNN():
    def __init__(self,k=3):
        self.k=k
    
    def fit(self,X,y):
        self.Xt=X
        self.yt=y
    def predict(self,X):
        predicted_labels = [self._predict(x) for x in X]
        return predicted_labels
    def _predict(self,x):
        distances = [euclidean_distance(x,xt) for xt in self.Xt]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest = [self.yt[i] for i in k_indices]
        most_common = Counter(k_nearest).most_common(1)
        return most_common[0][0]
knn=myKNN(k=3)
knn.fit(X_train,y_train)
preds=knn.predict(X_test)
y_test
from sklearn.metrics import accuracy_score
accuracy_score(preds,y_test)

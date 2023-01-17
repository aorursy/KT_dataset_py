# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.plotting import plot_decision_regions
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#load data and map species to numeric
iris_data = pd.read_csv('/kaggle/input/iris/Iris.csv')
X = iris_data[['PetalLengthCm','SepalLengthCm','PetalWidthCm','SepalWidthCm']]
y = iris_data['Species'].map({'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3})
from sklearn.model_selection import ____
# use train_test_split to split the data - make sure to keep the seed the same to have reproducable results
X_train, X_test, y_train, y_test = __________(X, y, train_size = 0.8, random_state = 64)
#quick fit
from sklearn.neighbors import KNeighborsClassifier

knn = ____(n_neighbors=3)

knn.fit(X_train,y_train) 

from sklearn.metrics import accuracy_score, precision_score, recall_score
# calculate the prediction for each of the test set observations
____ = knn.predict(____)
# Because the problem is not binary we need to average the precision somehow - try googling it 
print(accuracy_score(____, y_test))
print(precision_score(____, y_test, average = ____))
print(recall_score(____, y_test, average = ____))

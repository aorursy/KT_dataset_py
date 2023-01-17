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
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
iris = pd.read_csv("/kaggle/input/iris/Iris.csv")
iris.head()
iris.Species.value_counts()
iris.isnull().sum()
def species_class(species):
    if species == 'Iris-setosa':
        return 0
    elif species == 'Iris-virginica':
        return 1
    else:
        return 2
iris['Species_class'] = iris.Species.map(species_class)
iris.head()
iris.Species_class.value_counts()
X = iris[['SepalLengthCm' , 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris.Species_class
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
knn.score(X_train, y_train)
accuracy_score(y_test, y_predict)
Species_name = ['Iris-setosa', 'Iris-virginica', 'Iris-versicolor']
Species_name[y_predict[1]], Species_name[y_test.iloc[1]]



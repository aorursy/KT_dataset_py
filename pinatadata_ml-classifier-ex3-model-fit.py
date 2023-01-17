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
#quick fit
from sklearn.neighbors import KNeighborsClassifier

knn = __________________(n_neighbors=3)

knn.fit(X,y) 

# Use the function plot_decision_regions from mlxtend.plotting to visualise the decision boundary 
# Experiment with the features selected and the value of k

X = iris_data[['PetalLengthCm','SepalLengthCm']]
y = iris_data['Species'].map({'Iris-setosa':1, 'Iris-versicolor':2, 'Iris-virginica':3})

knn = _________________(n_neighbors=_)
____________________(np.array(X), np.array(y), clf=knn, legend=3)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('knn on Iris with 2 features')
plt.show()

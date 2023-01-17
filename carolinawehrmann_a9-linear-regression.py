# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import datasets
from sklearn.svm import LinearSVR
from sklearn.metrics import confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

unknown_iris = [[4.8,2.5,5.3,2.4]]

iris_set = datasets.load_iris()

iris_data = iris_set.data
iris_labels = iris_set.target

np.random.seed(42)
indices = np.random.permutation(len(iris_data))
training_samples = 50

X_train = iris_data[indices[:-training_samples]]
y_train = iris_labels[indices[:-training_samples]]

X_test = iris_data[indices[-training_samples:]]
y_test = iris_labels[indices[-training_samples:]]
svr = make_pipeline(StandardScaler(), LinearSVR(random_state=42))
svr.fit(X_train, y_train)
y_pred = svc.predict(X_test)

iris_predict = svr.predict(unknown_iris)

iris_predict = int(round(iris_predict[0]))
confidence = svr.score(X_test, y_test)
 

print("Confidence percentage {}\n".format(confidence * 100))
print("the flower is iris {}".format(iris_set.target_names[iris_predict]))
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import pyplot as plt
from wordcloud import WordCloud
from sklearn.metrics import accuracy_score, log_loss
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/iris/Iris.csv')
print("Colums: ", data.columns.values)
print("Shape: ", data.shape)
print("Missing values:")
print(data.isnull().sum())
from sklearn.preprocessing import StandardScaler

X = data.drop(['Id', 'Species'], axis=1)
y = data['Species']
sc = StandardScaler()
x_train = sc.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion='entropy',
                             n_estimators=700,
                             min_samples_split=5,
                             min_samples_leaf=1,
                             max_features = "auto",
                             oob_score=True,
                             random_state=0,
                             n_jobs=-1)

clf.fit(X_train, y_train)
print("RF Accuracy: " + repr(round(clf.score(X_test, y_test) * 100, 2)) + "%")

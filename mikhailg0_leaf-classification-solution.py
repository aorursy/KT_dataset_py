# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data = pd.read_csv('/kaggle/input/leaf-classification/train.csv.zip')
test_data = pd.read_csv('/kaggle/input/leaf-classification/test.csv.zip')
train_data.describe()
print("Colums: ", train_data.columns.values)
print("Shape: ", train_data.shape)
print("Missing values:")
print(train_data.isnull().sum())
def encode(train, test):
    le = LabelEncoder().fit(train.species) 
    labels = le.transform(train.species)
    classes = list(le.classes_)                   
    test_ids = test.id                 

    train = train.drop(['species', 'id'], axis=1)  
    test = test.drop(['id'], axis=1)
    
    return train, labels, test, test_ids, classes

X, y, test_data, test_ids, classes = encode(train_data, test_data)
train_data.head(1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
from sklearn.preprocessing import normalize
X_train_norm=normalize(X_train)
X_test_norm=normalize(X_test)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()

clf.fit(X_train_norm, y_train)
print("RF Accuracy: " + repr(round(clf.score(X_test_norm, y_test) * 100, 2)) + "%")
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(criterion='entropy',
                             n_estimators=700,
                             min_samples_split=5,
                             min_samples_leaf=1,
                             max_features = "auto",
                             oob_score=True,
                             random_state=0,
                             n_jobs=-1)

clf.fit(X_train_norm, y_train)
print("RF Accuracy: " + repr(round(clf.score(X_test_norm, y_test) * 100, 2)) + "%")
clf = LinearDiscriminantAnalysis()
clf.fit(X_train_norm, y_train)
result = clf.predict_proba(normalize(test_data))
df = pd.DataFrame(result, columns=classes)
df.insert(0, 'id', test_ids)
df.reset_index()

print(result.shape)
filename = 'Prediction.csv'
df.to_csv(filename,index=False)
print('Saved file: ' + filename)
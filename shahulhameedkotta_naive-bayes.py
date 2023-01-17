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
import sklearn
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
data
label_names = data["target_names"]

labels = data["target"]

feature_names = data["feature_names"]

features = data["data"]
print(label_names)

print(labels[0])

print(feature_names[0])

print(features[0])
from sklearn.model_selection import train_test_split
train, test, train_labels, test_labels = train_test_split(features, labels, test_size = 0.2, random_state = 42)
print(train.shape)

print(test.shape)

print(train_labels.shape)

print(test_labels.shape)
# import the module

from sklearn.naive_bayes import GaussianNB



# Initialize the Classifier

gnb = GaussianNB()



# Train the Classifier

model = gnb.fit(train, train_labels)
# make predictions



preds = gnb.predict(test)

preds
from sklearn.metrics import accuracy_score

accuracy_score(test_labels, preds)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
file_path = "../input/zoo-animal-classification/zoo.csv"

data = pd.read_csv(file_path)
data.head()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data.columns
data.class_type.value_counts().sort_index()
sns.countplot(data['class_type'])
corrmat = data.corr()

plt.figure(figsize=(8,8))

sns.heatmap(data=corrmat, vmax=0.8, annot=True)
features = list(set(data.columns.tolist())  - set(['hair', 'eggs', 'tail', 'airborne', 'toothed', 'legs', 'tail']))
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split



X = data[features]

y = data['class_type']

X = X.drop([ 'class_type', 'animal_name'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, train_size=0.7, random_state=0) 
clf = LogisticRegression()

clf.fit(X_train, y_train)

predictions = clf.predict(X_valid)
from sklearn.metrics import confusion_matrix, accuracy_score

scores = confusion_matrix(y_valid, predictions)

accuracy = accuracy_score(y_valid, predictions)
scores
accuracy
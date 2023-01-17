# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
data = pd.read_csv("../input/heart.csv")
data.head()
data.describe()
data.info()
sns.pairplot(data, hue='target')
plt.figure(figsize = (10,7))

sns.heatmap(data.corr(), annot=True)
data.columns
X = data[['age', 'sex', 'cp', 'trestbps', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal']]

y = data['target']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, predictions))

print(confusion_matrix(y_test, predictions))
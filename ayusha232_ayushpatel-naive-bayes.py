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
iris_data = pd.read_csv("../input/iris-flower-dataset/Iris_flower_dataset.csv")

iris_data.head()
print(iris_data.shape)

iris_data.isnull().sum()
print(iris_data.info())
iris_data['Species'].unique()

iris_data["Species"].value_counts()
iris_data.describe()
iris_data.columns
cols = iris_data.columns

features = cols[0:4]

labels = cols[4]

print(features)

print(labels)
X = iris_data.iloc[:,:-1].values

y = iris_data.iloc[:,-1 ].values

X
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.3, random_state = 10)

X_train, y_train
from sklearn.naive_bayes import GaussianNB



# model

clf_g = GaussianNB()



# model fitting

clf_g.fit(X_train, y_train)



#prediction

results = cross_val_score(clf_g, X_train, y_train)

results.mean()
y_pred = clf_g.predict(X_valid)

y_pred
print(classification_report(y_valid, y_pred))
print(confusion_matrix(y_valid, y_pred))
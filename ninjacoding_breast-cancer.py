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
# Importing few libraries first

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix, accuracy_score
dataset = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-benign-or-malignant/tumor.csv')
# Printing first 5 rows

dataset.head()
# Fetching info about the dataset

dataset.info()
# Separating the features and labels

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, -1].values # The last column has the label value
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
feature_scaling = StandardScaler()

X_train = feature_scaling.fit_transform(X_train)

X_test = feature_scaling.transform(X_test)
rand_for = RandomForestClassifier(n_estimators = 10, max_depth = 5 ,criterion = 'entropy', random_state = 0)

rand_for.fit(X_train, y_train)
y_pred = rand_for.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
output = pd.DataFrame({'Real_class': y_test, 'Predicted_class': y_pred})
output.head()
output.to_csv('breast_cancer.csv', index=False)

print("Submission was successfully saved!")
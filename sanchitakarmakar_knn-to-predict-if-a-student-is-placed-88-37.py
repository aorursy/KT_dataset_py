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

dataset = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
dataset.head()
dataset = dataset.fillna(0)
x = dataset.iloc[:,[4,7,9,10,11,12]].values

y = dataset.iloc[:,-2].values
print(x)
print(x[0])
print(y)
# Encoding

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers = [('encoder',OneHotEncoder(), [2,4])],remainder = 'passthrough')

x = np.array(ct.fit_transform(x))
print(x)
print(x[0])
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

y = le.fit_transform(y)
print(y)
# Splitting the dataset into training set and test set

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=1)
# Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x_train[:,4:] = sc.fit_transform(x_train[:,4:])

x_test[:,4:] = sc.transform(x_test[:,4:])
# KNN

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

ac = accuracy_score(y_test, y_pred)

print(ac)
# I got accuracy of 88.37% by selecting hsc_p, degree_p, workex, etest_p, specialisation and mba_p and found by

# selecting these features I got highest accuracy.
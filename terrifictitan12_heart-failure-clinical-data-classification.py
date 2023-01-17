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
import matplotlib.pyplot as plt
data = pd.read_csv("/kaggle/input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv")
data.head()
data.isnull().sum()
data.dtypes
data.info()
data.describe()
import seaborn as sns
data['DEATH_EVENT'].hist()
from sklearn.model_selection import train_test_split
X = data.drop('DEATH_EVENT', axis=1)
y = data['DEATH_EVENT']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train.shape, x_test.shape, y_train.shape, y_test.shape
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
a = DecisionTreeClassifier()
b = RandomForestClassifier()
a.fit(x_train, y_train)
b.fit(x_train, y_train)
a_pred = a.predict(x_test)
print(a_pred)

print('\n')

b_pred = b.predict(x_test)
print(b_pred)
from sklearn.metrics import accuracy_score
a_acc = accuracy_score(y_test, a_pred)
a_acc*100
b_acc = accuracy_score(y_test, b_pred)
b_acc*100
data
a.predict([[50.0, 0, 196, 0, 45, 0, 395000.00, 1.6, 136, 1, 1, 285]])
b.predict([[50.0, 0, 196, 0, 45, 0, 395000.00, 1.6, 136, 1, 1, 285]])
a.predict([[55.0, 0, 7861, 0, 38, 0, 263358.03, 1.1, 136, 1, 0, 6]])
b.predict([[55.0, 0, 7861, 0, 38, 0, 263358.03, 1.1, 136, 1, 0, 6]])

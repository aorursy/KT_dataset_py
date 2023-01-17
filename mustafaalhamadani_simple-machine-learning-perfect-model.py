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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
import seaborn as sns
data = pd.read_csv('../input/iris/Iris.csv')
data
data.info()
data.isna().sum()
data['Species'].value_counts()
# changing the species column to category
for label, content in data.items():
    if pd.api.types.is_string_dtype(content):
        data[label] = content.astype('category')
for label, content in data.items():
    if not pd.api.types.is_numeric_dtype(content):
        data[label] = pd.Categorical(content).codes
sns.heatmap(data.corr(), annot=True)
# dropping the Id column because it's useless
data.drop('Id', axis=1, inplace=True)
#creating a function to try models
def model(m):
    x = data.drop('Species', axis=1)
    y = data['Species']
    np.random.seed(0)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    clf = m
    clf.fit(x_train, y_train)
    s = clf.score(x_test, y_test)
    return s

# Trying RandomForestClassifier
model(RandomForestClassifier())
model(LogisticRegression(max_iter=1000))
x = data.drop('Species', axis=1)
y = data['Species']
np.random.seed(0)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
clf = LogisticRegression(max_iter=1000)
clf.fit(x_train, y_train)
clf.score(x_test, y_test)
y_preds = clf.predict(x_test)
print(classification_report(y_test, y_preds))
cross_val_score(clf, x, y, cv=5)
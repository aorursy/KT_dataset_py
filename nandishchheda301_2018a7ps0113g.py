# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('/kaggle/input/minor-project-2020/train.csv')
df.head()
df.describe()
# drop duplicate rows (but first remove id)
df.drop(columns = ['id'], inplace = True)
df.drop_duplicates(inplace=True)
df.info()
X = df.iloc[:,:-1]
y = df.iloc[:, -1]
y.value_counts()
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state = 2)
X_over, y_over  = smote.fit_sample(X, y)

print(X_over.shape)
print(y_over.shape)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()              
scaled_X_train = scaler.fit_transform(X_over)     
scaled_X_train.shape
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(scaled_X_train,y_over)
test = pd.read_csv('/kaggle/input/minor-project-2020/test.csv')
# scale the test input
X_test = test.iloc[:,:-1]
scaled_X_test = scaler.fit_transform(X_test)
print(scaled_X_test.shape)
y_preds = clf.predict(scaled_X_test)
y_preds = pd.DataFrame(y_preds, columns = ['target'])
print(y_preds.value_counts())
y_preds['id'] = test['id']
y_preds.to_csv('trial4.csv', columns = ['id', 'target'], index = False)
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
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
df = pd.read_csv("/kaggle/input/heart-disease-prediction-using-logistic-regression/framingham.csv")
df.head()
df.info()
df.describe()
plt.figure(figsize=(20,15))
sns.heatmap(df.isna(), cmap= 'winter')
df.isnull().sum()
df.dropna(inplace=True)
plt.figure(figsize=(20,15))
sns.heatmap(df.isna(), cmap= 'winter')
df.isnull().sum()
df.corr()
df.head()
X = df.drop('TenYearCHD', inplace=False, axis=1)
y = df['TenYearCHD']
X.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 42)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
classification_model = LogisticRegression(max_iter = 10000)
classification_model.fit(X_train, y_train)
predictions = classification_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
accuracy
y_test[0]
predictions[0]
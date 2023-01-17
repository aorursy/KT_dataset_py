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
        file = os.path.join(dirname, filename)

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv(file)
data.head()
data.info()
sns.pairplot(data , hue = 'Kyphosis')
from sklearn.model_selection import train_test_split
X = data.drop('Kyphosis',axis = 1)
y = data['Kyphosis']
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size = 0.3 , random_state = 101)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier()
t = tree.fit(X_train , y_train)
pred = tree.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train , y_train)
pred_r = rfc.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,pred_r))
print(confusion_matrix(y_test,pred_r))
from sklearn import tree
plt.figure(figsize=(20,20))
tree.plot_tree(t)
plt.show()
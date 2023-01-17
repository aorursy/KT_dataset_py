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
data = pd.read_csv("/kaggle/input/predicte-heart-disease/data_hospital.csv")
data.head()
data.info()
data.isnull().any()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']] = scaler.fit_transform(data[['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']])
data.head()
data.describe()
len(data)
from sklearn.model_selection import train_test_split
X = data.drop('target', axis=1).values
y = data['target'].values
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
a = LogisticRegression().fit(x_train, y_train)
b = DecisionTreeClassifier(criterion='entropy').fit(x_train, y_train)
c = RandomForestClassifier(criterion='entropy', min_samples_leaf=20).fit(x_train, y_train)
d = SVC().fit(x_train, y_train)
a_pred = a.predict(x_test)
a_pred
b_pred = b.predict(x_test)
b_pred
c_pred = c.predict(x_test)
c_pred
d_pred = d.predict(x_test)
d_pred
from sklearn.metrics import accuracy_score
a_acc = accuracy_score(y_test, a_pred)
print('Logistic Regression Score :', a_acc*100)
b_acc = accuracy_score(y_test, b_pred)
print('Decision Tree Score :', b_acc*100)
c_acc = accuracy_score(y_test, c_pred)
print('Random Forest Score :', c_acc*100)
d_acc = accuracy_score(y_test, d_pred)
print('SVC Score :', d_acc*100)
x_test[43], y_test[43]
a.predict([[-1.47415758,  0.68100522,  1.00257707, -0.09273778, -0.62351787,
        -0.41763453, -1.00583187,  0.80259187, -0.69663055,  0.82852939,
        -0.64911323, -0.71442887, -0.51292188]])
b.predict([[-1.47415758,  0.68100522,  1.00257707, -0.09273778, -0.62351787,
        -0.41763453, -1.00583187,  0.80259187, -0.69663055,  0.82852939,
        -0.64911323, -0.71442887, -0.51292188]])
c.predict([[-1.47415758,  0.68100522,  1.00257707, -0.09273778, -0.62351787,
        -0.41763453, -1.00583187,  0.80259187, -0.69663055,  0.82852939,
        -0.64911323, -0.71442887, -0.51292188]])
d.predict([[-1.47415758,  0.68100522,  1.00257707, -0.09273778, -0.62351787,
        -0.41763453, -1.00583187,  0.80259187, -0.69663055,  0.82852939,
        -0.64911323, -0.71442887, -0.51292188]])
#Our Classifiers are giving good results

# This project was successful

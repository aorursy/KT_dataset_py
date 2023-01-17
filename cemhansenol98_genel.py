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
data = pd.read_csv('/kaggle/input/pima-indians-diabetes-database/diabetes.csv')

data.head()
data.columns
data.shape
data.info()
data.Age.median()
import matplotlib.pyplot as plt



plt.plot( 'Age', 'Glucose', data=data, linestyle='none', marker='o')
data.head()
y = data.Outcome

X = data.drop('Outcome', axis=1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
train_scale = scale.fit_transform(X_train)
test_scale = scale.fit_transform(X_test)
from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)
X_test
y_pred = model.predict(X_test)
y_pred
from sklearn.metrics import accuracy_score



score = accuracy_score(y_test, y_pred)

print('Accuracy : ', score)
from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators=150, max_depth=5)

rf.fit(X_train, y_train)
y_pred2 = rf.predict(X_test)
y_pred2
print("Accuracy Score:",accuracy_score(y_test, y_pred2) )
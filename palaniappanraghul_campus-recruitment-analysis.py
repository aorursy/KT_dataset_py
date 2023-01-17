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
data = pd.read_csv("/kaggle/input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv")

print(data)
data = data.fillna(0)

print(data)
data = data.drop(columns=['sl_no','salary'])

print(data)
x = data.drop(columns=['status'])

print(x)
y = data['status']

print(y)
from sklearn.preprocessing import LabelEncoder 

le = LabelEncoder() 

x = x.apply(le.fit_transform)

print(x)
y = le.fit_transform(y) 

print(y)
from sklearn.model_selection import train_test_split as tts

X_train,X_test,y_train,y_test = tts(x,y,test_size=0.3)
from sklearn.tree import DecisionTreeClassifier as DTC

model = DTC(criterion='entropy')

model.fit(X_train,y_train)
from sklearn.metrics import accuracy_score

print(accuracy_score(y_test,model.predict(X_test)))
a = model.predict(X_test)

print(a)
z= le.inverse_transform(a)

print(z)
from sklearn.metrics import accuracy_score, classification_report

print(classification_report(y_test, a))
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(n_estimators=100, random_state=0)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

print(y_pred)
z= le.inverse_transform(y_pred)

print(z)
accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt

import seaborn as sns
sns.countplot(x="degree_t", data=data, hue='specialisation')

plt.title("Candidate degree vs  Placement")

plt.xlabel("courses")

plt.ylabel("Number of candidate")

plt.show()
data.plot.scatter(x='degree_t', y='mba_p',title='Candidate Performance')


data.drop(['status'], axis=1).plot.line(title='Candidate Performance')

plt.show()
data['status'].value_counts().sort_index().plot.bar()
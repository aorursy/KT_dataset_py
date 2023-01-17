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

import seaborn as sns

import matplotlib.pyplot as plt



df = pd.read_csv("/kaggle/input/passenger-list-for-the-estonia-ferry-disaster/estonia-passenger-list.csv")

df1 = df.copy()

df.head()
df.isnull().sum()
df.isna().sum()
df.Survived.unique()
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()



df['Country'] = le.fit_transform(df['Country'])

df['Firstname'] = le.fit_transform(df['Firstname'])

df['Lastname'] = le.fit_transform(df['Lastname'])

df['Sex'] = le.fit_transform(df['Sex'])

df['Category'] = le.fit_transform(df['Category'])
df.head()
X = df.drop(['Survived'],axis = 1)

y = df['Survived']
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 43)
clf = RandomForestClassifier(n_estimators = 100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

from sklearn import metrics

# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
fig, ax=plt.subplots(figsize=(8,6))

sns.countplot(x='Survived', data=df1, hue='Sex')

#ax.set_ylim(0,500)

plt.title("Impact of Sex on Survived")

plt.show()
fig, ax=plt.subplots(figsize=(8,6))

sns.countplot(x='Survived', data=df1, hue='Category')

#ax.set_ylim(0,500)

plt.title("Impact of Category on Survived")

plt.show()
from sklearn.model_selection import GridSearchCV
param_grid = { 

    'n_estimators': [100, 200,300,400, 500, 600],

    'max_features': ['auto', 'sqrt', 'log2'],

    'max_depth' : [2,3,4,5,6,7,8,9],

    'criterion' :['gini', 'entropy']

}
rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv= 5)

rfc.fit(X_train, y_train)
rfc.best_params_
rfc1=RandomForestClassifier(random_state=100, max_features='sqrt', n_estimators= 200, max_depth=9, criterion='gini')
rfc1.fit(X_train, y_train)
pred=rfc1.predict(X_test)
print("Accuracy for Random Forest with Grid Search CV: ", metrics.accuracy_score(y_test,pred))
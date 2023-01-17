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
DataSet = pd.read_csv('../input/titanic/train.csv')

DataSet.head(5)
DataSet.shape
def f(name):

    if '.' in name:

        return name.split(',')[1].split('.')[0].strip()

    else:

        return 'unknown'

def g(i):

    if i in ['Mr']:

        return 1

    elif i in ['Master']:

        return 3

    elif i in ['Ms', 'Mlle', 'Miss']:

        return 4

    elif i in ['Mrs', 'Mme']:

        return 5

    else:

        return 2

    
DataSet['title'] = DataSet['Name'].apply(f).apply(g)

DataSet.head(20)
t = pd.crosstab(DataSet['title'], DataSet['Survived'])

t
t_pct = t.div(t.sum(1).astype(float), axis=0)

t_pct
t_pct.plot(kind='bar',stacked=True, title='title survival ratio')

plt.xlabel('title')

plt.ylabel('Survival ratio')
DataSet = DataSet.drop(['PassengerId','Name','Ticket'], axis=1) 

DataSet.head()
edt = pd.get_dummies(DataSet['Embarked'])

edt
edt.drop(['S'], axis = 1, inplace = True )
DataSet = DataSet.join(edt)

DataSet.head(20)
DataSet.drop(['Embarked'], axis = 1, inplace = True)

DataSet.head(20)
DataSet['Age'] = DataSet.groupby(['Pclass'])['Age'].transform(lambda x: x.fillna(x.mean()))

DataSet.drop("Cabin",axis=1,inplace=True)
s = sorted(DataSet['Sex'].unique())

z=zip(s, range(0, len(s) + 1))

gm = dict(z)

DataSet['Sex'] = DataSet['Sex'].map(gm).astype(int)

DataSet
DataSet.to_csv('train2.csv')
DataSet.head()
X = DataSet.iloc[:, 2:10].values

y = DataSet.iloc[: , 1].values
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(X_train)

X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators = 100000, criterion = 'entropy', random_state = 0)

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(y_test, y_pred)

print(cm)

accuracy_score(y_test, y_pred)
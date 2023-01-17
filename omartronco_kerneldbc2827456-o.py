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
# Importamos las librerías adicionales que necesitaremos



import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier
data = pd.read_csv('../input/titanic/train.csv')
X = data.drop(columns=['PassengerId','Survived','Name','Ticket','Fare','Embarked'])

X.Cabin = X.Cabin.str[0]
X_men = X[X.Sex=='male']

X_women = X[X.Sex=='female']

X.loc[(X["Sex"] == 'male') & (X["Age"].isnull()),'Age'] = X_men.Age.mean()

X.loc[(X["Sex"] == 'female') & (X["Age"].isnull()),'Age'] = X_women.Age.mean()
X = pd.get_dummies(X)



# Nos deshacemos de alguna de las dos variables de género, ya que contienen la misma información

X = X.drop(labels='Sex_male',axis=1)
print(X.head())
y = data.Survived

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=14, stratify=y)

y.describe()
params = np.arange(1, 30)

train_score = np.empty(len(params))

test_score = np.empty(len(params))

for i, k in enumerate(params):

    dt = DecisionTreeClassifier(max_depth=k,random_state=14).fit(X_train, y_train)

    train_score[i] = dt.score(X_train,y_train)

    test_score[i] = dt.score(X_test,y_test)

    

plt.style.use('classic')

plt.title('Árboles de decisión X_1: Cambiando la profundidad máxima')

plt.plot(params, test_score, label = 'Score de pruebas')

plt.plot(params, train_score, label = 'Score de entrenamiento')

plt.legend()

plt.xlabel('Profundidad máxima')

plt.ylabel('Score')

plt.show()



print(test_score)
dt = DecisionTreeClassifier(max_depth=4,random_state=14).fit(X_train, y_train)

print(dt.score(X_test,y_test))
params = np.arange(1, 20)



train_score = np.empty(len(params))

test_score = np.empty(len(params))

for i, k in enumerate(params):

    gbt = GradientBoostingClassifier(max_depth=k, random_state=14).fit(X_train,y_train)

    train_score[i] = gbt.score(X_train,y_train)

    test_score[i] = gbt.score(X_test,y_test)



plt.style.use('classic')

plt.title('Gradient Boosting: Cambiando la profundidad máxima')

plt.plot(params, test_score, label = 'Score de pruebas')

plt.plot(params, train_score, label = 'Score de entrenamiento')

plt.legend()

plt.xlabel('Profundidad máxima')

plt.ylabel('Score')

plt.show()



print(test_score)
gbt = GradientBoostingClassifier(max_depth=2, random_state=14).fit(X_train,y_train)

print(gbt.score(X_test,y_test))
test = pd.read_csv('../input/titanic/test.csv')

X_sub = test.drop(columns=['PassengerId','Name','Ticket','Fare','Embarked'])

X_sub.Cabin = X_sub.Cabin.str[0]



# Le asignamos la edad promedio de la base de pruebas

X_sub.loc[(X_sub["Sex"] == 'male') & (X_sub["Age"].isnull()),'Age'] = X_men.Age.mean()

X_sub.loc[(X_sub["Sex"] == 'female') & (X_sub["Age"].isnull()),'Age'] = X_women.Age.mean()



X_sub = pd.get_dummies(X_sub)

X_sub = X_sub.drop(labels='Sex_male',axis=1)

print(X_sub.head())
print('Tamaño de la base original: '+str(X.shape))

print('Tamaño de la base para entrega: '+str(X_sub.shape))
X_sub['Cabin_T']=0
print(X_sub.head())
print(X_sub.columns == X.columns)
prediction = gbt.predict(X_sub)
submission = pd.DataFrame()

submission['PassengerId'] = test['PassengerId']

submission['Survived'] = prediction

print(submission.head())
submission.to_csv('gender_submission.csv',index=False)
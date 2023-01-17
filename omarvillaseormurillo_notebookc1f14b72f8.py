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
import matplotlib.pyplot as plt

import pandas as pd

import numpy as np

%matplotlib inline



y_test = pd.read_csv('/kaggle/input/titanic/gender_submission.csv')[['Survived']]

df_train = pd.read_csv('/kaggle/input/titanic/train.csv')

df_test = pd.read_csv('/kaggle/input/titanic/test.csv')
df_train
df_test
y_test
import seaborn as sns



correlation_matrix = df_train.corr()

plt.figure(figsize=(15,10))

sns.heatmap(correlation_matrix, annot=True, square=True)
X = df_train.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket','Cabin', 'Parch', 'SibSp', 'Age'])

Xdf_test = df_test.drop(columns=['PassengerId', 'Name', 'Ticket','Cabin', 'Parch', 'SibSp', 'Age'])

Y = df_train.Survived

X
print(pd.isnull(X).sum())

print(pd.isnull(Xdf_test).sum())
X.fillna(value=X.mean(), inplace=True)

Xdf_test.fillna(value=Xdf_test.mean(), inplace=True)

X
print(pd.isnull(X).sum())

print(pd.isnull(Xdf_test).sum())
X.fillna(method='bfill',inplace=True)

print(pd.isnull(X).sum())
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

X['New Sex'] = encoder.fit_transform(X.Sex)

X['New Embarqued'] = encoder.fit_transform(X.Embarked)

Xdf_test['New Sex'] = encoder.fit_transform(Xdf_test.Sex)

Xdf_test['New Embarqued'] = encoder.fit_transform(Xdf_test.Embarked)

X
X.drop(columns=['Sex'], inplace=True)

Xdf_test.drop(columns=['Sex'], inplace=True)

X.drop(columns=['Embarked'], inplace=True)

Xdf_test.drop(columns=['Embarked'], inplace=True)

x = np.asanyarray(X.values)

print(f'X: {x.shape}, y: {Y.shape}')

X
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression

from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import confusion_matrix

from sklearn.pipeline import Pipeline

from sklearn.svm import SVC

import seaborn as sns

%matplotlib inline





xtrain, xtest, ytrain, ytest = train_test_split(x,Y)



model = Pipeline([

    ('scaler', StandardScaler()),

    ('poly', PolynomialFeatures(degree=2)),

    #('knn', KNeighborsClassifier(5))

    ('tree', DecisionTreeClassifier(max_depth=3))

    #('svm', SVC(gamma=2, C=1))

    #('mlp', MLPClassifier(alpha=1, max_iter=10000))

])



model.fit(xtrain,ytrain)



print(model.score(xtrain,ytrain))

print(model.score(xtest,ytest))



cm = confusion_matrix(Y, model.predict(x))

sns.heatmap(cm, annot=True)

plt.show()
print(model.score(StandardScaler().fit_transform(Xdf_test.values), y_test))

prediccion = model.predict(StandardScaler().fit_transform(Xdf_test.values))

print(confusion_matrix(y_test.values, prediccion))

for i in range(len(prediccion)):

    print(f'Ori: {y_test.values[i]} vs Pred: {prediccion[i]}')
df_test['Survived'] = prediccion

df = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': df_test.Survived})

df
df.to_csv('prediccion.csv', index=False)
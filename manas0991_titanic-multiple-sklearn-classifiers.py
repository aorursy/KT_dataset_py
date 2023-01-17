# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.model_selection import KFold

from sklearn.neural_network import MLPClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.gaussian_process import GaussianProcessClassifier

from sklearn.gaussian_process.kernels import RBF

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/titanic/train.csv')

df.head()
df.columns[df.isna().any()]
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

df.replace('', np.nan, inplace=True)

df.info()
df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Embarked'].fillna('S', inplace=True)

df.set_index('PassengerId', inplace=True)

df['Sex'] = df['Sex'].astype('category')

df['Sex'] = df['Sex'].cat.codes

df['Embarked'] = df['Embarked'].astype('category')

df['Embarked'] = df['Embarked'].cat.codes

df.head()
def convertRanges(data):

    _, bins = pd.qcut(data, 5, retbins=True)

    for i in range(len(data.values)):

        x = data.values[i]

        if(x>=bins[0] and x<bins[1]):

            x = 1

        elif(x>=bins[1] and x<bins[2]):

            x = 2

        elif(x>=bins[2] and x<bins[3]):

            x = 3

        elif(x>=bins[3] and x<bins[4]):

            x = 4

        elif(x>=bins[4] and x<=bins[5]):

            x = 5

        data.values[i] = x

    return data 
df['Age'] = convertRanges(df['Age'])

df['Fare'] = convertRanges(df['Fare'])

df.head()
classifiers = [

    KNeighborsClassifier(3),

    SVC(kernel="linear", C=0.025),

    SVC(gamma=2, C=1),

    GaussianProcessClassifier(1.0 * RBF(1.0)),

    DecisionTreeClassifier(max_depth=5),

    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),

    MLPClassifier(alpha=1, max_iter=1000),

    AdaBoostClassifier(),

    GaussianNB(),

    QuadraticDiscriminantAnalysis()]



modelAccuracy = []

for model in classifiers:

    kf = KFold(n_splits=10, shuffle=False)

    X = df.iloc[:,1:].values

    y = df.iloc[:,0].values

    scores = []

    for train_index, test_index in kf.split(df.values):

        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]

        model.fit(X_train, y_train)

        scores.append(model.score(X_test, y_test))



    modelAccuracy.append(np.mean(scores))



maxIndex = np.argmax(modelAccuracy)

model = classifiers[maxIndex]

model.fit(X, y)
df = pd.read_csv('../input/titanic/test.csv')

df.head()
df.columns[df.isna().any()]
df.drop(columns=['Name', 'Ticket', 'Cabin'], inplace=True)

df.replace('', np.nan, inplace=True)

df.info()
df['Age'].fillna(df['Age'].mean(), inplace=True)

df['Fare'].fillna(df['Fare'].mean(), inplace=True)

df['Age'] = convertRanges(df['Age'])

df['Fare'] = convertRanges(df['Fare'])



df['Sex'] = df['Sex'].astype('category')

df['Sex'] = df['Sex'].cat.codes

df['Embarked'] = df['Embarked'].astype('category')

df['Embarked'] = df['Embarked'].cat.codes

df.head()
X_test = df.iloc[:,1:].values

y_test = model.predict(X_test)



pred = pd.DataFrame({"PassengerId": df.iloc[:,0].values, "Survived": y_test})

pred.to_csv('results.csv', index=False, header=True)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.preprocessing import StandardScaler



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/titanic_data.csv')

df.shape

df.head()
df[['Name', 'Survived', 'Age']].values[55:60]
from sklearn import tree

from sklearn.model_selection import train_test_split



df2 = df

df2 = df2.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)



df2 = df2.dropna()



sexConvDict = {"male":1 ,"female" :2}

df2['Sex'] = df2['Sex'].apply(sexConvDict.get).astype(int)



df2.head()
#features = ['Sex']

#features = ['Sex', 'Parch']

#features = ['Sex', 'Parch', 'Pclass']

#features = ['Sex', 'Parch', 'Pclass', 'Age']

#features = ['Sex', 'Parch', 'Pclass', 'Age', 'Fare']

features = ['Sex', 'Parch', 'Pclass', 'Age', 'Fare', 'SibSp']



X = df2[features].values

y = df2['Survived'].values

scaler = StandardScaler()

X_standard = scaler.fit_transform(df2[features].values)

X_train, X_test, X_std_train, X_std_test, y_train, y_test = train_test_split(X, X_standard, y, test_size=0.50, random_state=1)



clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)
from sklearn.metrics import confusion_matrix



confusion_matrix(y_test, y_predicted)
from sklearn.metrics import accuracy_score



accuracy_score(y_test, y_predicted)
from sklearn.neural_network import MLPClassifier



clf = MLPClassifier(solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(3, 5), random_state=0)

clf = clf.fit(X_std_train, y_train)

y_predicted = clf.predict(X_std_test)



confusion_matrix(y_test, y_predicted)
accuracy_score(y_test, y_predicted)
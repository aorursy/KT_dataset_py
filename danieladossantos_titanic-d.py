# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train=pd.read_csv('../input/titanic/train.csv')

test=pd.read_csv('../input/titanic/test.csv')

gen=pd.read_csv('../input/titanic/gender_submission.csv')

test_df=test
train["Sex"] = train["Sex"].map({"male": 0, "female":1})

train.head()
test["Sex"] = test["Sex"].map({"male": 0, "female":1})

test.head()
xTreino=train[['Sex','Pclass']]

yTreino=train['Survived']
xTreino.head()
Xdummies_Treino = pd.get_dummies(xTreino, columns = ["Pclass"],prefix="Pc").astype(int)

Xdummies_Treino.head()
yTreino.head()
test.drop(["PassengerId","Name","Age","SibSp","Parch","Ticket","Fare","Cabin","Embarked"], axis = 1, inplace = True)
test.head()
Xdummies_df_test = pd.get_dummies(test, columns = ["Pclass"],prefix="Pc").astype(int)

Xdummies_df_test.head()
X_TESTE=Xdummies_df_test.values

X=Xdummies_Treino.values

Y=yTreino.values
print(X)
svc_modelo = SVC()

svc_modelo.fit(X, Y)

previsao=svc_modelo.predict(X_TESTE)
acc_svc = round(svc_modelo.score(X, Y) * 100, 2)

acc_svc
decision_tree_modelo = DecisionTreeClassifier()

decision_tree_modelo.fit(X, Y)

previsao2 = decision_tree_modelo.predict(X_TESTE)

acc_decision_tree = round(decision_tree_modelo.score(X, Y) * 100, 2)

acc_decision_tree
saida = pd.DataFrame({"PassengerId": test_df['PassengerId'],

                      "Survived": previsao})

saida.to_csv('titanic_submissao_dani.csv', index=False)
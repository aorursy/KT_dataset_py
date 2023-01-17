import numpy as np

import pandas as pd

import seaborn as sns

!ls -l ../input/titanic
train = pd.read_csv('../input/titanic/train.csv')
train.describe()
train.head()
train.info()
train.loc[train.Sex=='male', 'nsex'] = 0

train.loc[train.Sex=='female', 'nsex'] = 1
train['nage'] = train.Age.fillna(train.Age.mean())
train.loc[train.Embarked=='S', 'nembarked'] = 0

train.loc[train.Embarked=='C', 'nembarked'] = 1

train.loc[train.Embarked=='Q', 'nembarked'] = 2

train.loc[train.Embarked.isnull(), 'nembarked'] = 0
train.loc[~train.Cabin.isnull(), 'ncabin'] = 1

train.loc[train.Cabin.isnull(), 'ncabin'] = 0
train.info()
from sklearn.svm import SVC
model = SVC()
X = train[['Pclass', 'nsex', 'nage', 'SibSp', 'Parch', 'Fare', 'ncabin', 'nembarked']]

y = train.Survived
from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(X, y, random_state=0)
model.fit(X_train, y_train)
model.score(X_train, y_train)
model.score(X_test, y_test)
from sklearn.decomposition import PCA #主成分分析

from sklearn.linear_model import LogisticRegression # ロジスティック回帰

from sklearn.neighbors import KNeighborsClassifier # K近傍法

from sklearn.svm import SVC # サポートベクターマシン

from sklearn.tree import DecisionTreeClassifier # 決定木

from sklearn.ensemble import RandomForestClassifier # ランダムフォレスト

from sklearn.ensemble import AdaBoostClassifier # AdaBoost

from sklearn.naive_bayes import GaussianNB # ナイーブ・ベイズ

#from sklearn.lda import LDA # 線形判別分析

#from sklearn.qda import QDA # 二次判別分析



names = ["Logistic Regression", "Nearest Neighbors", 

         "Linear SVM", "RBF SVM", "Sigmoid SVM", 

         "Decision Tree","Random Forest", "AdaBoost", "Naive Bayes" ]

#         "Linear Discriminant Analysis","Quadratic Discriminant Analysis"]

'''names = ["Logistic Regression", "Nearest Neighbors", 

         "Linear SVM", "Polynomial SVM", "RBF SVM", "Sigmoid SVM", 

         "Decision Tree","Random Forest", "AdaBoost", "Naive Bayes",

         "Linear Discriminant Analysis","Quadratic Discriminant Analysis"]'''



classifiers = [

    LogisticRegression(),

    KNeighborsClassifier(),

    SVC(kernel="linear"),

#    SVC(kernel="poly"),

    SVC(kernel="rbf"),

    SVC(kernel="sigmoid"),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GaussianNB()]

#    LDA(),

#    QDA()]



result = []

for name, clf in zip(names, classifiers): # 指定した複数の分類機を順番に呼び出す

    clf.fit(X_train, y_train) # 学習

    score1 = clf.score(X_train, y_train) # 正解率（train）の算出

    score2 = clf.score(X_test, y_test) # 正解率（test）の算出

    result.append([score1, score2]) # 結果の格納

    print(name + ' ended')





df_result = pd.DataFrame(result, columns=['train', 'test'], index=names).sort_values('test', ascending=False)



print(df_result)
df_result
def prepare(train):

    train.loc[train.Sex=='male', 'nsex'] = 0

    train.loc[train.Sex=='female', 'nsex'] = 1

    train['nage'] = train.Age.fillna(train.Age.mean())

    train.loc[train.Embarked=='S', 'nembarked'] = 0

    train.loc[train.Embarked=='C', 'nembarked'] = 1

    train.loc[train.Embarked=='Q', 'nembarked'] = 2

    train.loc[train.Embarked.isnull(), 'nembarked'] = 0

    train.loc[~train.Cabin.isnull(), 'ncabin'] = 1

    train.loc[train.Cabin.isnull(), 'ncabin'] = 0

    return train
test = pd.read_csv('../input/titanic/test.csv')
ntest = prepare(test)
ntest[['Pclass', 'nsex', 'nage', 'SibSp', 'Parch', 'Fare', 'ncabin', 'nembarked']].info()
ntest.Fare = ntest.Fare.fillna(ntest.Fare.mean())

predict = classifiers[0].predict(ntest[['Pclass', 'nsex', 'nage', 'SibSp', 'Parch', 'Fare', 'ncabin', 'nembarked']])
predict
submit = pd.DataFrame(columns={'PassengerId', "Survived"})
submit.PassengerId = ntest.PassengerId

submit.Survived = pd.Series(predict)
submit = submit[submit.columns[::-1]]
submit
submit.to_csv("/kaggle/working/submit_logistic_regression.csv", index=False)
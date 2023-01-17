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
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
df = pd.read_csv('../input/creditcardfraud/version/3.csv')
df.head()
print(df['Class'].sum())
print(df['Class'].count() - df['Class'].sum())
scaler = RobustScaler()
df['Amount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
df['Time'] = scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df.head()
from sklearn.model_selection import StratifiedShuffleSplit
def stratified_shuffle_split(X, y):
    stratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    for train_index, test_index in stratifiedShuffleSplit.split(X, y):
        print("Número de Transações:\nTreino: ", len(train_index), "\nTeste: ", len(test_index))
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        print("\nNúmero de Fraudes:\nTreino: ", np.sum(y_train), "\nTeste: ", np.sum(y_test))
    return X_train, X_test, y_train, y_test
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
def get_measures(X_train, X_test, y_train, y_test, y_pred):
    print("\nAcurácia (Treino)", clf.score(X_train, y_train))
    print("Acurácia (Teste)", clf.score(X_test, y_test))
from sklearn.metrics import confusion_matrix
def show_confusion_matrix(y_test, y_pred):
    print("\n", confusion_matrix(y_test, y_pred))
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.tree import export_text
def run_classifier(clf, X, y):
    #divide os dados entre treino e teste
    X_train, X_test, y_train, y_test = stratified_shuffle_split(X, y)
    #executa o classificador
    clf = clf.fit(X_train, y_train)
    #predição das classes para os dados de teste
    y_pred = clf.predict(X_test)
    #exibe algumas métricas
    #get_measures(X_train, X_test, y_train, y_test, y_pred)
    print(classification_report(y_test, y_pred))
    print(clf.get_params())
    #exibe confusion matriz
    show_confusion_matrix(y_test, y_pred)
    #print(export_text(clf))
#cria as variáveis
X = df.drop('Class', axis=1).values
y = df['Class'].values
from sklearn import tree
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(class_weight="balanced")
run_classifier(clf, X, y)

#instancia a árvore de decisão
clf = tree.DecisionTreeClassifier()
run_classifier(clf, X, y)
from sklearn import tree

#instancia a árvore de decisão
clf = tree.DecisionTreeClassifier()
run_classifier(clf, X, y)
from sklearn.ensemble import RandomForestClassifier

clf = RandomForestClassifier(class_weight="balanced")
run_classifier(clf, X, y)
from sklearn.ensemble import AdaBoostClassifier

clf = AdaBoostClassifier(n_estimators=100, random_state=0, class_weight="balanced")
run_classifier(clf, X, y)
from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(min_samples_leaf=10, class_weight="balanced")
run_classifier(clf, X, y)
import xgboost as xgb

clf = xgb.XGBClassifier(class_weight="balanced")
run_classifier(clf, X, y)
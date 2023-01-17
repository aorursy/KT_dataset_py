import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline
# Vamos iniciar o notebook importanto o Dataset

titanic_df = pd.read_csv("../input/train.csv")

test_df    = pd.read_csv("../input/test.csv")



# Podemos observar as primeiras linhas dele.

titanic_df.head()
numeric_features = ['Pclass', 'SibSp', 'Parch', 'Fare']
titanic_df[numeric_features].head()
train_X = titanic_df[numeric_features].as_matrix()

print(train_X.shape)

train_y = titanic_df['Survived'].as_matrix()

print(train_y.shape)
train_X
train_y
from sklearn.tree import DecisionTreeClassifier

dt_clf = DecisionTreeClassifier(random_state=42, max_depth=5, criterion='entropy')
dt_clf.fit(train_X, train_y)
dt_clf.score(train_X, train_y)
dt_clf.feature_importances_
# Se voce estiver usando linux e tiver curiosidade em ver a arvore...

#from sklearn.tree import export_graphviz

#export_graphviz(dt_clf, feature_names=numeric_features, out_file='/tmp/x.dot', filled=True, class_names=True, impurity=False, proportion=True)

#!dot -Tpng /tmp/x.dot -o /tmp/x.png
# cria um array cuja posição 1 é 1, posição 2 é 2, ...

max_depth_arr = np.arange(1, 21)

# criar um array com 1000 posições zeradas

accuracy_arr = np.zeros(20)



for i, max_depth in enumerate(max_depth_arr):

    ## calcula accuracy usando o max_depth em questao

    accuracy_arr[i] = 0 # coloque aqui o seu calculo de accuracy



plt.plot(max_depth_arr, accuracy_arr);    
optimal_max_depth = 5 # coloque aqui o max_depth que voce encontrou
dt_clf = DecisionTreeClassifier(random_state=42, max_depth=optimal_max_depth)

dt_clf.fit(train_X, train_y)
train_X[0:5]
train_y[0:5]
dt_clf.predict(train_X[0:5])
test_df.head()
test_df['Fare'] = test_df['Fare'].fillna(0)
test_X = test_df[numeric_features].as_matrix()

print(test_X.shape)
test_X
y_pred = dt_clf.predict(test_X)
y_pred
sample_submission_df = pd.DataFrame()
sample_submission_df['PassengerId'] = test_df['PassengerId']

sample_submission_df['Survived'] = y_pred

sample_submission_df
sample_submission_df.to_csv('basic_decision_tree.csv', index=False)
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
from sklearn.model_selection import train_test_split

train_X, valid_X, train_y, valid_y = train_test_split(titanic_df[numeric_features].as_matrix(), 

                                                      titanic_df['Survived'].as_matrix(),

                                                      test_size=0.2,

                                                      random_state=42)

                                                      

                                                      

print(train_X.shape)

print(valid_X.shape)                                           

print(train_y.shape)

print(valid_y.shape)
train_X
train_y
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(random_state=42, n_estimators=10, max_depth=5)
rf_clf.fit(train_X, train_y)
print(rf_clf.score(train_X, train_y))

print(rf_clf.score(valid_X, valid_y))
rf_clf.feature_importances_
import seaborn as sns

sns.barplot(rf_clf.feature_importances_, numeric_features);
rf_clf.estimators_
optimal_max_depth = 5 # coloque aqui o max_depth que voce encontrou

optimal_n_estimators = 10 # coloque aqui o n_estimators que voce encontrou
rf_clf = RandomForestClassifier(random_state=42, max_depth=optimal_max_depth, n_estimators=optimal_n_estimators)

rf_clf.fit(titanic_df[numeric_features].as_matrix(), titanic_df['Survived'].as_matrix())
test_df['Fare'] = test_df['Fare'].fillna(0)
test_X = test_df[numeric_features].as_matrix()

print(test_X.shape)
test_X
y_pred = rf_clf.predict(test_X)
y_pred
submission_df = pd.DataFrame()
submission_df['PassengerId'] = test_df['PassengerId']

submission_df['Survived'] = y_pred

submission_df
submission_df.to_csv('basic_random_forest.csv', index=False)
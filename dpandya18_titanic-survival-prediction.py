import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline
path1 = "../input/train.csv"
path2 = "../input/test.csv"
path3 = "../input/gender_submission.csv"
training_df = pd.read_csv(path1, header = 0)
test_df = pd.read_csv(path2, header = 0)

training_df.head()
test_df.head()
training_df1 = training_df.loc[:,["Survived", "Pclass", "Sex", "SibSp", "Age", "Parch", "Embarked"]]
test_df1 = test_df.loc[:,["Pclass", "Sex","SibSp", "Age", "Parch", "Embarked"]]
training_df1.head()
test_df1.head()
training_df1.isnull().sum(axis = 0)
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(training_df1.loc[:, ['Age']])
training_df1.loc[:, ['Age']] = imputer.transform(training_df1.loc[:, ['Age']])
imputer_test = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer_test = imputer_test.fit(test_df1.loc[:, ['Age']])
test_df1.loc[:, ['Age']] = imputer_test.transform(test_df1.loc[:, ['Age']])
training_df1 = training_df1.dropna(axis=0, how='any', subset = ['Embarked'])
test_df1 = test_df1.dropna(axis=0, how='any', subset = ['Embarked'])
X_train = training_df1.loc[:, training_df1.columns != 'Survived']
y_train = training_df1.loc[:, ['Survived']]

X_test = test_df1
X_train = pd.get_dummies(X_train, columns = ["Sex", "Embarked"])
X_test = pd.get_dummies(X_test, columns = ["Sex", "Embarked"])

training_columns = X_train.columns
test_columns = X_test.columns
X_train["Age"].median()
X_test["Age"].median()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = pd.DataFrame(sc.fit_transform(X_train), columns = training_columns)
X_test = pd.DataFrame(sc.transform(X_test), columns = test_columns)
X_train.head()
y_test = pd.read_csv(path3, header = 0)
y_test = y_test['Survived']
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

lr_clf = LogisticRegression(random_state = 42)

lr_clf.fit(X_train, y_train)

y_pred = lr_clf.predict(X_test)
cf_lr = confusion_matrix(y_test, y_pred)
print(classification_report(y_test, y_pred))
from sklearn.neighbors import KNeighborsClassifier
import sklearn.metrics as ms
knn_clf = KNeighborsClassifier(n_neighbors = 18, metric = 'minkowski', p = 2)
knn_clf.fit(X_train, y_train)

y_pred_knn = knn_clf.predict(X_test)
cf_knn = confusion_matrix(y_test, y_pred_knn)
print(classification_report(y_test, y_pred_knn))
ms.accuracy_score(y_test, y_pred_knn)
from sklearn.svm import SVC
svm_clf = SVC(kernel = 'linear', random_state = 42)
svm_clf.fit(X_train,y_train)
y_pred_svm = svm_clf.predict(X_test)


print(confusion_matrix(y_test,y_pred_svm))
print(classification_report(y_test, y_pred_svm))
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(criterion = 'entropy')
dtree = dtree.fit(X_train, y_train)  

y_pred_dt = dtree.predict(X_test)

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))
from sklearn.model_selection import GridSearchCV
neighbors_list = list(range(1,21))
parameters = [{'n_neighbors': neighbors_list}]
grid_search = GridSearchCV(estimator = knn_clf, param_grid = parameters, scoring = 'accuracy', n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_parameters
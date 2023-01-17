# Code you have previously used to load data
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from xgboost import XGBClassifier


# Path of the file to read. We changed the directory structure to simplify submitting to a competition
train_file_path = '../input/train.csv'
# path to file you will use for predictions on submission data
submission_data_path = '../input/test.csv'

data = pd.read_csv(train_file_path)
data.dropna(axis=0, subset=['Survived'], inplace=True)
submission_data = pd.read_csv(submission_data_path)

features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']
y = data.Survived
X = data[features]
submission_X = submission_data[features]

# Feature Engineering on Age
X['Elderly'] = np.where(X['Age'] >= 50, 1, 0)
X['Child'] = np.where(X['Age'] <= 13, 1, 0)
submission_X['Elderly'] = np.where(submission_X['Age'] >= 50, 1, 0)
submission_X['Child'] = np.where(submission_X['Age'] <= 13, 1, 0)

# One-hot Encoding
X = pd.get_dummies(X)
submission_X = pd.get_dummies(submission_data)

final_train, final_submission = X.align(submission_X, join='left', axis=1)

# Test-train Split
train_X, test_X, train_y, test_y = train_test_split(final_train.as_matrix(), y.as_matrix(), test_size=0.25)

# Handle Missing Values
my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
final_submission = my_imputer.transform(final_submission)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(train_X, train_y)
logreg_pred = logreg.predict(test_X)
acc_log = accuracy_score(test_y, logreg_pred)

# Support Vector Machines
svc = SVC()
svc.fit(train_X, train_y)
svc_pred = svc.predict(test_X)
acc_svc = accuracy_score(test_y, svc_pred)

# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(train_X, train_y)
linear_svc_pred = linear_svc.predict(test_X)
acc_linear_svc = accuracy_score(test_y, linear_svc_pred)

# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_X, train_y)
knn_pred = knn.predict(test_X)
acc_knn = accuracy_score(test_y, knn_pred)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(train_X, train_y)
gaussian_pred = gaussian.predict(test_X)
acc_gaussian = accuracy_score(test_y, gaussian_pred)

# Perceptron
perceptron = Perceptron()
perceptron.fit(train_X, train_y)
perceptron_pred = perceptron.predict(test_X)
acc_perceptron = accuracy_score(test_y, perceptron_pred)

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(train_X, train_y)
sgd_pred = sgd.predict(test_X)
acc_sgd = accuracy_score(test_y, sgd_pred)

# Decision Tree
decision_tree = DecisionTreeClassifier(max_leaf_nodes=100, random_state=1)
decision_tree.fit(train_X, train_y)
decision_tree_pred = decision_tree.predict(test_X)
acc_decision_tree = accuracy_score(test_y, decision_tree_pred)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(train_X, train_y)
random_forest_pred = random_forest.predict(test_X)
acc_random_forest = accuracy_score(test_y, random_forest_pred)

# XGBoost default
xgb_def = XGBClassifier(n_estimators=1000, learning_rate=0.05)
xgb_def.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
xgb_def_pred = xgb_def.predict(test_X)
acc_xgb_def = accuracy_score(test_y, xgb_def_pred)

# XGBoost Binary Logistic Regression
xgb_binl = XGBClassifier(n_estimators=1000, learning_rate=0.05, objective='binary:logistic')
xgb_binl.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
xgb_binl_pred = xgb_binl.predict(test_X)
acc_xgb_binl = accuracy_score(test_y, xgb_binl_pred)

# XGBoost Reg Logistic Regression
xgb_regl = XGBClassifier(n_estimators=1000, learning_rate=0.05, objective='reg:logistic')
xgb_regl.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
xgb_regl_pred = xgb_regl.predict(test_X)
acc_xgb_regl = accuracy_score(test_y, xgb_regl_pred)

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Stochastic Gradient Decent', 'Linear SVC', 
              'Decision Tree', 'XGBoost Default', 'XGBoost Binary Logistic',
              'XGBoost Reg Logistic'],
    'Score': [acc_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_sgd, acc_linear_svc, acc_decision_tree, acc_xgb_def,
              acc_xgb_binl, acc_xgb_regl]})
print(models.sort_values(by='Score', ascending=False))
# make predictions which we will submit. 
submission_preds = xgb_regl.predict(final_submission)

print(submission_preds)
output = pd.DataFrame({
        "PassengerId": submission_data.PassengerId,
        "Survived": submission_preds})

output.to_csv('submission.csv', index=False)

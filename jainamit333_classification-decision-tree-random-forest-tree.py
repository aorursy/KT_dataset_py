import pandas as pd

import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV



from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score
data  = pd.read_csv('../input/heart.csv')
data.head()
data.describe()
data.apply(lambda s: data.corrwith(s))
feature_coloums = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 

                   'ca', 'thal']

target_col = ['target']
def shuffle(data_frame):

     return data_frame.reindex(np.random.permutation(data_frame.index))
data = shuffle(data)
def split_training_and_test(data_frame, training_percentage):

    training_number = data_frame.shape[0] * training_percentage / 100

    test_number = data_frame.shape[0] - training_number

    return data_frame.head(int(training_number)), data_frame.tail(int(test_number))
training_data, test_data = split_training_and_test(data, 80)
X_train = np.array(training_data[feature_coloums])

y_train = np.array(training_data[target_col])
X_test = np.array(test_data[feature_coloums])

y_test = np.array(test_data[target_col])
print('Training data shape', X_train.shape)

print('Training data shape', X_test.shape)



print('Training label shape', y_train.shape)

print('Training label shape', y_test.shape)
parameters = {'max_depth':[3,5, 7,10,None], 'min_samples_split':[2, 5,10,30,50], 'criterion':['gini', 'entropy']}

decision_tree = DecisionTreeClassifier()

grid_search = GridSearchCV(decision_tree, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train, y_train)
print('Best Params', grid_search.best_params_)

print("Best Score", grid_search.best_score_)
prediction = grid_search.best_estimator_.predict(X_test)
print('Prediction shape', prediction.shape)
print('Accuracy Score', accuracy_score(y_test, prediction))

print('Classification  Report', classification_report(y_test, prediction))

print('ROC AUC', roc_auc_score(y_test, prediction))
parameters = {'max_depth':[3,5, 7,10,None], 'min_samples_split':[2, 5,10,30,50], 'criterion':['gini', 'entropy'],

             'n_estimators':[10,50,70,100,150,200]}

random_tree = RandomForestClassifier(n_jobs=-1)
grid_search = GridSearchCV(random_tree, parameters, n_jobs=-1, cv=3, scoring='roc_auc')
grid_search.fit(X_train, y_train)
print('Best Params', grid_search.best_params_)

print("Best Score", grid_search.best_score_)



prediction = grid_search.best_estimator_.predict(X_test)
print('Accuracy Score', accuracy_score(y_test, prediction))

print('Classification  Report', classification_report(y_test, prediction))

print('ROC AUC', roc_auc_score(y_test, prediction))
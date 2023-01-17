import warnings
warnings.filterwarnings("ignore")
import pandas as pd
columns = ['Age','Workclass','fnlgwt','Education','Education Num','Marital Status','Occupation',
           'Relationship','Race','Sex','Capital Gain','Capital Loss','Hours/Week','Native Country','Target']
orig_data = pd.read_csv("../input/adult-training.csv", 
                        header=None, na_values='?', sep=', ', engine='python', names=columns)
orig_data.head()
test_data = pd.read_csv("../input/adult-test.csv", 
                        header = None, skiprows=1, na_values='?', sep=', ', engine='python', names=columns)
test_data.head()
data = orig_data.copy()
data.shape
data.info()
data.describe().T
import matplotlib.pyplot as plt
import numpy as np
for column in data.columns:
    if data.dtypes[column] == np.object:
        data[column].value_counts().plot(kind="bar", title=column)
    else:
        data[column].hist()
        plt.title(column)
    plt.show()
import matplotlib.pyplot as plt
import missingno as msno
msno.bar(data)
plt.show()
modes = data.mode().iloc[0]
data.fillna(modes, inplace=True)

#Verifying
msno.bar(data)
plt.show()
cols_to_encode = [data.columns[i] for i in range(data.shape[1]) if data.dtypes[i] == np.object]
cols_to_encode
data.groupby('Education').nunique()['Education Num']
data.drop('Education', axis = 1, inplace = True)
cols_to_encode.remove('Education')
data.head()
data = pd.get_dummies(data, drop_first = True)
data.head()
from sklearn.model_selection import KFold, GridSearchCV, cross_val_score, cross_val_predict
cv = KFold(5, random_state = 1)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
                             classification_report, confusion_matrix)
Model = []
Accuracy = []
Precision = []
Recall = []
F1 = []
AUC = []
x = data[data.columns[:-1]]
y = data[data.columns[-1]]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
from sklearn.dummy import DummyClassifier
clf = DummyClassifier(strategy = 'most_frequent',random_state = 1)
Model.append("Dummy")
Accuracy.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='accuracy').mean())
Precision.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='precision').mean())
Recall.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='recall').mean())
F1.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='f1').mean())
AUC.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='roc_auc').mean())
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
param_grid = {'C': [0.1, 0.4, 0.7, 1, 4, 7, 10]}
grid1 = GridSearchCV(lr, param_grid, cv=cv).fit(x_scaled, y)
print("Grid Logistic Regression: ", grid1.best_score_, grid1.best_params_)
#Appending results from the best of all above models
from sklearn.linear_model import LogisticRegression
clf = grid1.best_estimator_
Model.append("Logistic Regression")
Accuracy.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='accuracy').mean())
Precision.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='precision').mean())
Recall.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='recall').mean())
F1.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='f1').mean())
AUC.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='roc_auc').mean())
from sklearn.svm import SVC
svc = SVC()
param_grid = {'C': [0.1, 0.4, 0.7, 1, 4, 7, 10],
              'kernel': ['linear', 'rbf']}
grid1 = GridSearchCV(svc, param_grid, cv=cv).fit(x_scaled, y)
print("Grid SVC: ", grid1.best_score_, grid1.best_params_)
#Appending results from the best of all above models
from sklearn.svm import SVC
clf = grid1.best_estimator_
Model.append("SVC")
Accuracy.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='accuracy').mean())
Precision.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='precision').mean())
Recall.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='recall').mean())
F1.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='f1').mean())
AUC.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='roc_auc').mean())
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
param_grid = {'max_depth': [10, 40, 70, 100, 400, 700, None],
              'criterion': ['gini','entropy']}
grid1 = GridSearchCV(dtc, param_grid, cv=cv).fit(x_scaled, y)
print("Grid DTC: ", grid1.best_score_, grid1.best_params_)
#Appending results from the best of all above models
from sklearn.tree import DecisionTreeClassifier
clf = grid1.best_estimator_
Model.append("Decision Tree Classifier")
Accuracy.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='accuracy').mean())
Precision.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='precision').mean())
Recall.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='recall').mean())
F1.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='f1').mean())
AUC.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='roc_auc').mean())
#finding the optimum number of trees first in Random Forest
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1)
param_grid = {'n_estimators': [10,40,70,100,400,700,1000]}
grid1 = GridSearchCV(rfc, param_grid, cv=cv).fit(x_scaled, y)
print("Grid RFC: ", grid1.best_score_, grid1.best_params_)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=1)
param_grid = {'n_estimators': [250,300,350,400,450,500,550]}
grid2 = GridSearchCV(rfc, param_grid, cv=cv).fit(x_scaled, y)
print("Grid RFC: ", grid2.best_score_, grid2.best_params_)
from sklearn.ensemble import RandomForestClassifier
rfc = grid2.best_estimator_
param_grid = {'criterion': ['gini','entropy'],
              'max_depth': [5, 10, 15, 20, 25, 30, None]}
grid3 = GridSearchCV(rfc, param_grid, cv=cv).fit(x_scaled, y)
print("Grid RFC: ", grid3.best_score_, grid3.best_params_)
#Appending results from the best of all above models
from sklearn.ensemble import RandomForestClassifier
clf = grid3.best_estimator_
Model.append("Random Forest Classifier")
Accuracy.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='accuracy').mean())
Precision.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='precision').mean())
Recall.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='recall').mean())
F1.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='f1').mean())
AUC.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='roc_auc').mean())
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(random_state=1)
param_grid = {'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
              'activation': ['tanh', 'relu'],
              'solver': ['sgd', 'adam'],
              'alpha': [0.0001, 0.05],
              'learning_rate': ['constant','adaptive']}
grid1 = GridSearchCV(mlp, param_grid, cv=cv).fit(x_scaled, y)
print("Grid MLP: ", grid1.best_score_, grid1.best_params_)
#Appending results from the best of all above models
from sklearn.neural_network import MLPClassifier
clf = grid1.best_estimator_
Model.append("MLP Classifier")
Accuracy.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='accuracy').mean())
Precision.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='precision').mean())
Recall.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='recall').mean())
F1.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='f1').mean())
AUC.append(cross_val_score(clf, x_scaled, y, cv=cv, scoring='roc_auc').mean())
evaluation = pd.DataFrame({'Model': Model, 
                           'Accuracy': Accuracy, 
                           'Precision': Precision, 
                           'Recall': Recall,
                           'F1 Score': F1, 
                           'AUC': AUC})
print("FOLLOWING ARE THE TRAINING SCORES: ")
evaluation
#applying model on test set
test_data.drop('Education', axis = 1, inplace = True)
test_data.shape
#imputing missing test data values with same values as train
test_data.fillna(modes, inplace=True) 
test_data.shape
test_data = pd.get_dummies(test_data, drop_first = True)
test_data.shape
#since get_dummies gave number of columns mismatch
missing_cols = set(data.columns) - set(test_data.columns )
for col in missing_cols:
    test_data[col] = 0
test_data.head()
#Because of the dot (.) in target value in test data, new column 'Target_ >50K' is added from training data
test_data['Target_>50K'] = test_data['Target_>50K.']
test_data.drop('Target_>50K.', axis = 1, inplace = True)
#to ensure same column placement as train data
test_data = test_data[data.columns]
test_data.head()
x_test = scaler.transform(test_data[test_data.columns[:-1]])
y_test = test_data[test_data.columns[-1]].values
x_test.shape
clf = RandomForestClassifier(random_state = 1, criterion = 'gini', 
                             max_depth = 20, n_estimators = 400)
print("Test Accuracy:",cross_val_score(clf, x_test, y_test, cv=cv, scoring='accuracy').mean())
print("Test Precision:",cross_val_score(clf, x_test, y_test, cv=cv, scoring='precision').mean())
print("Test Recall:",cross_val_score(clf, x_test, y_test, cv=cv, scoring='recall').mean())
print("Test F1 Score:",cross_val_score(clf, x_test, y_test, cv=cv, scoring='f1').mean())
print("Test AUC:",cross_val_score(clf, x_test, y_test, cv=cv, scoring='roc_auc').mean())
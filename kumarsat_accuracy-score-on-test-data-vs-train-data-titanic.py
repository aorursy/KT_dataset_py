import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head()
test_data.head()
train_data.describe()
train_data.columns
train_data.dtypes
column_names = train_data.columns
for column in column_names:
    print(column + ' --> ' + str(train_data[column].isnull().sum()))
train_data.Survived.value_counts()
plt = train_data.Survived.value_counts().plot('bar')
plt.set_xlabel('Survived or Not')
plt.set_ylabel('Passenger Count')
plt = train_data.Pclass.value_counts().sort_index().plot('bar')
plt.set_xlabel('Pclass')
plt.set_ylabel('Passenger Count')
train_data[['Pclass', 'Survived']].groupby('Pclass').count()
train_data[['Pclass', 'Survived']].groupby('Pclass').sum()
plt = train_data[['Pclass', 'Survived']].groupby('Pclass').mean().Survived.plot('bar')
plt.set_xlabel('Pclass')
plt.set_ylabel('Survival Probability')
plt = train_data.Sex.value_counts().sort_index().plot('bar')
plt.set_xlabel('Sex')
plt.set_ylabel('Passenger Count')
plt = train_data[['Sex', 'Survived']].groupby('Sex').mean().Survived.plot('bar')
plt.set_xlabel('Sex')
plt.set_ylabel('Survival Probability')
plt = train_data.Embarked.value_counts().sort_index().plot('bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Passenger Count')
plt = train_data[['Embarked', 'Survived']].groupby('Embarked').mean().Survived.plot('bar')
plt.set_xlabel('Embarked')
plt.set_ylabel('Survival Probability')
plt = train_data.SibSp.value_counts().sort_index().plot('bar')
plt.set_xlabel('Sibling/Spouse')
plt.set_ylabel('Passenger Count')
plt = train_data[['SibSp', 'Survived']].groupby('SibSp').mean().Survived.plot('bar')
plt.set_xlabel('Sibling/Spouse')
plt.set_ylabel('Survival Probability')
plt = train_data.Parch.value_counts().sort_index().plot('bar')
plt.set_xlabel('Parent/Children')
plt.set_ylabel('Passenger Count')
plt = train_data[['Parch', 'Survived']].groupby('Parch').mean().Survived.plot('bar')
plt.set_xlabel('Parent/Children')
plt.set_ylabel('Survival Probability')
sns.catplot('Pclass', col = 'Embarked', data = train_data, kind = 'count')
sns.catplot('Sex', col = 'Pclass', data = train_data, kind = 'count')
sns.catplot('Sex', col = 'Embarked', data = train_data, kind = 'count')
train_data.head()
train_data['Family_Size'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data.head()
train_data = train_data.drop(columns = ['PassengerId', 'Ticket', 'Cabin'])
train_data.head()
train_data['Sex'] = train_data['Sex'].map({'male':0, 'female':1})
train_data['Embarked'] = train_data['Embarked'].map({'C':0, 'Q':1, 'S':2})
train_data.head()
train_data['Title'] = train_data.Name.str.extract('([A-Za-z]+)\.', expand = False)
train_data = train_data.drop(columns = 'Name')
train_data.Title.value_counts().plot('bar')
train_data['Title'] = train_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer',
                                                   'Capt', 'Lady', 'Don'], 'Others')
train_data['Title'] = train_data['Title'].replace(['Ms', 'Mlle'], 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')


plt = train_data.Title.value_counts().sort_index().plot('bar')
plt.set_xlabel('Title')
plt.set_ylabel('Passenger Count')
plt = train_data[['Title', 'Survived']].groupby('Title').mean().Survived.plot('bar')
plt.set_xlabel('Title')
plt.set_ylabel('Survival Probability')
train_data['Title'] = train_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})
train_data.head()
corr_matrix = train_data.corr()

import matplotlib.pyplot as plt
plt.figure(figsize = (9, 8))
sns.heatmap(data = corr_matrix, cmap='BrBG', annot = True, linewidths = 0.2)
train_data.isnull().sum()
train_data['Embarked'].isnull().sum()
train_data['Embarked'] = train_data['Embarked'].fillna(2)
train_data['Embarked'].isnull().sum()
corr_matrix = train_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']].corr()

plt.figure(figsize=(7, 6))
sns.heatmap(data = corr_matrix,cmap='BrBG', annot=True, linewidths=0.2)
Nan_indexes = train_data['Age'][train_data['Age'].isnull()].index

for i in Nan_indexes:
    pred_age = train_data['Age'][((train_data.SibSp == train_data.iloc[i]["SibSp"])
                                  & (train_data.Parch == train_data.iloc[i]["Parch"])
                                  & (train_data.Pclass == train_data.iloc[i]["Pclass"]))].median()
    if not np.isnan(pred_age):
        train_data['Age'].iloc[i] = pred_age
    else:
        train_data['Age'].iloc[i] = train_data['Age'].median()
train_data.isnull().sum()
test_data.head()
test_data.isnull().sum()
test_data = test_data.drop(columns = ['Ticket', 'PassengerId', 'Cabin'])
test_data.head()
test_data['Sex'] = test_data['Sex'].map({'male':0, 'female':1})
test_data['Embarked'] = test_data['Embarked'].map({'C':0, 'Q':1, 'S':2})
test_data.head()
test_data['Title'] = test_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test_data = test_data.drop(columns='Name')

test_data['Title'] = test_data['Title'].replace(['Dr', 'Rev', 'Col', 'Major', 'Countess', 'Sir', 'Jonkheer', 'Lady', 'Capt', 'Don'], 'Others')
test_data['Title'] = test_data['Title'].replace('Ms', 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')
test_data['Title'] = test_data['Title'].replace('Mlle', 'Miss')

test_data['Title'] = test_data['Title'].map({'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Others':4})
test_data.head()
test_data.isnull().sum()
NaN_indexes = test_data['Age'][test_data['Age'].isnull()].index

for i in NaN_indexes:
    pred_age = train_data['Age'][((train_data.SibSp == test_data.iloc[i]["SibSp"]) & (train_data.Parch == test_data.iloc[i]["Parch"]) & (test_data.Pclass == train_data.iloc[i]["Pclass"]))].median()
    if not np.isnan(pred_age):
        test_data['Age'].iloc[i] = pred_age
    else:
        test_data['Age'].iloc[i] = train_data['Age'].median()
title_mode = train_data.Title.mode()[0]
test_data.Title = test_data.Title.fillna(title_mode)
fare_mean = train_data.Fare.mean()
test_data.Fare = test_data.Fare.fillna(fare_mean)
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data.head()
test_data.isnull().sum()
train_data.head()
X_train = train_data.drop(columns = 'Survived')
y_train = train_data.Survived
y_train = pd.DataFrame({'Survived': y_train.values})
X_test = test_data
X_train.head()
y_train.head()
X_train.shape
y_train.shape
X_test.head()
X_test.shape
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

test_data = pd.read_csv('../input/test.csv')
clf = LogisticRegression()
clf.fit(X_train, np.ravel(y_train))

y_pred_log_reg = clf.predict(X_test)
y_pred_log_reg = pd.DataFrame(y_pred_log_reg)
y_pred_log_reg.columns = ['Survived']
log_reg_pred = pd.DataFrame()
log_reg_pred['PassengerId'] = test_data.PassengerId
log_reg_pred['Survived'] = y_pred_log_reg.Survived
log_reg_pred.head()
log_reg_pred.to_csv('Logistic_regression_prediction.csv', index = False)

acc_log_reg = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy on train data: %i %% \n"%acc_log_reg)

print("Accuracy on test data: 72.727 %")
clf = SVC()
clf.fit(X_train, np.ravel(y_train))

y_pred_svc = clf.predict(X_test)
y_pred_svc = pd.DataFrame(y_pred_svc)
y_pred_svc.columns = ['Survived']
svm_pred = pd.DataFrame()
svm_pred['PassengerId'] = test_data.PassengerId
svm_pred['Survived'] = y_pred_svc.Survived
svm_pred.head()
svm_pred.to_csv('svm_prediction.csv', index = False)

acc_svm = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy on train data: %i %% \n"%acc_svm)

print('Accuracy on test data: 62.2 %')
clf = LinearSVC()
clf.fit(X_train, np.ravel(y_train))
y_pred_linear_svc = clf.predict(X_test)
y_pred_linear_svc = pd.DataFrame(y_pred_linear_svc)
y_pred_linear_svc.columns = ['Survived']
linear_svm_pred = pd.DataFrame()
linear_svm_pred['PassengerId'] = test_data.PassengerId
linear_svm_pred['Survived'] = y_pred_linear_svc.Survived
linear_svm_pred.head()
linear_svm_pred.to_csv('linear_svm_prediction.csv', index = False)

acc_linear_svm = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy on train data: %i %% \n"%acc_linear_svm)

print('Accuracy on test data: 45.454%')
sgd = linear_model.SGDClassifier()
sgd.fit(X_train, np.ravel(y_train))

y_pred = sgd.predict(X_test)
y_pred = pd.DataFrame(y_pred)
y_pred.columns = ['Survived']
sgd_pred = pd.DataFrame()
sgd_pred['PassengerId'] = test_data.PassengerId
sgd_pred['Survived'] = y_pred.Survived

sgd_pred.head()
sgd_pred.to_csv('sgd_prediction.csv', index = False)

acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)
print ("Accuracy on train data: %i %% \n"%acc_sgd)

print('Accuracy on test data: 65.55%')
clf = RandomForestClassifier()
clf.fit(X_train, np.ravel(y_train))

y_pred_randomforest = clf.predict(X_test)
y_pred_randomforest = pd.DataFrame(y_pred_randomforest)
y_pred_randomforest.columns = ['Survived']
randomforest_pred = pd.DataFrame()
randomforest_pred['PassengerId'] = test_data.PassengerId
randomforest_pred['Survived'] = y_pred_randomforest.Survived

randomforest_pred.head()
randomforest_pred.to_csv('RandomForest_prediction.csv', index = False)

acc_randomforest = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy on train data: %i %% \n"%acc_randomforest)

print('Accuracy on test data: 53.11%')
clf = KNeighborsClassifier()
clf.fit(X_train, np.ravel(y_train))

y_pred_knn = clf.predict(X_test)
y_pred_knn = pd.DataFrame(y_pred_knn)
y_pred_knn.columns = ['Survived']
knn_pred = pd.DataFrame()
knn_pred['PassengerId'] = test_data.PassengerId
knn_pred['Survived'] = y_pred_knn.Survived
knn_pred.head()
knn_pred.to_csv('knn_prediction.csv', index = False)

acc_knn = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy on train data: %i %% \n"%acc_knn)

print('Accuracy on test data: 60.765%')
clf = DecisionTreeClassifier()
clf.fit(X_train, np.ravel(y_train))

y_pred_decision_tree = clf.predict(X_test)
y_pred_decision_tree = pd.DataFrame(y_pred_decision_tree)
y_pred_decision_tree.columns = ['Survived']
decision_tree_pred = pd.DataFrame()
decision_tree_pred['PassengerId'] = test_data.PassengerId
decision_tree_pred['Survived'] = y_pred_decision_tree.Survived

decision_tree_pred.head()
decision_tree_pred.to_csv('decision_tree_prediction.csv', index = False)

acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy on train data: %i %% \n"%acc_decision_tree)

print('Accuracy on test data: 44.497 %')
clf = GaussianNB()
clf.fit(X_train, np.ravel(y_train))

y_pred_gnb = clf.predict(X_test)
y_pred_gnb = pd.DataFrame(y_pred_gnb)
y_pred_gnb.columns = ['Survived']
gnb_pred = pd.DataFrame()
gnb_pred['PassengerId'] = test_data.PassengerId
gnb_pred['Survived'] = y_pred_gnb.Survived
gnb_pred.head()
gnb_pred.to_csv('GaussianNB_prediction.csv', index = False)

acc_gnb = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy on train data: %i %% \n"%acc_gnb)

print('Accuracy on test data: 71.291%')
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_decision_tree_training_set = clf.predict(X_train)
acc_decision_tree = round(clf.score(X_train, y_train) * 100, 2)
print ("Accuracy on train data: %i %% \n"%acc_decision_tree)

class_names = ['Survived', 'Not Survived']

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_train, y_pred_decision_tree_training_set)
np.set_printoptions(precision=2)

print ('Confusion Matrix in Numbers')
print (cnf_matrix)
print ('')

cnf_matrix_percent = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

print ('Confusion Matrix in Percentage')
print (cnf_matrix_percent)
print ('')

true_class_names = ['True Survived', 'True Not Survived']
predicted_class_names = ['Predicted Survived', 'Predicted Not Survived']

df_cnf_matrix = pd.DataFrame(cnf_matrix, 
                             index = true_class_names,
                             columns = predicted_class_names)

df_cnf_matrix_percent = pd.DataFrame(cnf_matrix_percent, 
                                     index = true_class_names,
                                     columns = predicted_class_names)

plt.figure(figsize = (15,5))

plt.subplot(121)
sns.heatmap(df_cnf_matrix, annot=True, fmt='d')

plt.subplot(122)
sns.heatmap(df_cnf_matrix_percent, annot=True)
plt.xkcd()
models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Support Vector Machines', 'Linear SVC', 
              'KNN', 'Decision Tree', 'Random Forest', 'Naive Bayes', 'Stochastic Gradient Decent'],
    
    'Train_Score': [acc_log_reg, acc_svm, acc_linear_svm, 
              acc_knn,  acc_decision_tree, acc_randomforest, acc_gnb, acc_sgd],
    
    'Test_Score': [ 72.727, 62.2, 45.454, 
              60.765,  44.497, 53.11, 71.291, 65.55]
    })
models.sort_values(by='Train_Score', ascending=False)
models.sort_values(by='Test_Score', ascending=False)
plt.plot(models['Model'], models['Train_Score'], color='g', label = 'Train Accuracy Score')
plt.plot(models['Model'], models['Test_Score'], color='orange', label = 'Test Accuracy Score')
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.axis()
plt.title('Titanic ML Model Performance')
plt.legend(loc = 'center left', bbox_to_anchor=(1, 0.5))
plt.xticks(rotation=90)
plt.show()

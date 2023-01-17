import numpy as np
import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline 
sns.set(color_codes=True)
train = pd.read_csv("/kaggle/input/titanic/train.csv")
test = pd.read_csv("/kaggle/input/titanic/test.csv")
merged = train.append(test)
merged.info()
train.head()
test.head()
train.count()
test.count()
train.isnull().sum()
test.isnull().sum()
merged = merged.drop(['Cabin', 'Ticket', 'Name'], axis = 1)
train = train.drop(['Cabin', 'Ticket', 'Name'], axis = 1)
test = test.drop(['Cabin', 'Ticket', 'Name'], axis = 1)
plt.figure(figsize = (12,8))
sns.boxplot(x=train['Age'])
train['Age'].mean()
plt.figure(figsize = (12,8))
sns.boxplot(x=train['Fare'])
print(train[train.Fare == train.Fare.max()])
correlation = train.corr()
correlation
sns.heatmap(correlation, annot=True)
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Pclass', y = 'Survived', data=train)
plt.show()
sns.barplot(x='Survived', y = 'Fare', data=train)
plt.show()
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
sns.barplot(x='Sex', y = 'Survived', data=train)
plt.show()
sns.scatterplot(x="Age", y="Survived", hue="Sex", data=train);
print(train.isnull().sum())
print('*'* 40)
print(test.isnull().sum())
print('*'*40)
merged.isnull().sum()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(test[['Fare']])
test['Fare'] = imputer.transform(test[['Fare']])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(merged[['Fare']])
merged['Fare'] = imputer.transform(merged[['Fare']])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(train[['Age']])
train['Age'] = imputer.transform(train[['Age']])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(test[['Age']])
test['Age'] = imputer.transform(test[['Age']])
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='median')
imputer.fit(merged[['Age']])
merged['Age'] = imputer.transform(merged[['Age']])
train['Embarked'].value_counts()
train['Embarked'].fillna('S', inplace = True)
merged['Embarked'].fillna('S', inplace = True)
#Checking to make sure all values are filled
print(train.isnull().sum())
print('*'* 40)
print(test.isnull().sum())
print('*'*40)
merged.isnull().sum()
train.head()
train.select_dtypes(include=[object])
train.head(3)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
train = train.apply(le.fit_transform)
train.head()
test.select_dtypes(include=[object])
test.head(3)
le = preprocessing.LabelEncoder()
test = test.apply(le.fit_transform)
test.head()
le = preprocessing.LabelEncoder()
merged = merged.apply(le.fit_transform)
merged.head()
#Splitting the dependent variable from independent
x = train.drop(['Survived', 'PassengerId'], axis=1)
y = train['Survived']
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,
                                                    random_state = 1)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(random_state = 0)
lr.fit(x_train, y_train)
#predicting the test set results
y_pred = lr.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), 1))
#Making the confusion matrix. How many mistakes/correct predictions model made
from sklearn.metrics import confusion_matrix
lr_cm = confusion_matrix(y_test, y_pred)
print(lr_cm)
#Calculating accuracy on test set
from sklearn.metrics import accuracy_score
LR_Accuracy = accuracy_score(y_test, y_pred)
print('Logistic Regression Accuracy:', LR_Accuracy * 100)
from sklearn.tree import DecisionTreeClassifier
dec_tree_class = DecisionTreeClassifier(criterion = 'entropy')
dec_tree_class.fit(x_train, y_train)
#predicting the test set results
y_pred = dec_tree_class.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), 1))
#Making the confusion matrix. How many mistakes/correct predictions model made
from sklearn.metrics import confusion_matrix
dec_tree_cm = confusion_matrix(y_test, y_pred)
print(dec_tree_cm)
#Calculating accuracy on test set
from sklearn.metrics import accuracy_score
dec_tree_accuracy = accuracy_score(y_test, y_pred)
print('Decision Tree Accuracy:', dec_tree_accuracy * 100)
from sklearn.ensemble import RandomForestClassifier
RF_class = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
RF_class.fit(x_train, y_train)
y_pred = RF_class.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), 1))
from sklearn.metrics import confusion_matrix
RF_cm = confusion_matrix(y_test, y_pred)
print(RF_cm)
from sklearn.metrics import accuracy_score
RF_Accuracy = accuracy_score(y_test, y_pred)
print('Random Forest:', RF_Accuracy * 100)
from sklearn.svm import SVC
SV_class = SVC(kernel='rbf', random_state = 0)
SV_class.fit(x_train, y_train)
y_pred = SV_class.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), 1))
from sklearn.metrics import confusion_matrix
SV_class_cm = confusion_matrix(y_test, y_pred)
print(SV_class_cm)
from sklearn.metrics import accuracy_score
SV_class_accuracy = accuracy_score(y_test, y_pred)
print('Kernel Support Vector:', SV_class_accuracy * 100)
from sklearn.svm import SVC
SV_classifier = SVC(kernel = 'linear', random_state = 0)
SV_classifier.fit(x_train, y_train)
y_pred = SV_classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), 1))
from sklearn.metrics import confusion_matrix
SV_cm = confusion_matrix(y_test, y_pred)
print(SV_cm)
from sklearn.metrics import accuracy_score
SV_accuracy = accuracy_score(y_test, y_pred)
print('Support Vector:', SV_accuracy * 100)
from sklearn.naive_bayes import GaussianNB
NB_classifier = GaussianNB()
NB_classifier.fit(x_train, y_train)
y_pred = NB_classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), 1))
from sklearn.metrics import confusion_matrix
NB_cm = confusion_matrix(y_test, y_pred)
print(NB_cm)
from sklearn.metrics import accuracy_score
NB_accuracy = accuracy_score(y_test, y_pred)
print('Gaussian Naive Bayes:', NB_accuracy * 100)
from sklearn.neighbors import KNeighborsClassifier
KN_classifier = KNeighborsClassifier(n_neighbors = 5, p=2, metric='minkowski')
KN_classifier.fit(x_train, y_train)
y_pred = KN_classifier.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.values.reshape(len(y_test), 1)), 1))
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)
from sklearn.metrics import accuracy_score
KN_accuracy = accuracy_score(y_test, y_pred)
print('KNeighbors Accuracy:', KN_accuracy * 100)
print('Logistic Regression Accuracy:', LR_Accuracy * 100)
print('Decision Tree Accuracy:', dec_tree_accuracy * 100)
print('Random Forest:', RF_Accuracy * 100)
print('Kernel Support Vector:', SV_class_accuracy * 100)
print('Support Vector:', SV_accuracy * 100)
print('Gaussian Naive Bayes:', NB_accuracy * 100)
print('KNeighbors Accuracy:', KN_accuracy * 100)
ids = test['PassengerId']
predictions = lr.predict(test.drop('PassengerId', axis=1))
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})
submission
submission.to_csv('submission.csv', index=False)
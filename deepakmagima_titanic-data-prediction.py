import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# ML Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
train = pd.read_csv('../input/train.csv')
train.shape
train.dtypes
train.describe()
train.isnull().any()
train.head()
sns.heatmap(train.corr(),annot = True,cmap="YlGnBu");
train['Family'] = train['Parch'] + train['SibSp']
train.head()
train.Sex.value_counts()
train.Embarked.value_counts()
train.info()
sns.countplot(x='Survived',data=train)#no of non survived persons are more
sns.countplot(x='Sex',data=train);
sns.factorplot(x='Survived',col='Sex',kind='count',data=train);
sns.barplot(x='Pclass', y='Survived', data=train);
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
print("Percent of Female Survived: ",train[train.Sex=='female'].Survived.sum()/train[train.Sex=='female'].Survived.count())
print("Percent of Male Survived: ",train[train.Sex=='male'].Survived.sum()/train[train.Sex=='male'].Survived.count())
#Filling Missing Values

train.Age.fillna(train.Age.mean(),inplace=True)
train['Embarked']=train.Embarked.fillna('S')
#Categoical Value

train['Embarked']=train.Embarked.map({'S':1,'C':2,'Q':3})
train['Sex']=train.Sex.map({'female':0,'male':1})

train['Cabin'] = np.where(train['Cabin'].notnull(), 1, 0)
train.isnull().any()
train.drop(['Name','Ticket'],axis=1,inplace=True)
train.drop(['PassengerId'], axis=1,inplace=True)
train['Fare']=train.Fare.fillna(train.Fare.median())
pd.get_dummies(train, columns = ["Pclass",'Sex','Cabin','Embarked']).head()
train.head()
test = pd.read_csv('../input/test.csv')
test.head()
test.shape
test.describe()
test.isnull().any()
test['Family'] = test['Parch'] + test['SibSp']
#Filling Missing Values

test.Age.fillna(test.Age.mean(),inplace=True)
test['Embarked']=test.Embarked.fillna('S')
#Categoical Value

test['Embarked']=test.Embarked.map({'S':1,'C':2,'Q':3})
test['Sex']=test.Sex.map({'female':0,'male':1})

test['Cabin'] = np.where(test['Cabin'].notnull(), 1, 0)
test.drop(['Name','Ticket'],axis=1,inplace=True)
test['Fare']=test.Fare.fillna(test.Fare.median())
test.head()
X_new = test.iloc[:,1:].values
X_train = train.drop('Survived',axis=1).values
y_train = train.Survived.values
X_test = test.drop("PassengerId", axis=1).copy().values
#Fitting Logistic Regression
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state =0)
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
acc_log = round(clf.score(X_train, y_train) * 100, 2)
print(round(acc_log,2,), "%")
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting logistic regression to training set
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state = 0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc_log_s = round(clf.score(X_train, y_train) * 100, 2)
print(round(acc_log,2,), "%")
# Random Forest
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train, y_train)

Y_prediction = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
acc_random_forest = round(random_forest.score(X_train, y_train) * 100, 2)
print(round(acc_random_forest,2,), "%")
# KNN
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, y_train)

Y_pred = knn.predict(X_test)

acc_knn = round(knn.score(X_train, y_train) * 100, 2)
print(round(acc_knn,2,), "%")
# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(X_train, y_train)

Y_pred = gaussian.predict(X_test)

acc_gaussian = round(gaussian.score(X_train, y_train) * 100, 2)
print(round(acc_gaussian,2,), "%")
# Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, y_train)

Y_pred = perceptron.predict(X_test)

acc_perceptron = round(perceptron.score(X_train, y_train) * 100, 2)
print(round(acc_perceptron,2,), "%")
# Linear SVC
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)

Y_pred = linear_svc.predict(X_test)

acc_linear_svc = round(linear_svc.score(X_train, y_train) * 100, 2)
print(round(acc_linear_svc,2,), "%")
# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

Y_pred = decision_tree.predict(X_test)

acc_decision_tree = round(decision_tree.score(X_train, y_train) * 100, 2)
print(round(acc_decision_tree,2,), "%")
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, y_train)
Y_pred = sgd.predict(X_test)

sgd.score(X_train, y_train)

acc_sgd = round(sgd.score(X_train, y_train) * 100, 2)


print(round(acc_sgd,2,), "%")
results = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 
              'Logistic Regression (Feature Scaling)', 'Stochastic Gradient Decent',
              'Decision Tree'],
    'Score': [acc_linear_svc, acc_knn, acc_log, 
              acc_random_forest, acc_gaussian, acc_perceptron, 
              acc_log_s, acc_sgd, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df
result_df.reset_index().plot.bar();
from sklearn.model_selection import cross_val_score
rf = RandomForestClassifier(n_estimators=100)
scores = cross_val_score(rf, X_train, y_train, cv=10, scoring = "accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
from sklearn.ensemble import ExtraTreesClassifier
tree_clf = ExtraTreesClassifier()
tree_clf.fit(X_train, y_train)

tree_clf.feature_importances_
importances = tree_clf.feature_importances_
feature_names = train.iloc[:, 1:].columns.tolist()
feature_names
feature_imp_dir = dict(zip(feature_names, importances))
features = sorted(feature_imp_dir.items(), key=lambda x: x[1], reverse=True)
feature_imp_dir
important_features = pd.DataFrame.from_dict(feature_imp_dir, orient='index',columns=['Importance'])
important_features = important_features.sort_values('Importance',ascending=False)
important_features.plot.bar();
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
predictions = cross_val_predict(random_forest, X_train, y_train, cv=3)
confusion_matrix(y_train, predictions)
from sklearn.metrics import precision_score, recall_score

print("Precision:", precision_score(y_train, predictions))
print("Recall:",recall_score(y_train, predictions))
##F-Score

from sklearn.metrics import f1_score
f1_score(y_train, predictions)
titanic_submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": Y_prediction
    })
titanic_submission.to_csv('titanic_submission.csv', index=False)
titanic_submission.head()
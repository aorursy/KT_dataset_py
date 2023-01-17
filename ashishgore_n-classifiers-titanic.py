import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

train.head()

test.head()
train.info()
# %Null values in Train set
train.isnull().sum()/len(train)
train['Age'].value_counts()
# Feature Name,Ticket & Cabin does not add any value to the dataset. So we will drop these features.
train.drop(['Name','Ticket','Cabin'],axis=1,inplace=True)
train['Age'].describe()
train.drop(['Age'],axis=1,inplace=True)
train['Embarked'].value_counts()
train['Embarked'].fillna('S',inplace=True)
# %Null values in Train set
train.isnull().sum()/len(train)
train.info()
sns.set_style('whitegrid')
sns.countplot(data= train, x= 'Survived',hue= 'Sex')
sns.countplot(train['Pclass'])
sns.countplot(train['SibSp'])
sns.countplot(train['Parch'])
train['n_family_member'] = train['SibSp'] + train['Parch']
sns.countplot(data= train,x='n_family_member',hue = 'Sex')
train.drop(['SibSp','Parch'],axis=1,inplace=True)
train.info()
train['Sex_male'] = pd.get_dummies(data=train['Sex'],drop_first=True)
train.head()
train = pd.get_dummies(train, columns=['Embarked'],drop_first=True, prefix='Embarked')
train.info()
train.head()
sns.distplot(train['Fare'])
train['Fare'] = train['Fare'].astype('int')
train.info()
train.head()
train.columns
X = train[['Pclass', 'Fare', 'n_family_member','Sex_male', 'Embarked_Q', 'Embarked_S']]
y = train['Survived']
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, stratify = y)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(accuracy_score(y_test,y_pred))
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(accuracy_score(y_test,y_pred))
from sklearn.svm import SVC
svc = SVC()
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(accuracy_score(y_test,y_pred))
from sklearn.neighbors import KNeighborsClassifier as KNN
knn = KNN(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test,y_pred))
SEED = 101
lr = LogisticRegression(random_state=SEED)
knn = KNN()
dt = DecisionTreeClassifier(random_state=SEED)

classifiers = [('Logistic Regression', lr),
('K Nearest Neighbours', knn),
('Classification Tree', dt)]

for clf_name, clf in classifiers:
   
    clf.fit(X_train, y_train)
  
    y_pred = clf.predict(X_test)
  
    print('{:s} : {:.3f}'.format(clf_name, accuracy_score(y_test, y_pred)))
from sklearn.ensemble import VotingClassifier
vc = VotingClassifier(estimators=classifiers)

vc.fit(X_train, y_train)

y_pred = vc.predict(X_test)

print('Voting Classifier: {:.3f}'.format(accuracy_score(y_test, y_pred)))
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=SEED)
from sklearn.model_selection import GridSearchCV

params_dt = {
'max_depth': [3, 4,5, 6],
'min_samples_leaf': [0.04, 0.06, 0.08],
'max_features': [0.2, 0.4,0.6, 0.8]
}

grid_dt = GridSearchCV(estimator=dt,
param_grid=params_dt,
scoring='accuracy',
cv=5,
n_jobs=-1)
# Fit 'grid_dt' to the training data
grid_dt.fit(X_train, y_train)
best_hyperparams = grid_dt.best_params_
print('Best hyerparameters:\n', best_hyperparams)
best_CV_score = grid_dt.best_score_
print('Best CV accuracy: {:.3f}'.format(best_CV_score))
best_model = grid_dt.best_estimator_

test_acc = best_model.score(X_test,y_test)

print("Test set accuracy of best model: {:.3f}".format(test_acc))
import xgboost as xgb
xg_cl = xgb.XGBClassifier(objective='binary:logistic',
n_estimators=10, seed=101)
xg_cl.fit(X_train, y_train)
preds = xg_cl.predict(X_test)
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))
# As XGBoost Model shows better Accuracy lets test it on test set.
# Lets Transform test set
train.head()
test.head()
test.drop(['Name','Ticket','Cabin','Age'],axis=1,inplace=True)
test.info()
test['n_family_member'] = test['SibSp'] + test['Parch']
test.drop(['SibSp','Parch'],axis=1,inplace=True)
test = pd.get_dummies(test, columns=['Sex'],drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'],drop_first=True, prefix='Embarked')
test['Fare'] = test['Fare'].fillna(0)
test['Fare'] = test['Fare'].astype('int')
X_test_test = test[['Pclass', 'Fare', 'n_family_member','Sex_male', 'Embarked_Q', 'Embarked_S']]
preds = xg_cl.predict(X_test_test)
submission = pd.DataFrame({'PassengerId':test.PassengerId,'Survived':preds})
submission.Survived = submission.Survived.astype(int)
print(submission.shape)
filename = 'Titanic Predictions.csv'
submission.to_csv(filename,index=False)
print('Saved file: ' + filename)

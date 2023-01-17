import numpy as np 
import pandas as pd 
from sklearn.preprocessing import LabelEncoder  #for encoding the data to int values
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
train_data = pd.read_csv('../input/titanic/train.csv')
test_data  = pd.read_csv('../input/titanic/test.csv')
# checking train data first 5 rows
train_data.head() 
train_data = train_data.drop(['Ticket', 'Cabin','PassengerId','Name'], axis=1)
test_data = test_data.drop(['Ticket', 'Cabin','PassengerId','Name'], axis=1)
print("Training data :","\n\n",train_data.isnull().sum(),"\n")
print('_'*10,"Testing data",'-'*10)
test_data.isnull().sum()
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].median())
train_data.Fare.fillna(train_data.Fare.mean(), inplace=True)
test_data.Fare.fillna(test_data.Fare.mean(), inplace=True)
train_data.Embarked.fillna('S', inplace=True)
embarked_dummies = pd.get_dummies(train_data['Embarked'], prefix='Embarked')
train_data = pd.concat([train_data, embarked_dummies], axis=1)
train_data.drop('Embarked', axis=1, inplace=True)

test_data.Embarked.fillna('S', inplace=True)
embarked_dummies1 = pd.get_dummies(test_data['Embarked'], prefix='Embarked')
test_data = pd.concat([test_data, embarked_dummies1], axis=1)
test_data.drop('Embarked', axis=1, inplace=True)
print("Training data :","\n\n",train_data.isnull().sum(),"\n")
print('_'*10,"Testing data",'-'*10)
test_data.isnull().sum()
encounder=LabelEncoder()
train_data['Sex']=encounder.fit_transform(train_data['Sex'].values)
test_data['Sex']=encounder.fit_transform(test_data['Sex'].values)

train_data.head()
# test_data.head()
train_data.describe()
train_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_data.head()
test_data.head()
train= train_data
train = train.drop(['Survived'], axis=1)


clf = RandomForestClassifier(n_estimators=50, max_features='sqrt')
clf = clf.fit(train, train_data['Survived'])
model = SelectFromModel(clf, prefit=True)
train_reduced = model.transform(train)
print (train_reduced.shape)

test_reduced = model.transform(test_data)
print (test_reduced.shape)
logreg = LogisticRegression()
logreg_cv = LogisticRegressionCV()
rf = RandomForestClassifier()
gboost = GradientBoostingClassifier()

models = [logreg, logreg_cv, rf, gboost]
for model in models:
    print ('Cross-validation of : {0}'.format(model.__class__))
    score = compute_score(clf=model, X=train_reduced, y=targets, scoring='accuracy')
    print ('CV score = {0}'.format(score))
    print ('****')
y=train_data['Survived']
X = train
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
print(scaler.fit(X,y))
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size=0.3,
                                                    random_state=10)
# Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(max_depth=1,
                                    max_features=1,
                                    n_estimators = 9,
                                    random_state = 13,
    criterion='gini',
    min_samples_split=2,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0.0,
#     max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    verbose=0,
    warm_start=False,
    class_weight=None)
#-------------------------------------------------------------------------------------------------------
classifier.fit(X_train, Y_train)
# ---------------------------------------------------------------------------------------------------------------------
Y_pred = classifier.predict(X_test)
# ----------------------------------------------------------------------------------------------------------------------
from sklearn.metrics import accuracy_score, classification_report
print("                  Accuracy score=",accuracy_score(Y_test, Y_pred)*100,"%") #most accurate
# ------------------------------------------------------------------------------------------------------------------------
from sklearn.metrics import confusion_matrix #confusuon matrix for randomforest
pd.crosstab(Y_test, Y_pred)
from sklearn.linear_model import LogisticRegression
classifier1 = LogisticRegression(random_state = 0)
classifier1.fit(X_train, Y_train)
#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier2 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier2.fit(X_train, Y_train)
#Using SVC method of svm class to use Support Vector Machine Algorithm
from sklearn.svm import SVC
classifier3 = SVC(kernel = 'linear', random_state = 0)
classifier3.fit(X_train, Y_train)
# Using SVC method of svm class to use Kernel SVM Algorithm
from sklearn.svm import SVC
classifier4 = SVC(kernel = 'rbf', random_state = 1)
classifier4.fit(X_train, Y_train)
#Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
classifier5 = GaussianNB()
classifier5.fit(X_train, Y_train)
#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
classifier6 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier6.fit(X_train, Y_train)
Y_pred1 = classifier1.predict(X_test)
Y_pred2 = classifier2.predict(X_test)
Y_pred3 = classifier3.predict(X_test)
Y_pred4 = classifier4.predict(X_test)
Y_pred5= classifier5.predict(X_test)
Y_pred6 = classifier6.predict(X_test)
print(accuracy_score(Y_test, Y_pred1))
print(accuracy_score(Y_test, Y_pred2))
print(accuracy_score(Y_test, Y_pred3))
print(accuracy_score(Y_test, Y_pred4))
print(accuracy_score(Y_test, Y_pred5))
print(accuracy_score(Y_test, Y_pred6))
pred = classifier1.predict(test_data)
u=pd.DataFrame(pred)
u.c
u.to_csv('mycsvfile.csv',index=False)

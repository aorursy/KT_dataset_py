%matplotlib inline
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
data = pd.read_csv("../input/train.csv")
data.head()
test_data = pd.read_csv("../input/test.csv")
test_data.head()
data.shape
data.describe()
data.info()
null_columns = data.columns[data.isnull().any()]
null_columns
data.isnull().sum()
data.hist(bins=10,figsize=(10,10),grid=False)
sns.barplot(x="Embarked", y="Survived", hue="Pclass", data=data);

sns.barplot(x="Embarked", y="Survived", hue="Sex", data=data);

sns.pointplot(x="Pclass", y="Survived",hue="Sex", data=data, palette={"male":"blue", "female":"pink"}, markers=["*", "o"], 
             linestyles=["-", "--"])
data.corr()
data.corr()["Survived"]
data[data["Embarked"].isnull()]
def categorize_Age(data):
    data.Age = data.Age.fillna(-0.5)
    bins = (-1, 0, 5, 12, 18, 25, 35, 60, 120)
    group_names = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
    categories = pd.cut(data.Age, bins, labels=group_names)
    data.Age = categories
    return data

def simplify_cabin(data):
    data.Cabin = data.Cabin.fillna("N")
    data.Cabin = data.Cabin.apply(lambda x: x[0])
    return data

def simplify_fares(data):
    data.Fare = data.Fare.fillna(-0.5)
    bins = (-1, 0, 8, 15, 31, 1000)
    group_names = ['Unknown', '1_quartile', '2_quartile', '3_quartile', '4_quartile']
    categories = pd.cut(data.Fare, bins, labels=group_names)
    data.Fare = categories
    return data

def drop_features(data):
    return data.drop(["Ticket", "Name", "Embarked"], axis=1)

def transform_features(data):
    data = categorize_Age(data)
    data = simplify_cabin(data)
    data = simplify_fares(data)
    data = drop_features(data)
    return data

data = transform_features(data)
test_data = transform_features(test_data)
data.head()
from sklearn import preprocessing

def encode_features(data, test_data):
    features = ['Fare', 'Cabin', 'Age', 'Sex']
    data_combined = pd.concat([data[features], test_data[features]])
    
    for feature in features:
        le = preprocessing.LabelEncoder()
        le = le.fit(data_combined[feature])
        data[feature] = le.transform(data[feature])
        test_data[feature] = le.transform(test_data[feature])
    return data, test_data

data, test_data = encode_features(data, test_data)
data.head()
from sklearn.model_selection import train_test_split

X = data.drop(['PassengerId', 'Survived'], axis=1)
Y = data['Survived']

trainx, testx, trainy, testy = train_test_split(X, Y, test_size=0.2, random_state=0)

trainx.head()
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

clf = RandomForestClassifier()
parameters = {'n_estimators': [2,5,10],
             'criterion': ['gini', 'entropy'],
             'max_features': ['log2', 'sqrt', 'auto'],
             'max_depth': [2,3,5,10],
             'min_samples_split': [2,3,5],
             'min_samples_leaf': [1,5,8]}

acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf, parameters, scoring = acc_scorer)
grid_obj = grid_obj.fit(trainx, trainy)

clf = grid_obj.best_estimator_
clf.fit(trainx, trainy)
predictions = clf.predict(testx)
print(accuracy_score(testy, predictions))
from sklearn.cross_validation import KFold

N = 891
def run_kfold(clf):
    kf = KFold(N, n_folds = 10)
    accuracies = []
    fold = 0
    for x,y in kf:
        trainx, testx = X.iloc[x], X.iloc[y]
        trainy, testy = Y.iloc[x], Y.iloc[y]
        clf.fit(trainx, trainy)
        pred = clf.predict(testx)
        accuracies.append(accuracy_score(testy, pred))
    mean_accuracy = np.mean(accuracies)
    print(mean_accuracy)

run_kfold(clf)
ids = test_data['PassengerId']
pred = clf.predict(test_data.drop('PassengerId', axis = 1))
output1= pd.DataFrame({"PassengerId": ids,
                       "Survived": pred
})
output1.head()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit

lr = LogisticRegression(random_state=None)

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=50)

scores = cross_val_score(lr, trainx, trainy, scoring='f1', cv=cv)

print(scores.mean())
run_kfold(lr)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

def run_kfold_1(clf):
    kf = KFold(N, n_folds = 10)
    accuracies = []
    fold = 0
    for x,y in kf:
        trainx, testx = X.iloc[x], X.iloc[y]
        trainy, testy = Y.iloc[x], Y.iloc[y]
        clf.fit(trainx, trainy)
        pred = clf.predict(testx)
        pred[pred > .5] = 1
        pred[pred <=.5] = 0
        accuracies.append(accuracy_score(testy, pred))
    mean_accuracy = np.mean(accuracies)
    print(mean_accuracy)

run_kfold_1(model)
from sklearn.svm import SVC

clf1 = SVC()

parameters = {'C': [1,2,5,10],
             'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
             }

acc_scorer = make_scorer(accuracy_score)

grid_obj = GridSearchCV(clf1, parameters, scoring = acc_scorer)
grid_obj = grid_obj.fit(trainx, trainy)

clf1 = grid_obj.best_estimator_
clf1

cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=None)

scores = cross_val_score(clf1, trainx, trainy, scoring='f1', cv=cv)

print(scores.mean())
run_kfold(clf1)
pred = clf1.predict(test_data.drop("PassengerId", axis=1))
submission= pd.DataFrame({"PassengerId": test_data["PassengerId"],
                         "Survived": pred
                         })

submission.head()
submission.to_csv("titanic_submission.csv", index=False)
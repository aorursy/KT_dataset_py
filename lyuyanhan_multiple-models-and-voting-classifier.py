import pandas as pd

def fetch_data(route):

    data = pd.read_csv(route)

    return data
train = fetch_data("train.csv")

test = fetch_data("test.csv")
test.describe()
median_train = train["Age"].median()

train["Age"] = train["Age"].fillna(median_train)

median_test = test["Age"].median()

test["Age"]=test["Age"].fillna(median_test)

median_test_fare = test["Fare"].median()

test["Fare"]=test["Fare"].fillna(median_test)
data_numeric = ["Age", "SibSp", "Parch", "Fare"]
%matplotlib inline

import matplotlib.pyplot as plt

train[data_numeric].hist(bins=50, figsize=(20,15))

plt.show()
import numpy as np

def age_process(age):

    return np.ceil(age/8)
train["Age"]=train["Age"].apply(age_process)

test["Age"]=test["Age"].apply(age_process)
crosstab = pd.crosstab(train["Age"],train["Survived"])

crosstab_age_by_survive = crosstab.astype('float').div(crosstab.sum(axis=1), axis=0)

crosstab_age_by_survive.plot(kind = "bar")
def fare_process(fare):

    return np.ceil(fare/20)
train["Fare"] = train["Fare"].apply(fare_process)

test["Fare"] = test["Fare"].apply(fare_process)
crosstab_fare = pd.crosstab(train["Fare"],train["Survived"])

crosstab_fare_by_survive = crosstab_fare.astype('float').div(crosstab_fare.sum(axis=1), axis=0)

crosstab_fare_by_survive

crosstab_fare_by_survive.plot(kind = "bar")
crosstab_parch = pd.crosstab(train["Parch"],train["Survived"])

crosstab_parch_by_survive = crosstab_parch.astype('float').div(crosstab_parch.sum(axis=1), axis=0)

crosstab_parch_by_survive.plot(kind = "bar")
crosstab_sibSp = pd.crosstab(train["SibSp"],train["Survived"])

crosstab_sibSp_by_survive = crosstab_sibSp.astype('float').div(crosstab_sibSp.sum(axis=1), axis=0)

crosstab_sibSp_by_survive.plot(kind = "bar")
def family_process(sib, parch):

    return np.ceil(sib+parch)
train["family"] = list(map(family_process,train["SibSp"],train["Parch"]))

test["family"] = list(map(family_process,test["SibSp"],test["Parch"]))
crosstab_fam = pd.crosstab(train["family"],train["Survived"])

crosstab_fam_by_survive =crosstab_fam.astype('float').div(crosstab_fam.sum(axis=1), axis=0)

crosstab_fam_by_survive.plot(kind = "bar")
crosstab_sex = pd.crosstab(train["Sex"],train["Survived"])

crosstab_sex_by_survive =crosstab_sex.astype('float').div(crosstab_sex.sum(axis=1), axis=0)

crosstab_sex_by_survive.plot(kind = "bar")
crosstab_embark = pd.crosstab(train["Embarked"],train["Survived"])

crosstab_embark_by_survive =crosstab_embark.astype('float').div(crosstab_embark.sum(axis=1), axis=0)

crosstab_embark_by_survive.plot(kind = "bar")
crosstab_pclass = pd.crosstab(train["Pclass"],train["Survived"])

crosstab_pclass_by_survive =crosstab_pclass.astype('float').div(crosstab_pclass.sum(axis=1), axis=0)

crosstab_pclass_by_survive.plot(kind = "bar")
data.head(3)
def title_process(name):

    namelist = name.split(',')

    title = namelist[1].split(' ')

    return title[1]    
train["title"] = train["Name"].apply(title_process)

test["title"] = test["Name"].apply(title_process)
pd.crosstab(train['title'], train['Sex'])
def title_reduce(title):

    if (title == "Miss." or title == "Mrs." or title == "Master." 

        or title == "Mr."):

        return title

    else:

        return "Other"
train["title"] = train["title"].apply(title_reduce)

test["title"] = test["title"].apply(title_reduce)
crosstab_title = pd.crosstab(train["title"],train["Survived"])

crosstab_title_by_survive =crosstab_title.astype('float').div(crosstab_title.sum(axis=1), axis=0)

crosstab_title_by_survive.plot(kind = "bar")
train_prepare = train.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis=1)

test_prepare = test.drop(["PassengerId","Name","SibSp","Parch","Ticket","Cabin","Embarked"],axis=1)

train_prepare = pd.get_dummies(train_prepare, columns=["Sex","Pclass", "title"])

test_prepare = pd.get_dummies(test_prepare, columns=["Sex","Pclass", "title"])

train_prepare.head()
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(data_prepare, test_size=0.2, random_state=42)

X_train = train_set.drop("Survived",axis=1)

y_train = train_set["Survived"]

X_test = test_set.drop("Survived",axis=1)

y_test = test_set["Survived"]
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(n_iter=750, penalty=None, eta0=0.1) 

sgd_clf.fit(X_train, y_train)

score = cross_val_score(sgd_clf, X_train, y_train, cv=20, scoring="accuracy")

score.mean()
from sklearn.tree import DecisionTreeClassifier

dtc_clf = DecisionTreeClassifier(random_state=42)

dtc_clf.fit(X_train,y_train)

score = cross_val_score(dtc_clf, X_train, y_train, cv=20, scoring="accuracy")

score.mean()
from sklearn.neighbors import KNeighborsClassifier



knn_clf = KNeighborsClassifier(n_neighbors=10)

knn_clf.fit(X_train,y_train)

score = cross_val_score(knn_clf, X_train, y_train, cv=20, scoring="accuracy")

score.mean()
from sklearn.model_selection import GridSearchCV

param_grid = [

        {'n_neighbors': [3,10,30]}

]



grid_search = GridSearchCV(knn_clf, param_grid, cv=5,

                               scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

grid_search.best_params_
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=100,

                             criterion='entropy',

                             max_depth=5,

                             min_samples_split=10,

                             min_samples_leaf=5,

                             random_state=0)

rf_clf.fit(X_train,y_train)

score = cross_val_score(rf_clf, X_train, y_train, cv=20, scoring="accuracy")

score.mean()
from sklearn.model_selection import GridSearchCV

param_grid = [

        {'n_estimators': [1,10,100], 'max_leaf_nodes': [16,32,64,128], 'max_features': [2, 4, 6, 8]}

]



grid_search = GridSearchCV(rf_clf, param_grid, cv=5,

                               scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

grid_search.best_params_
rf_clf = RandomForestClassifier(n_estimators =10, max_leaf_nodes = 32,max_features = 2)

rf_clf.fit(X_train,y_train)

score = cross_val_score(rf_clf, X_train, y_train, cv=20, scoring="accuracy")

score.mean()
from sklearn.svm import LinearSVC

svm_clf =  LinearSVC(C=1, loss="hinge")

svm_clf.fit(X_train, y_train)

score = cross_val_score(svm_clf, X_train, y_train, cv=20, scoring="accuracy")

score.mean()
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import PolynomialFeatures



polynomial_svm_clf = Pipeline((

            ("poly_features", PolynomialFeatures(degree=2)),

            ("scaler", StandardScaler()),

            ("svm_clf", LinearSVC(C=10, loss="hinge"))

        ))

polynomial_svm_clf.fit(X_train, y_train)

score = cross_val_score(polynomial_svm_clf, X_train, y_train, cv=20, scoring="accuracy")

score.mean()
from sklearn.naive_bayes import GaussianNB

gnb_clf = GaussianNB()

score = cross_val_score(gnb_clf, X_train, y_train, cv=20, scoring="accuracy")

score.mean()
from sklearn.linear_model import LogisticRegression

log_clf = LogisticRegression()

score = cross_val_score(log_clf, X_train, y_train, cv=20, scoring="accuracy")

score.mean()
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(

            estimators=[("sgd_clf", sgd_clf), ("dtc_clf", dtc_clf), ("knn_clf", knn_clf),

                       ("rf_clf", rf_clf), ("svm_clf", svm_clf), 

                        ("polynomial_svm_clf", polynomial_svm_clf),

                       ("gnb_clf", gnb_clf),("log_clf",log_clf)],

            voting='hard'

)

score = cross_val_score(voting_clf, X_train, y_train, cv=20, scoring="accuracy")

score.mean()
voting_clf.fit(X_train,y_train)

final_predictions = voting_clf.predict(X_test)
from sklearn.metrics import accuracy_score

accuracy_score(y_test, final_predictions)
from sklearn.model_selection import train_test_split

X_train_final = train_prepare.drop("Survived",axis=1)

y_train_final =  train_prepare["Survived"]

voting_clf.fit(X_train_final,y_train_final)

predict = voting_clf.predict(test_prepare)
submission = pd.DataFrame({

    "PassengerId": test["PassengerId"],

    "Survived": predict

})

submission.to_csv('submission.csv', index=False)
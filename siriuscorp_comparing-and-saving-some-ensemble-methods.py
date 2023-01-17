import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
Iris = pd.read_csv("../input/Iris.csv")
Iris.head()
Iris.Species.unique()
def Encode_Species(species):

    return {"Iris-setosa":1,

            "Iris-versicolor":2,

            "Iris-virginica":3

    } [species]



Iris.Species = Iris.Species.apply(lambda species: Encode_Species(species))

Iris.Species.unique()
Ids = Iris.Id

Species = Iris.Species

Iris = Iris.drop("Id",1)

Iris= Iris.drop("Species",1)
Iris.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(Iris, Species, test_size=0.33, random_state=42)

np.random.seed(123)

accuracies = []
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import BaggingClassifier

from sklearn.neighbors import KNeighborsClassifier

bagging = BaggingClassifier(KNeighborsClassifier(),

                            max_samples=0.5, max_features=0.5)

bagging.fit(X_train,y_train)

scores = cross_val_score(bagging,X_test,y_test)

accuracies.append(scores.mean())

scores.mean()
bagging.get_params()
from sklearn.model_selection import RandomizedSearchCV

bag_params = {

 'base_estimator__leaf_size': [10,100],

 'base_estimator__n_neighbors': [1,2],

 'base_estimator__weights': [ 'uniform', 'distance'],

 'bootstrap': [True,False],

 'bootstrap_features': [True,False],

 'max_features': [0.5,1.0],

 'max_samples': [0.1,1.0],

 'n_estimators': [5,50]}



n_iter_search = 40

random_search = RandomizedSearchCV(bagging, param_distributions=bag_params,

                                   n_iter=n_iter_search)



random_search.fit(X_train, y_train)

scores = cross_val_score(random_search,X_test,y_test)

accuracies.append(scores.mean())

scores.mean()
from sklearn.ensemble import RandomForestClassifier

rf_clf = RandomForestClassifier(n_estimators=10)

rf_clf.fit(X_train,y_train)

scores = cross_val_score(rf_clf,X_test,y_test)

accuracies.append(scores.mean())

scores.mean()
rf_clf.get_params()


rf_params = {"max_depth": [3, None],

              "max_features": [0.1,1.0],

              "min_samples_split": [0.1,1.0],

              "min_samples_leaf": [1,10],

              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"]}



n_iter_search = 20

random_search = RandomizedSearchCV(rf_clf, param_distributions=rf_params,

                                   n_iter=n_iter_search)



random_search.fit(X_train, y_train)

scores = cross_val_score(random_search,X_test,y_test)

accuracies.append(scores.mean())

scores.mean()
from sklearn.ensemble import ExtraTreesClassifier

XT_clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=123)

scores = cross_val_score(XT_clf,X_test,y_test)

accuracies.append(scores.mean())

scores.mean()
XT_clf.get_params()
XT_params = {

 'criterion': ['gini','entropy'],

 'max_depth': [3,None],

 'max_features': ['auto','sqrt','log2'],

 'min_impurity_decrease': [0.0,0.5],

 'min_samples_leaf': [1,10],

 'min_samples_split': [0.1,1.0],

 'min_weight_fraction_leaf': [0.0,0.5],

 'n_estimators': [10,100],

}



n_iter_search = 20

random_search = RandomizedSearchCV(XT_clf, param_distributions=XT_params,

                                   n_iter=n_iter_search)



random_search.fit(X_train, y_train)

scores = cross_val_score(random_search,X_test,y_test)

accuracies.append(scores.mean())

scores.mean()
from sklearn.ensemble import AdaBoostClassifier

aboost = AdaBoostClassifier(n_estimators=100)

aboost.fit(X_train,y_train)

scores = cross_val_score(aboost,X_test,y_test)

accuracies.append(scores.mean())

scores.mean()
aboost.get_params()
aboost_params = {

   

 'learning_rate': [0.001,100],

 'n_estimators': [10,100]

}



n_iter_search = 4

random_search = RandomizedSearchCV(aboost, param_distributions=aboost_params,

                                   n_iter=n_iter_search)



random_search.fit(X_train, y_train)

scores = cross_val_score(random_search,X_test,y_test)

accuracies.append(scores.mean())

scores.mean()


from xgboost import XGBClassifier

xgb = XGBClassifier()

xgb.fit(X_train, y_train)

scores = cross_val_score(xgb,X_test,y_test)

accuracies.append(scores.mean())

scores.mean()
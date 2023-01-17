from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.metrics import classification_report, accuracy_score

import numpy as np

import pandas as pd

import seaborn as sns
df = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
print(df.describe())
print(df_test.describe())
y = df["type"]

indexes_test = df_test["id"]



df = df.drop(["id", "color"],axis=1)

df_test = df_test.drop(["id", "color"],axis=1)
%matplotlib inline

sns.set()

sns.pairplot(df,hue="type")



df = df.drop(["type"],axis=1)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=0)
knn = KNeighborsClassifier()

params={'n_neighbors':[1,5,10,20,30,40,50,60,70,80,90,100], 'weights':('uniform', 'distance'), 'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute'), 'leaf_size':[1,5,10,20,30,40,50,60,70,80,90,100], 'p':[1,2,3]  }

grid_search_knn = GridSearchCV(knn, param_grid=params, n_jobs=8, cv=5)
grid_search_knn.fit(X_train, y_train).predict(X_test)

print(grid_search_knn.best_params_)
knn = KNeighborsClassifier(**grid_search_knn.best_params_)



knn.fit(X_train,y_train)

y_pred_knn= knn.predict(X_test) 
print(classification_report(y_pred_knn,y_test))
y_pred = knn.predict(df_test)
Y = pd.DataFrame()

Y["id"] = indexes_test

Y["type"] = y_pred

Y.to_csv("submission.csv",index=False)



print(Y.head(5))
from sklearn.ensemble import AdaBoostClassifier

ada = AdaBoostClassifier()



ada = AdaBoostClassifier()

params = {'n_estimators':[10,20,30,40,50,60,70,80,90,100], 'learning_rate':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], 'algorithm':('SAMME','SAMME.R')}

grid_search_ada = GridSearchCV(ada ,params, n_jobs=8, cv=5)



grid_search_ada.fit(X_train, y_train).predict(X_test)

print(grid_search_ada.best_params_)



ada = AdaBoostClassifier(**grid_search_ada.best_params_)



ada.fit(X_train,y_train)

y_pred_ada= ada.predict(X_test) 



print(classification_report(y_pred_ada,y_test))
svc = SVC()

params = {'kernel':('linear', 'poly', 'rbf'), 'coef0':[0.001,0.01,0.05,0.5,1],'C':[1,5,10,0.1,0.01],'gamma':[0.001,0.01,0.05,0.5,1]}

grid_search_svc = GridSearchCV(svc ,params, n_jobs=8, cv=5)



grid_search_svc.fit(X_train, y_train).predict(X_test)

print(grid_search_svc.best_params_)



svc = SVC(**grid_search_svc.best_params_)



svc.fit(X_train,y_train)

y_pred_svc= svc.predict(X_test) 



print(classification_report(y_pred_svc,y_test))



from sklearn.naive_bayes import GaussianNB



gnb = GaussianNB(priors=None)



gnb.fit(X_train,y_train)

y_pred_gnb= gnb.predict(X_test) 



print(classification_report(y_pred_gnb,y_test))
from sklearn.tree import DecisionTreeClassifier

tree = DecisionTreeClassifier()



tree = DecisionTreeClassifier()

params = {'criterion':('gini', 'entropy'),'splitter':('best','random')}

grid_search_tree = GridSearchCV(tree ,params, n_jobs=8, cv=5)



grid_search_tree.fit(X_train, y_train).predict(X_test)

print(grid_search_tree.best_params_)



tree = DecisionTreeClassifier(**grid_search_tree.best_params_)



tree.fit(X_train,y_train)

y_pred_tree= tree.predict(X_test) 



print(classification_report(y_pred_tree,y_test))
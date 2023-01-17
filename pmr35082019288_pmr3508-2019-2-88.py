import pandas as pd

import sklearn as sk

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB

from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import GridSearchCV



train_data = pd.read_csv("/kaggle/input/adult-pmr3508/train_data.csv",na_values = "?")



test_data = pd.read_csv("/kaggle/input/adult-pmr3508/test_data.csv", na_values = "?")





X_train1 = train_data[["age", 

                      "workclass", 

                      "fnlwgt", 

                      "education", 

                      "education.num", 

                      "marital.status", 

                      "occupation", 

                      "relationship", 

                      "race", 

                      "sex", 

                      "capital.gain", 

                      "capital.loss", 

                      "hours.per.week", 

                      "native.country"] ]



X_test = test_data[["age", 

                      "workclass", 

                      "fnlwgt", 

                      "education", 

                      "education.num", 

                      "marital.status", 

                      "occupation", 

                      "relationship", 

                      "race", 

                      "sex", 

                      "capital.gain", 

                      "capital.loss", 

                      "hours.per.week", 

                      "native.country"] ]





Y_train = train_data[["income"]]







X_train = pd.get_dummies(X_train1)
clf = DecisionTreeClassifier(random_state=0)

scores = cross_val_score(clf, X_train, Y_train, cv=5, verbose = 2, n_jobs = -1)

scores
clf = GaussianNB()

scores = cross_val_score(clf, X_train, Y_train, cv=5, verbose = 2, n_jobs = -1)

scores
clf = MLPClassifier()

scores = cross_val_score(clf, X_train, Y_train, cv=5, verbose = 2, n_jobs = -1)

scores


parameters={'min_samples_split' : range(10,500,50),'max_depth': range(1,20,4)}

clf_tree=DecisionTreeClassifier()

clf=GridSearchCV(clf_tree,parameters, n_jobs = -1, verbose = 2)

clf.fit( X_train, Y_train)

clf.best_score_
clf.best_params_


frames = [X_test, X_train1]

result = pd.concat(frames, sort = False)

result = pd.get_dummies(result)

X_test = result.head(16280)
final = DecisionTreeClassifier(max_depth = 9, min_samples_split = 10)

final  = final.fit( X_train, Y_train)
out = final.predict(X_test)

df = pd.DataFrame(out, columns=['Income'])

df.to_csv("submission.csv", index_label = 'Id')
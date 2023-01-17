# linear algebra

import numpy as np 



# data processing

import pandas as pd 



# data visualization

import seaborn as sns

%matplotlib inline

from matplotlib import pyplot as plt

from matplotlib import style



# Algorithms

from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC, LinearSVC

from sklearn.naive_bayes import GaussianNB



#ignore warnings

import warnings

warnings.filterwarnings("ignore")
train_df=pd.read_csv("../input/heart.csv")
a = pd.get_dummies(train_df['cp'], prefix = "cp")

b = pd.get_dummies(train_df['thal'], prefix = "thal")

c = pd.get_dummies(train_df['slope'], prefix = "slope")
frames = [train_df, a, b, c]

train_df = pd.concat(frames, axis = 1)

train_df = train_df.drop(columns = ['cp', 'thal', 'slope'])

train_df.head()
from sklearn.model_selection import train_test_split



X_train, X_test, Y_train, Y_test = train_test_split(train_df.drop('target',axis=1), 

                                                    train_df['target'], test_size=0.20, 

                                                    random_state=0)
from sklearn.linear_model import LogisticRegression
# Logistic Regression

logreg = LogisticRegression()

logreg.fit(X_train, Y_train)



Y_pred = logreg.predict(X_test)



acc_log = round(logreg.score(X_train, Y_train) * 100, 2)

print(round(acc_log,2,), "%")
# stochastic gradient descent (SGD) learning

sgd = linear_model.SGDClassifier(max_iter=4, tol=None)

sgd.fit(X_train, Y_train)

Y_pred = sgd.predict(X_test)



sgd.score(X_train, Y_train)



acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)





print(round(acc_sgd,2,), "%")
# Random Forest

random_forest = RandomForestClassifier(n_estimators=1)

random_forest.fit(X_train, Y_train)



Y_prediction = random_forest.predict(X_test)



random_forest.score(X_train, Y_train)

acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

print(round(acc_random_forest,2,), "%")
# KNN

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(X_train, Y_train)



Y_pred = knn.predict(X_test)



acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

print(round(acc_knn,2,), "%")
# Gaussian Naive Bayes

gaussian = GaussianNB()

gaussian.fit(X_train, Y_train)



Y_pred = gaussian.predict(X_test)



acc_gaussian = round(gaussian.score(X_train, Y_train) * 100, 2)

print(round(acc_gaussian,2,), "%")
# Perceptron

perceptron = Perceptron(max_iter=5)

perceptron.fit(X_train, Y_train)



Y_pred = perceptron.predict(X_test)



acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)

print(round(acc_perceptron,2,), "%")
# Linear SVC

linear_svc = LinearSVC()

linear_svc.fit(X_train, Y_train)



Y_pred = linear_svc.predict(X_test)



acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)

print(round(acc_linear_svc,2,), "%")
# Decision Tree

decision_tree = DecisionTreeClassifier(criterion='gini',splitter='best',max_depth=3,min_samples_split=3,min_samples_leaf=1,

                                       min_weight_fraction_leaf=0.0,max_features=None,random_state=None,max_leaf_nodes=None,

                                       min_impurity_decrease=0.0,min_impurity_split=None)

decision_tree.fit(X_train, Y_train)



Y_pred = decision_tree.predict(X_test)



acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)

print(round(acc_decision_tree,2,), "%")
results = pd.DataFrame({

    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 

              'Random Forest', 'Naive Bayes', 'Perceptron', 

              'Stochastic Gradient Decent', 

              'Decision Tree'],

    'Score': [acc_linear_svc, acc_knn, acc_log, 

              acc_random_forest, acc_gaussian, acc_perceptron, 

              acc_sgd, acc_decision_tree]})

result_df = results.sort_values(by='Score', ascending=False)

result_df = result_df.set_index('Score')

result_df.head(9)
# Predicted values

y_lr = logreg.predict(X_test)

knn3 = KNeighborsClassifier(n_neighbors = 3)

knn3.fit(X_train, Y_train)

y_knn = knn3.predict(X_test)

y_svm = linear_svc.predict(X_test)

y_nb = gaussian.predict(X_test)

y_dtc = decision_tree.predict(X_test)

y_rf = random_forest.predict(X_test)
from sklearn.metrics import confusion_matrix



cm_lr = confusion_matrix(Y_test,y_lr)

cm_knn = confusion_matrix(Y_test,y_knn)

cm_svm = confusion_matrix(Y_test,y_svm)

cm_nb = confusion_matrix(Y_test,y_nb)

cm_dtc = confusion_matrix(Y_test,y_dtc)

cm_rf = confusion_matrix(Y_test,y_rf)
plt.figure(figsize=(24,12))



plt.suptitle("Confusion Matrixes",fontsize=24)

plt.subplots_adjust(wspace = 0.4, hspace= 0.4)



plt.subplot(2,3,1)

plt.title("Logistic Regression Confusion Matrix")

sns.heatmap(cm_lr,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,2)

plt.title("K Nearest Neighbors Confusion Matrix")

sns.heatmap(cm_knn,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,3)

plt.title("Support Vector Machine Confusion Matrix")

sns.heatmap(cm_svm,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,4)

plt.title("Naive Bayes Confusion Matrix")

sns.heatmap(cm_nb,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,5)

plt.title("Decision Tree Classifier Confusion Matrix")

sns.heatmap(cm_dtc,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.subplot(2,3,6)

plt.title("Random Forest Confusion Matrix")

sns.heatmap(cm_rf,annot=True,cmap="Blues",fmt="d",cbar=False)



plt.show()
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn import tree

from sklearn.neighbors import NearestCentroid

from mlxtend.plotting import plot_decision_regions

from matplotlib.colors import ListedColormap

from sklearn.metrics import confusion_matrix

from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

import xgboost as xgb

from sklearn.metrics import recall_score

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from xgboost import plot_tree

from xgboost import XGBClassifier

import graphviz 

from sklearn.naive_bayes import GaussianNB
file = pd.read_csv("../input/ecoli-data-set-sampled-random-over-sampler/ecoli_sampled.csv")

df = file.values

X = df[:,0:7]

y = df[:,7]
#XGB Classifier



X_train, X_test, y_train, y_test = train_test_split(X, y)

clf1 = XGBClassifier()

clf1.fit(X_train, y_train)

y_pred = clf1.predict(X_test)



accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: %.2f%%" % (accuracy * 100.0))

recall1 = recall_score(y_test, y_pred, average='micro')

print("Recall: %.2f%%" % (recall1 * 100.0))

conf_mat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print('Confusion matrix:\n', conf_mat)

#Decision Tree Classifier



clf2 = tree.DecisionTreeClassifier()

clf2 = clf2.fit(X,y)



X_train, X_test, y_train, y_test = train_test_split(X, y)



clf2.fit(X_train,y_train)

y_pred1 = clf2.predict(X_test)

accuracy1 = accuracy_score(y_test, y_pred1)

print('Accuracy Score',accuracy1*100)

recall2 = recall_score(y_test, y_pred1, average = 'macro')

print('Recall Score',recall2*100)

conf_mat1 = confusion_matrix(y_true=y_test, y_pred=y_pred1)

print('Confusion matrix:\n', conf_mat1)

#KNN Classifer



clf3 = KNeighborsClassifier()

clf3 = clf3.fit(X,y)



X_train, X_test, y_train, y_test = train_test_split(X, y)



clf3.fit(X_train,y_train)

y_pred2 = clf3.predict(X_test)

accuracy2 = accuracy_score(y_test, y_pred2)

print('Accuracy Score',accuracy2*100)

recall3 = recall_score(y_test, y_pred2, average = 'macro')

print('Recall Score',recall3*100)

conf_mat2 = confusion_matrix(y_true=y_test, y_pred=y_pred2)

print('Confusion matrix:\n', conf_mat2)

#SVC Classifier



clf4 = SVC()

clf4 = clf4.fit(X,y)



X_train, X_test, y_train, y_test = train_test_split(X, y)



clf4.fit(X_train,y_train)

y_pred3 = clf4.predict(X_test)

accuracy3 = accuracy_score(y_test, y_pred3)

print('Accuracy Score',accuracy3*100)

recall4 = recall_score(y_test, y_pred3, average = 'macro')

print('Recall Score',recall4*100)

conf_mat3 = confusion_matrix(y_true=y_test, y_pred=y_pred3)

print('Confusion matrix:\n', conf_mat3)

#Naive Base Classier



clf5 = GaussianNB()

clf5 = clf5.fit(X,y)



X_train, X_test, y_train, y_test = train_test_split(X, y)



clf5.fit(X_train,y_train)

y_pred4 = clf5.predict(X_test)

accuracy4 = accuracy_score(y_test, y_pred4)

print('Accuracy Score',accuracy4*100)

recall5 = recall_score(y_test, y_pred4, average = 'macro')

print('Recall Score',recall5*100)

conf_mat4 = confusion_matrix(y_true=y_test, y_pred=y_pred4)

print('Confusion matrix:\n', conf_mat4)

#Logistic Regression



clf6 = LogisticRegression()

clf6 = clf6.fit(X,y)



X_train, X_test, y_train, y_test = train_test_split(X, y)



clf6.fit(X_train,y_train)

y_pred5 = clf6.predict(X_test)

accuracy5 = accuracy_score(y_test, y_pred5)

print('Accuracy Score',accuracy5*100)

recall6 = recall_score(y_test, y_pred5, average = 'macro')

print('Recall Score',recall6*100)

conf_mat5 = confusion_matrix(y_true=y_test, y_pred=y_pred5)

print('Confusion matrix:\n', conf_mat5)
#Visualizing XGBboost Classifier



plot_tree(clf1)

image1 = xgb.to_graphviz(clf1)

image1.render('xgbboost')
#Visualizing Decision Tree Classifier



tree.plot_tree(clf2)



dot_data = tree.export_graphviz(clf2, out_file=None) 

image2 = graphviz.Source(dot_data) 

image2.render("decision_tree")
#Plotting the KNN Decision boundaries



value = 110

width = 880

plot_decision_regions(X, y.astype(np.integer), clf=clf3,filler_feature_values={0:value,1:value,2:value,3:value,4:value,5:value,6:value},

                      filler_feature_ranges={0:width,1:width,2:width,3:width,4:width,5:width,6:width},

                      X_highlight=None,zoom_factor=(8.0))





# Adding axes annotations

plt.xlabel('Various Ecoli Measures')

plt.ylabel('Various Ecoli Measures')

#plt.figure(figsize=(6,3))

plt.title('KNN with Ecoli Dataset')

plt.show()
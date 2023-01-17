# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
all_data= pd.read_csv("../input/creditcard.csv")

labels = all_data["Class"].values

times = all_data["Time"].values

features = all_data.drop('Class', 1)

features = features.drop('Time', 1)

print(all_data.shape)

print(features.shape)

print(labels.shape)
from sklearn.linear_model import LogisticRegression

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing

from sklearn.metrics import confusion_matrix, classification_report

from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=0)
lr = LogisticRegression(class_weight='balanced',random_state=0)

model_lr = lr.fit(features_train,labels_train)

y_predicted_lr = model_lr.predict(features_test)
print('Confusion matrix (Logistic Regression)\n')

print(confusion_matrix(labels_test, y_predicted_lr))
print('Classification report (Logistic Regression)\n')

print(classification_report(labels_test, y_predicted_lr, digits=5))
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced',n_estimators=500,min_samples_leaf=20,random_state=0)

model_rf = rf.fit(features_train,labels_train)

y_predicted_rf = model_rf.predict(features_test)
print('Confusion matrix (Random Forest)\n')

print(confusion_matrix(labels_test, y_predicted_rf))
print('Classification report (Random Forest)\n')

print(classification_report(labels_test, y_predicted_rf, digits=5))
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs', activation='tanh', alpha=1, hidden_layer_sizes=(58,58), learning_rate='adaptive', random_state=0)

model_mlp = mlp.fit(features_train,labels_train)

y_predicted_mlp = model_mlp.predict(features_test)
print('Confusion matrix (Multi-layer Perceptron)\n')

print(confusion_matrix(labels_test, y_predicted_mlp))
print('Classification report (Multi-layer Perceptron)\n')

print(classification_report(labels_test, y_predicted_mlp, digits=5))
from sklearn import tree

tree_clf = tree.DecisionTreeClassifier(random_state=0)

tree_model = tree_clf.fit(features_train,labels_train)

y_predicted_tree = tree_model.predict(features_test)
print('Confusion matrix (Decision Tree)\n')

print(confusion_matrix(labels_test, y_predicted_tree))
print('Classification report (Decision Tree)\n')

print(classification_report(labels_test, y_predicted_tree, digits=5))
features_class0 = features[labels == 0]

features_class1 = features[labels == 1]

print('FEATURES')

print('Class 0: ', features_class0.shape, 'Class 1: ', features_class1.shape)

labels_class0 = labels[(labels == 0)]

labels_class1 = labels[(labels == 1)]

print('LABELS')

print('Class 0: ', labels_class0.shape, 'Class 1: ', labels_class1.shape)
features_class0_sub = features_class0.sample(n=10000, random_state=0)

labels_class0_sub = labels_class0[:10000]

print('Class 0 :', features_class0_sub.shape[0], 'Class 1: ', features_class1.shape[0])



features_sub = np.concatenate((features_class0_sub,features_class1))

labels_sub = np.concatenate((labels_class0_sub,labels_class1))

features_sub_train, features_sub_test, labels_sub_train, labels_sub_test = train_test_split(

    features_sub, labels_sub, test_size=0.3, random_state=0)
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=500,min_samples_leaf=20,random_state=0)

model_rf_sub = rf.fit(features_sub_train,labels_sub_train)

y_predicted_rf_sub = model_rf_sub.predict(features_sub_test)
print('Confusion matrix (Random Forest, with subsample)\n')

print(confusion_matrix(labels_sub_test, y_predicted_rf_sub))
print('Classification report (Random Forest, with subsample)\n')

print(classification_report(labels_sub_test, y_predicted_rf_sub, digits=5))
from sklearn.neural_network import MLPClassifier

mlp = MLPClassifier(solver='lbfgs', activation='tanh', alpha=1, hidden_layer_sizes=(58, 58), learning_rate='adaptive', random_state=0)

model_mlp_sub = mlp.fit(features_sub_train,labels_sub_train)

y_predicted_mlp_sub = model_mlp_sub.predict(features_sub_test)
print('Confusion matrix (Multi-layer Perceptron, with subsample)\n')

print(confusion_matrix(labels_sub_test, y_predicted_mlp_sub))
print('Classification report (Multi-layer Perceptron, with subsample)\n')

print(classification_report(labels_sub_test, y_predicted_mlp_sub, digits=5))
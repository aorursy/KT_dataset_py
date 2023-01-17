# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Data Visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
# Text Vectorization and tfidf
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
# data preprocessing
from sklearn.preprocessing import LabelEncoder
# Classification models
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
# Training and testing data split
from sklearn.model_selection import train_test_split
# Performance matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_score, recall_score
import warnings
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# For example, here's several helpful packages to load in 
%matplotlib inline
# Loading the data
data = pd.read_csv('../input/cleanedmrdata.csv')
data.head()
# Data Information
data.info()
# counting the null values
data.isnull().sum()
# Removing the data row wise
data.dropna(axis=0,inplace=True)
data.info()
percents = data.iloc[:, 7:].mean() * 100
plt.figure(figsize=(15, 10))
plt.bar(range(len(percents)), percents)
plt.title("Blog Post Tags")
plt.ylabel("Percentage of Blog Post With Tag")
plt.gca().set_xticklabels(percents.index)
plt.gca().set_xticks(np.arange(len(percents)) + .45)
plt.xticks(rotation=90)
plt.show()
plt.figure(figsize=(15, 10))
t_percents = data[data['author'] == 'Tyler Cowen'].iloc[:, 7:].mean() * 100
a_percents = data[data['author'] == 'Alex Tabarrok'].iloc[:, 7:].mean() * 100
labels = data[data['author'] == 'Tyler Cowen'].iloc[:, 7:].mean().index
t_color = np.random.rand(3)
a_color = np.random.rand(3)
handles = [patches.Patch(label='Alex Tabarook', color=a_color), patches.Patch(label='Tyler Cowen', color=t_color)]
ind = np.arange(len(t_percents))
plt.bar(ind, t_percents, width=.45, color=t_color)
plt.bar(ind+.45, a_percents, width=.45, color=a_color)
plt.gca().set_xticklabels(labels)
plt.gca().set_xticks(ind + .45)
plt.legend(handles=handles)
plt.xticks(rotation=90)
plt.title("Blog Post Tags")
plt.ylabel("Percentage of Blog Post With Tag")
plt.show()
sns.boxplot(x='author', y='wordcount', data=data)
#plt.figure(figsize=(15, 10))
plt.show()
vectorizer = TfidfVectorizer().fit(data['text'])
feature_vect = vectorizer.transform(data['text'])
target_vect = LabelEncoder().fit_transform(data['author'])
train_features,test_features,train_targets,test_targets = train_test_split( feature_vect, target_vect, test_size=0.33, random_state=42)
'''train_features = feature_vect[:8000]
test_features = feature_vect[8000:]
train_targets = target_vect[:8000]
test_targets = target_vect[8000:]'''
# Random Forest Classifier
clf_rf = RandomForestClassifier()
#Traing my model
clf_rf.fit(train_features, train_targets)

#Making the prediction
Pridicted_test_targets = clf_rf.predict(test_features)

#Measuring the accuracy of machine
acc_rf = accuracy_score(test_targets, Pridicted_test_targets)
print ("random forest accuracy: ",acc_rf)
#Cross Validation
cross_val=cross_val_score(clf_rf, train_features, train_targets, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(clf_rf, train_features, train_targets, cv=3)
print(y_train_pred)
#Confussion matrix
conf_mx=confusion_matrix(train_targets, y_train_pred)
print("Confussion matrix:",conf_mx)
ps=precision_score(train_targets, y_train_pred,average="macro")
print("Precision Score:",ps)
rs=recall_score(train_targets, y_train_pred,average="macro")
print("Recall Score:",rs)
# plotting the confussion matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)
# roc curve score
roc_auc_score(test_targets, Pridicted_test_targets)
#Preparing ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_targets, Pridicted_test_targets)
roc_auc = auc(false_positive_rate, true_positive_rate)
#Plotting ROC Curve
plt.title('ROC CURVE')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True_Positive_Rate')
plt.xlabel('False_Positive_Rate')
plt.show()
# output data
my_submission = pd.DataFrame({'author': Pridicted_test_targets})
my_submission.to_csv('AutrorProfilingRandomForest.csv', index=False)
#Support vector classifier
clf_svm = LinearSVC()

#Training the model
clf_svm.fit(train_features, train_targets)
#Measuring the accuracy of machine
acc_svc = accuracy_score(test_targets, Pridicted_test_targets)
print ("SVC accuracy: ",acc_svc)
#Cross Validation
cross_val=cross_val_score(clf_svm, train_features, train_targets, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(clf_svm, train_features, train_targets, cv=3)
print(y_train_pred)
#Confussion matrix
conf_mx=confusion_matrix(train_targets, y_train_pred)
print("Confussion matrix:",conf_mx)
ps=precision_score(train_targets, y_train_pred,average="macro")
print("Precision Score:",ps)
rs=recall_score(train_targets, y_train_pred,average="macro")
print("Recall Score:",rs)
# plotting the confussion matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)
# ROC Score
roc_auc_score(test_targets, Pridicted_test_targets)
#Preparing ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_targets, Pridicted_test_targets)
roc_auc = auc(false_positive_rate, true_positive_rate)
#Plotting ROC Curve
plt.title('ROC CURVE')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True_Positive_Rate')
plt.xlabel('False_Positive_Rate')
plt.show()
# Bernoulli NB model
clf = BernoulliNB()
# training the data
clf.fit(train_features, train_targets)
# Accuracy
accuracy_score(test_targets, clf.predict(test_features))
#Cross Validation
cross_val=cross_val_score(clf, train_features, train_targets, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(clf, train_features, train_targets, cv=3)
print(y_train_pred)
#Confussion matrix
conf_mx=confusion_matrix(train_targets, y_train_pred)
print("Confussion matrix:",conf_mx)
ps=precision_score(train_targets, y_train_pred,average="macro")
print("Precision Score:",ps)
rs=recall_score(train_targets, y_train_pred,average="macro")
print("Recall Score:",rs)
# plotting the confussion matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)
# ROC Score
roc_auc_score(test_targets, Pridicted_test_targets)
#Preparing ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_targets, Pridicted_test_targets)
roc_auc = auc(false_positive_rate, true_positive_rate)
#Plotting ROC Curve
plt.title('ROC CURVE')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True_Positive_Rate')
plt.xlabel('False_Positive_Rate')
plt.show()
# MLP classifier Model
net = MLPClassifier(hidden_layer_sizes = (500, 250))
# Training the model
net.fit(train_features, train_targets)
# Accuracy
accuracy_score(test_targets, net.predict(test_features))
#Cross Validation
cross_val=cross_val_score(net, train_features, train_targets, cv=3, scoring="accuracy")
print(cross_val)
y_train_pred = cross_val_predict(net, train_features, train_targets, cv=3)
print(y_train_pred)
#Confussion matrix
conf_mx=confusion_matrix(train_targets, y_train_pred)
print("Confussion matrix:",conf_mx)
ps=precision_score(train_targets, y_train_pred,average="macro")
print("Precision Score:",ps)
rs=recall_score(train_targets, y_train_pred,average="macro")
print("Recall Score:",rs)
# plotting the confussion matrix
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
print("_"*100)
#ROC Score
roc_auc_score(test_targets, Pridicted_test_targets)
#Preparing ROC curve
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_targets, Pridicted_test_targets)
roc_auc = auc(false_positive_rate, true_positive_rate)
#Plotting ROC Curve
plt.title('ROC CURVE')
plt.plot(false_positive_rate, true_positive_rate, 'b',
label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True_Positive_Rate')
plt.xlabel('False_Positive_Rate')
plt.show()












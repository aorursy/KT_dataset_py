# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#for visualising
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
dataset=pd.read_csv("../input/spam.csv",encoding="latin-1")
dataset.head()
dataset.columns
dataset = dataset.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
dataset.tail()
dataset.info()
#for text preprocessing
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus=[]
for i in range(0,len(dataset)):
    review=re.sub("[^a-zA-Z]"," ",dataset.iloc[i][1])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    review=[ps.stem(word) for word in review if not word in set(stopwords.words("english"))]
    review=" ".join(review)
    corpus.append(review)
#before and after text preprocessing
print(dataset.iloc[6,1])
print(30*"*")
print(corpus[6])
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X=cv.fit_transform(corpus).toarray()
y=dataset.iloc[:,0].values
y=[0 if each=="ham" else 1 for each in y]
X
#Classification algorithms

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#These are for visualising
algoritma=[]
accuracy=[]
#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(random_state=0,criterion="entropy")
classifier.fit(X_train,y_train)

#predict
y_pred=classifier.predict(X_test)


from sklearn.metrics import confusion_matrix
cm_dt=confusion_matrix(y_test,y_pred)
dtscore=classifier.score(X_test,y_test)*100
print("Decision Tree accuracy=",classifier.score(X_test,y_test)*100)
algoritma.append("Decision Tree")
accuracy.append(dtscore)

#plot confusion matrix
f,ax=plt.subplots(figsize=(4,4))
sns.heatmap(cm_dt,annot=True,fmt=".0f",ax=ax,linewidths=2,linecolor="blue")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("predicted Values")
plt.ylabel("true values")
plt.show()
#Find the optimal  k-value

#error=[]
#knn_accuracy=[]
#for i in range(3, 9): 
#    classifier=KNeighborsClassifier(n_neighbors=i)
#    classifier.fit(X_train,y_train)
#    pred_i = classifier.predict(X_test)
#    error.append(np.mean(pred_i != y_test))
#    knn_accuracy.append(classifier.score(X_test,y_test)*100)
#plt.figure(figsize=(12, 6))  
#plt.plot(range(3, 9), error, color='red', linestyle='dashed', marker='o',  
#         markeclassifieracecolor='blue', markersize=10)
#plt.title('Error Rate K Value')  
#plt.xlabel('K Value')  
#plt.ylabel('Mean Error')  
#
#plt.figure(figsize=(12, 6))  
#plt.plot(range(3, 9), knn_accuracy, color='red', linestyle='dashed', marker='o',  
#         markeclassifieracecolor='blue', markersize=10)
#plt.title('Accuracy of K Value')  
#plt.xlabel('K Value')  
#plt.ylabel('Accuracy')
from sklearn.neighbors import KNeighborsClassifier

classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_knn= confusion_matrix(y_test, y_pred)
knn=classifier.score(X_test,y_test)*100
print("KNN accuracy=",classifier.score(X_test,y_test)*100)
algoritma.append("KNN")
accuracy.append(knn)

#plot confusion matrix
f,ax=plt.subplots(figsize=(4,4))
sns.heatmap(cm_knn,annot=True,fmt=".0f",ax=ax,linewidths=2,linecolor="blue")
plt.title("KNN Confusion Matrix")
plt.xlabel("predicted Values")
plt.ylabel("true values")
plt.show()
#Find the optimal n_estimators value

#n_estimators = np.arange(1,200,5)
#train_results = []
#test_results = []
#from sklearn.metrics import roc_curve, auc
#for estimator in n_estimators:
#   classifier = RandomForestClassifier(n_estimators=estimator)
#   classifier.fit(X_train, y_train)
#   train_pred = classifier.predict(X_train)
#   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_train, train_pred)
#   roc_auc = auc(false_positive_rate, true_positive_rate)
#   train_results.append(roc_auc)
#   y_pred = classifier.predict(X_test)
#   false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
#   roc_auc = auc(false_positive_rate, true_positive_rate)
#   test_results.append(roc_auc)
#
#plt.plot(n_estimators, train_results, "b", label="Train AUC")
#plt.plot(n_estimators, test_results, "r", label="Test AUC")
#plt.legend()
#plt.ylabel("AUC score")
#plt.xlabel("n_estimators")
#plt.show()
#Random Forest
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=67,criterion = 'entropy', random_state = 0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_rf= confusion_matrix(y_test, y_pred)
random_forest=classifier.score(X_test,y_test)*100
print("Random Forest accuracy=",classifier.score(X_test,y_test)*100)
algoritma.append("Random Forest")
accuracy.append(random_forest)

#plot confusion matrix
f,ax=plt.subplots(figsize=(4,4))
sns.heatmap(cm_rf,annot=True,fmt=".0f",ax=ax,linewidths=2,linecolor="blue")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("predicted Values")
plt.ylabel("true values")
plt.show()
#SVM
from sklearn.svm import SVC
classifier=SVC(kernel="linear",random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_svm = confusion_matrix(y_test, y_pred)
svm=classifier.score(X_test,y_test)*100
print("SVM accuracy=",classifier.score(X_test,y_test)*100)
algoritma.append("SVM")
accuracy.append(svm)

#plot confusion matrix
f,ax=plt.subplots(figsize=(4,4))
sns.heatmap(cm_svm,annot=True,fmt=".0f",ax=ax,linewidths=2,linecolor="blue")
plt.title("SVM Confusion Matrix")
plt.xlabel("predicted Values")
plt.ylabel("true values")
plt.show()
#Navie Bayes
from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_nb= confusion_matrix(y_test, y_pred)
nb=classifier.score(X_test,y_test)*100
print("Navie Bayes accuracy=",classifier.score(X_test,y_test)*100)
algoritma.append("Navie Bayes")
accuracy.append(nb)

#plot confusion matrix
f,ax=plt.subplots(figsize=(4,4))
sns.heatmap(cm_nb,annot=True,fmt=".0f",ax=ax,linewidths=2,linecolor="blue")
plt.title("Navie Bayes Confusion Matrix")
plt.xlabel("predicted Values")
plt.ylabel("true values")
plt.show()
#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm_lr = confusion_matrix(y_test, y_pred)
lr=classifier.score(X_test,y_test)*100
print("Logistic Regression accuracy=",classifier.score(X_test,y_test)*100)
algoritma.append("Logistic Regression")
accuracy.append(lr)


#plot confusion matrix
f,ax=plt.subplots(figsize=(4,4))
sns.heatmap(cm_lr,annot=True,fmt=".0f",ax=ax,linewidths=2,linecolor="blue")
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("predicted Values")
plt.ylabel("true values")
plt.show()
plt.figure(figsize=(24,24))

plt.suptitle("Confusion Matrixes",fontsize=24)
#Logistic Regression Confusion Matrix
plt.subplot(2,3,1)
plt.title("Logistic Regression Confusion Matrix")
sns.heatmap(cm_lr,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt=".0f")

#Decision Tree Confusion Matrix
plt.subplot(2,3,2)
plt.title("Decision Tree Classifier Confusion Matrix")
sns.heatmap(cm_dt,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt=".0f")

#K Nearest Neighbors Confusion Matrix
plt.subplot(2,3,3)
plt.title("K Nearest Neighbors Confusion Matrix")
sns.heatmap(cm_knn,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt=".0f")

#Naive Bayes Confusion Matrix
plt.subplot(2,3,4)
plt.title("Naive Bayes Confusion Matrix")
sns.heatmap(cm_nb,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt=".0f")

#Random Forest Confusion Matrix
plt.subplot(2,3,5)
plt.title("Random Forest Confusion Matrix")
sns.heatmap(cm_rf,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt=".0f")

#Support Vector Machine Confusion Matrix
plt.subplot(2,3,6)
plt.title("Support Vector Machine Confusion Matrix")
sns.heatmap(cm_svm,cbar=False,annot=True,linewidths=2,linecolor="orange",fmt="d")

plt.show()
f=plt.subplots(figsize=(20,20))
plt.bar(algoritma,accuracy,color="orange")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.show()
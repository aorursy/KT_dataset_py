import numpy as np
import pandas as pd

import os
data = pd.read_csv("../input/voice.csv")
data.head()
data.label = [1 if each == "male" else 0 for each in data.label]
data.head() # check if binary conversion worked
gender = data.label.values
features = data.drop(["label"], axis = 1)
features = (features-features.min())/(features.max()-features.min()) # normalization
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(features, gender, test_size = 0.2, random_state = 42)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
list_names = []
list_accuracy = []
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(x_train, y_train)
LR_accuracy = lr.score(x_test, y_test)*100
LR_accuracy = round(LR_accuracy, 3)

print("LR_accuracy is %", LR_accuracy)

list_names.append("Logistic Regression ")
list_accuracy.append(LR_accuracy)

# Confusion Matrix
y_pred_RF = lr.predict(x_test)
RF_cm = confusion_matrix(y_test, y_pred_RF)

f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(RF_cm, annot = True, linewidth = 0.5, linecolor = "black", fmt = ".0f", ax = ax)
plt.title('Confusion Matrix of the Logistic Regression Classifier')
plt.xlabel("Prediction")
plt.ylabel("True")
plt.show()
from sklearn.neighbors import KNeighborsClassifier

Knn = KNeighborsClassifier(n_neighbors = 8)
Knn.fit(x_train, y_train)
Knn_accuracy = Knn.score(x_test, y_test)*100
Knn_accuracy = round(Knn_accuracy, 3)

print("Knn_accuracy is %", Knn_accuracy)

list_names.append("K-nn ")
list_accuracy.append(Knn_accuracy)

# Confusion Matrix
y_pred_Knn = Knn.predict(x_test)
Knn_cm = confusion_matrix(y_test, y_pred_Knn)

f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(Knn_cm, annot = True, linewidth = 0.5, linecolor = "black", fmt = ".0f", ax = ax)
plt.title('Confusion Matrix of the K-nn Classifier')
plt.xlabel("Prediction")
plt.ylabel("True")
plt.show()
from sklearn.svm import SVC

svm = SVC(random_state=1)
svm.fit(x_train, y_train)
SVM_accuracy = svm.score(x_test, y_test)*100
SVM_accuracy = round(SVM_accuracy, 3)

print("SVM_accuracy is %", SVM_accuracy)

list_names.append("SVM ")
list_accuracy.append(SVM_accuracy)

# Confusion Matrix
y_pred_SVM = svm.predict(x_test)
SVM_cm = confusion_matrix(y_test, y_pred_SVM)

f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(SVM_cm, annot = True, linewidth = 0.5, linecolor = "black", fmt = ".0f", ax = ax)
plt.title('Confusion Matrix of the Support Vector Machine (SVM)')
plt.xlabel("Prediction")
plt.ylabel("True")
plt.show()
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(x_train, y_train)
NaiveBayes_acuracy = nb.score(x_test, y_test)*100
NaiveBayes_acuracy = round(NaiveBayes_acuracy,3)

print("NaiveBayes_acuracy is %", NaiveBayes_acuracy)

list_names.append("Naive Bayes ")
list_accuracy.append(NaiveBayes_acuracy)

# Confusion Matrix
y_pred_NB = nb.predict(x_test)
NB_cm = confusion_matrix(y_test, y_pred_NB)

f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(NB_cm, annot = True, linewidth = 0.5, linecolor = "black", fmt = ".0f", ax = ax)
plt.title('Confusion Matrix of the Naive Bayes')
plt.xlabel("Prediction")
plt.ylabel("True")
plt.show()
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)
DecisionTree_accuracy = dt.score(x_test, y_test)*100
DecisionTree_accuracy = round(DecisionTree_accuracy,3)

print("DecisionTree_accuracy is %", DecisionTree_accuracy)

list_names.append("Decision Tree ")
list_accuracy.append(DecisionTree_accuracy)

# Confusion Matrix
y_pred_DT = dt.predict(x_test)
DT_cm = confusion_matrix(y_test, y_pred_DT)

f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(DT_cm, annot = True, linewidth = 0.5, linecolor = "black", fmt = ".0f", ax = ax)
plt.title('Confusion Matrix of the Decision Tree')
plt.xlabel("Prediction")
plt.ylabel("True")
plt.show()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 10, random_state = 1)
rf.fit(x_train, y_train)
RandomForest_accuracy = rf.score(x_test, y_test)*100
RandomForest_accuracy = round(RandomForest_accuracy, 3)

print("RandomForest_accuracy is %", RandomForest_accuracy)

list_names.append("Random Forest ")
list_accuracy.append(RandomForest_accuracy)

# Confusion Matrix
y_pred_RF = rf.predict(x_test)
RF_cm = confusion_matrix(y_test, y_pred_RF)

f, ax = plt.subplots(figsize = (5,5))
sns.heatmap(RF_cm, annot = True, linewidth = 0.5, linecolor = "black", fmt = ".0f", ax = ax)
plt.title('Confusion Matrix of the Random Forest')
plt.xlabel("Prediction")
plt.ylabel("True")
plt.show()
x = list_names
y = list_accuracy
    
fig = plt.figure(figsize=(12,10))
width = 0.4 # the width of the bars 
ind = np.arange(len(x))  # the x locations for the groups
plt.ylim([90,99])
plt.bar(x, y, width)
plt.xticks(ind, rotation=90, fontsize = 16)
plt.yticks(fontsize = 16)
plt.grid(alpha=0.4)
plt.ylabel("ACCURACY (%)", fontsize = 16)
plt.title("COMPARISON of the ACCURACY of the MACHINE LEARNING METHODS", fontsize = 16, pad = 20)  
plt.show()
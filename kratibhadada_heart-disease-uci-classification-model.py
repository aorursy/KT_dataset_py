# importing libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import roc_auc_score, roc_curve
# reading data                        

data=pd.read_csv("../input/heart.csv")
# displaying first 5 rows

data.head()
# (no. of rows, no. of columns)

data.shape 
data.describe()
# finding any null values in data

data.isnull().any()
data.info()
# Finding the number of patients with heart disease.

sns.countplot(x="target",data=data,palette="pastel")

plt.show()
# Finding the ration of males and females in the data (1 = male; 0 = female)

sns.countplot(x="sex",data=data,palette="colorblind")

plt.show()
# Finding correaltion between all the parameters in the dataset.

fig,ax = plt.subplots(figsize=(11,8))

sns.heatmap(data.corr(),annot=True,cmap="Blues" ,ax=ax)

plt.show()
# creating dummy variables

a=pd.get_dummies(data["cp"],prefix="cp")

b=pd.get_dummies(data["restecg"],prefix="restecg")

c=pd.get_dummies(data["ca"],prefix="ca")

d=pd.get_dummies(data["thal"],prefix="thal")

e=pd.get_dummies(data["slope"],prefix="slope")
# joining dummy variables in the dataset.

data=pd.concat([data,a,b,c,d],axis=1)

data.head()
# no. of rows and columns after addition of dummy variables

data.shape
# dropping of columns whose dummy variables have been created.

data=data.drop(columns=["cp","restecg","thal","ca","slope"])

data.head()
# x= independent variables

x=data.drop("target",axis=1)

x.head()
# y=dependent variable (target) 

y=data["target"]

y.head()
# splitting data into train and test set.

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
# making object classifier of class LogisticRegression 

from sklearn.linear_model import LogisticRegression

classifier=LogisticRegression()
# Fitting training data set into classifier

classifier.fit(x_train,y_train)
# Predicting test results

y_pred=classifier.predict(x_test)
# Making confusion matrix

cm=confusion_matrix(y_test,y_pred)

cm
# Heatmap of confusion matrix

sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")

plt.show()
print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")

print("Precision = ",precision_score(y_test,y_pred)*100,"%")

print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")
sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])

print ("Sensitivity =",sensitivity)

specificity= cm[0,0]/(cm[0,0] + cm[0,1])

print("Specificity =",specificity)
# calculating AUC

auc=roc_auc_score(y_test,y_pred)

auc
# calculating ROC curve

fpr,tpr,thresholds= roc_curve(y_test,y_pred)
# plotting the roc curve for the model

plt.plot([0,1],[0,1],linestyle="--")

plt.plot(fpr,tpr,marker=".")

plt.xlabel("False Positive Rate")

plt.ylabel("TruePositive Rate")

plt.title("ROC Curve")

plt.show()
# making object classifier of class KNeighborsClassifier 

from sklearn.neighbors import KNeighborsClassifier

classifier= KNeighborsClassifier(n_neighbors=5,metric="minkowski",p=2)
# Fitting training data set into classifier

classifier.fit(x_train,y_train)
# Predicting test results

y_pred=classifier.predict(x_test)
# Making confusion matrix

cm=confusion_matrix(y_test,y_pred)

cm
# Heatmap of confusion matrix

sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")

plt.show()
print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")

print("Precision = ",precision_score(y_test,y_pred)*100,"%")

print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")
sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])

print ("Sensitivity =",sensitivity)

specificity= cm[0,0]/(cm[0,0] + cm[0,1])

print("Specificity =",specificity)
# calculating AUC

auc=roc_auc_score(y_test,y_pred)

auc
# calculating ROC curve

fpr,tpr,thresholds= roc_curve(y_test,y_pred)
# plotting the roc curve for the model

plt.plot([0,1],[0,1],linestyle="--")

plt.plot(fpr,tpr,marker=".")

plt.xlabel("False Positive Rate")

plt.ylabel("TruePositive Rate")

plt.title("ROC Curve")

plt.show()
# Cross Validation : Calculating cross validation score

from sklearn.model_selection import cross_val_score

score = cross_val_score(classifier,x_train,y_train,cv=10,scoring="accuracy")

score
score.mean()
# making object classifier of class SVC

from sklearn.svm import SVC

classifier= SVC(kernel="linear")
# Fitting training data set into classifier

classifier.fit(x_train,y_train)
# Predicting test results

y_pred=classifier.predict(x_test)
# Making confusion matrix

cm=confusion_matrix(y_test,y_pred)

cm
# Heatmap of confusion matrix

sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")

plt.show()
print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")

print("Precision = ",precision_score(y_test,y_pred)*100,"%")

print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")
sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])

print ("Sensitivity =",sensitivity)

specificity= cm[0,0]/(cm[0,0] + cm[0,1])

print("Specificity =",specificity)
# calculating AUC

auc=roc_auc_score(y_test,y_pred)

auc
# calculating ROC curve

fpr,tpr,thresholds= roc_curve(y_test,y_pred)
# plotting the roc curve for the model

plt.plot([0,1],[0,1],linestyle="--")

plt.plot(fpr,tpr,marker=".")

plt.xlabel("False Positive Rate")

plt.ylabel("TruePositive Rate")

plt.title("ROC Curve")

plt.show()
# making object classifier of class GaussianNB

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
# Fitting training data set into classifier

classifier.fit(x_train,y_train)
# Predicting test results

y_pred=classifier.predict(x_test)
# Making confusion matrix

cm=confusion_matrix(y_test,y_pred)

cm
# Heatmap of confusion matrix

sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")

plt.show()
print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")

print("Precision = ",precision_score(y_test,y_pred)*100,"%")

print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")
sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])

print ("Sensitivity =",sensitivity)

specificity= cm[0,0]/(cm[0,0] + cm[0,1])

print("Specificity =",specificity)
# calculating AUC

auc=roc_auc_score(y_test,y_pred)

auc
# calculating ROC curve

fpr,tpr,thresholds= roc_curve(y_test,y_pred)
# plotting the roc curve for the model

plt.plot([0,1],[0,1],linestyle="--")

plt.plot(fpr,tpr,marker=".")

plt.xlabel("False Positive Rate")

plt.ylabel("TruePositive Rate")

plt.title("ROC Curve")

plt.show()
# making object classifier of class DecisionTreeClassifier

from sklearn.tree import DecisionTreeClassifier

classifier=DecisionTreeClassifier(criterion="gini",random_state=0)
# Fitting training data set into classifier

classifier.fit(x_train,y_train)
# Predicting test results

y_pred=classifier.predict(x_test)
# Making confusion matrix

cm=confusion_matrix(y_test,y_pred)

cm
# Heatmap of confusion matrix

sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")

plt.show()
print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")

print("Precision = ",precision_score(y_test,y_pred)*100,"%")

print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")
sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])

print ("Sensitivity =",sensitivity)

specificity= cm[0,0]/(cm[0,0] + cm[0,1])

print("Specificity =",specificity)
# calculating AUC

auc=roc_auc_score(y_test,y_pred)

auc
# calculating ROC curve

fpr,tpr,thresholds= roc_curve(y_test,y_pred)
# plotting the roc curve for the model

plt.plot([0,1],[0,1],linestyle="--")

plt.plot(fpr,tpr,marker=".")

plt.xlabel("False Positive Rate")

plt.ylabel("TruePositive Rate")

plt.title("ROC Curve")

plt.show()
# making object classifier of class RandomForestClassifier

from sklearn.ensemble import RandomForestClassifier

classifier=RandomForestClassifier(n_estimators=10,criterion="gini")
# Fitting training data set into classifier

classifier.fit(x_train,y_train)
# Predicting test results

y_pred=classifier.predict(x_test)
# Making confusion matrix

cm=confusion_matrix(y_test,y_pred)

cm
# Heatmap of confusion matrix

sns.heatmap(pd.DataFrame(cm),annot=True,cmap="Reds")

plt.show()
print("Accuracy = ",accuracy_score(y_test,y_pred)*100,"%")

print("Precision = ",precision_score(y_test,y_pred)*100,"%")

print("Recall Score = ",recall_score(y_test,y_pred)*100,"%")
sensitivity = cm[1,1]/(cm[1,1] + cm[1,0])

print ("Sensitivity =",sensitivity)

specificity= cm[0,0]/(cm[0,0] + cm[0,1])

print("Specificity =",specificity)
# calculating AUC

auc=roc_auc_score(y_test,y_pred)

auc
# calculating roc curve

fpr,tpr,thresholds= roc_curve(y_test,y_pred)
# plotting the roc curve for the model

plt.plot([0,1],[0,1],linestyle="--")

plt.plot(fpr,tpr,marker=".")

plt.xlabel("False Positive Rate")

plt.ylabel("TruePositive Rate")

plt.title("ROC Curve")

plt.show()
methods = ["Logistic Regression", "KNN", "SVM", "Naive Bayes", "Decision Tree", "Random Forest"]

accuracy = [88.5,63.93,83.6,83.6,77.04,85.24]

plt.subplots(figsize=(11,8))

sns.barplot(x=methods,y=accuracy)

plt.xlabel("Classifier")

plt.ylabel("Accuracy")

plt.title("Comparison between Classifiers")

plt.show()
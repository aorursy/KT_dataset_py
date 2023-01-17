import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # plotting library, for simple plots
import seaborn as sns # plotting utility

from sklearn.metrics import confusion_matrix # to plot heatmaps

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
prostate_cancer_df = pd.read_csv("../input/prostate-cancer/Prostate_Cancer.csv")
prostate_cancer_df.shape
# There are 100 rows with 10 columns
prostate_cancer_df.info()
# We can see detailed information of the dataframe
prostate_cancer_df.isnull().sum()
# We can clearly see that there is no unfilled column
prostate_cancer_df.head(15)
# Since we do not need id column, we can drop it safely!
prostate_cancer_df.drop(["id"], axis=1, inplace=True)
prostate_cancer_df.head(15)
# We are going to do classification on diagnosis_result column, so we are converting them into integer values
#1 to M and 0 to B using list comprehension method
prostate_cancer_df["diagnosis_result"] = [1 if element == "M" else 0 for element in prostate_cancer_df["diagnosis_result"]]
prostate_cancer_df.head(15)
prostate_cancer_df["diagnosis_result"].value_counts()

# We need to split the data into train-test values
x = prostate_cancer_df.drop(['diagnosis_result'], axis=1)
y = prostate_cancer_df["diagnosis_result"].values
# Observing our values
x.head(15)
y
# Splitting the data into train-test values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25, train_size=0.75, random_state=42)
# 25% is test data, 75% is train data
# You can change the test and train sizes in order to see the differences. Use floats. 
clf_names = [] 
clf_scores = []
# At the end we are gonig to compare all the methods that we used.
x_train
# Logistic Regression Classification
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train) #Fitting
print("Logistic Regression Classification Test Accuracy : {}".format(log_reg.score(x_test,y_test)))
clf_names.append("Logistic Regression")
clf_scores.append(log_reg.score(x_test,y_test))

# Confusion Matrix
y_pred = log_reg.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)

# Visualization Confusion Matrix
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(conf_mat,annot=True,linewidths=1.5,linecolor="#000000",fmt=".0f",ax=ax, cmap="RdGy")
plt.title("Heatmap of Logisctic Regression Classification")
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
# KNN Classification
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)  # 5 is default.
knn.fit(x_train,y_train) # Fitting
print("KNN Test Accuracy with n = 5: {}".format(knn.score(x_test,y_test)))
clf_names.append("KNN")
clf_scores.append(knn.score(x_test,y_test))

# Confusion Matrix
y_pred = knn.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)

# Visualization Confusion Matrix
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(conf_mat,annot=True,linewidths=1.5,linecolor="#000000",fmt=".0f",ax=ax, cmap="RdGy")
plt.title("Heatmap of KNN Classification")
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
# Support Vector Machine (SVM) Classification
from sklearn.svm import SVC
svm = SVC(random_state=42)
svm.fit(x_train,y_train) #Fitting
print("SVM Classification Test Accuracy : {}".format(svm.score(x_test,y_test)))
clf_names.append("SVM")
clf_scores.append(svm.score(x_test,y_test))

# Confusion Matrix
y_pred = svm.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)

# Visualization Confusion Matrix
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(conf_mat,annot=True,linewidths=1.5,linecolor="#000000",fmt=".0f",ax=ax, cmap="RdGy")
plt.title("Heatmap of SVM Classification")
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
# Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
naive_bayes.fit(x_test,y_test) # Fitting
print("Naive Bayes Classification Test Accuracy : {}".format(naive_bayes.score(x_test,y_test)))
clf_names.append("Naive Bayes")
clf_scores.append(naive_bayes.score(x_test,y_test))

# Confusion Matrix
y_pred = naive_bayes.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)

# Visualization Confusion Matrix
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(conf_mat,annot=True,linewidths=1.5,linecolor="#000000",fmt=".0f",ax=ax, cmap="RdGy")
plt.title("Heatmap of Naive Bayes Classification")
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
# Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train,y_train) # Fitting
print("Decision Tree Classification Test Accuracy : ",dec_tree.score(x_test,y_test))
clf_names.append("Decision Tree")
clf_scores.append(dec_tree.score(x_test,y_test))

# Confusion Matrix
y_pred = dec_tree.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)

# Visualization Confusion Matrix
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(conf_mat,annot=True,linewidths=1.5,linecolor="#000000",fmt=".0f",ax=ax, cmap="RdGy")
plt.title("Heatmap of Decision Tree Classification")
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rand_forest.fit(x_train,y_train) # Fitting
print("Random Forest Classification Test Accuracy : ",rand_forest.score(x_test,y_test))
clf_names.append("Random Forest")
clf_scores.append(rand_forest.score(x_test,y_test))

# Confusion Matrix
y_pred = rand_forest.predict(x_test)
conf_mat = confusion_matrix(y_test,y_pred)

# Visualization Confusion Matrix
fig, ax = plt.subplots(figsize=(7,7))
sns.heatmap(conf_mat,annot=True,linewidths=1.5,linecolor="#000000",fmt=".0f",ax=ax, cmap="RdGy")
plt.title("Heatmap of Random Forest Classification")
plt.xlabel("Predicted Values")
plt.ylabel("True Values")
plt.show()
fig = plt.figure(figsize=(10,10))
plt.bar(clf_names, clf_scores, color = "green", width = 0.5)
plt.ylim(0.6, 0.9)
plt.title("Prostate Cancer Predictions with ML")
plt.xlabel("Method Name")
plt.ylabel("Method Score")
plt.savefig("25test75train.png")
#plt.savefig("20test80train.png")
#plt.savefig("30test70train.png")

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
dataframe=pd.read_csv('../input/heart.csv')

dataframe.tail()

dataframe.info()
#%% x ve y axis

y=dataframe.target.values   # values => np array

x_data=dataframe.drop(["target"],axis=1)
y
#%% normalization   feature scaling

x=(x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
x.head()
#%% train test splitting

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)
# holding results in a list

scores_accuracy=[]
# Logistic regression classication

# LR with sklearn



from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()

lr.fit(x_train,y_train)



lr_score = lr.score(x_test,y_test)

scores_accuracy.append(["LR",lr_score])



print("test accuracy {}".format(lr.score(x_test,y_test)))
# LR confusion matrix

y_predict = lr.predict(x_test)

y_true = y_test

cm = confusion_matrix (y_true,y_predict)

f, ax =plt.subplots(figsize=(5,5))

sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_predict")

plt.ylabel("y_true")

plt.show()
# KNN  classification

# Knn with sklearn

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 9) # n_neighbors = k

knn.fit(x_train,y_train)

prediction = knn.predict(x_test)



knn_score = knn.score(x_test,y_test)

scores_accuracy.append(["KNN",knn_score])

print(" {} nn score: {} ".format(9,knn.score(x_test,y_test)))

# KNN confusion matrix

y_predict_knn = knn.predict(x_test)

y_true_knn = y_test

cm_knn = confusion_matrix (y_true_knn,y_predict_knn)

f, ax =plt.subplots(figsize=(5,5))

sns.heatmap(cm_knn,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_predict_knn")

plt.ylabel("y_true_knn")

plt.show()
# Findind k values in range(1,15)

score_list = []

for each in range(1,15):

    knn2 = KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,15),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()



# 3 is the best k value in range(1,15) i used 9 above.
# SVM Classification

# SVM with sklearn



from sklearn.svm import SVC



svm = SVC(random_state=1)

svm.fit(x_train,y_train)



svm_score = svm.score(x_test,y_test)

scores_accuracy.append(["SVM",svm_score])



print("accuracy of svm algo: ", svm.score(x_test,y_test))

# SVM confusion matrix

y_predict_svm = svm.predict(x_test)

y_true_svm = y_test

cm_svm = confusion_matrix (y_true_svm,y_predict_svm)

f, ax =plt.subplots(figsize=(5,5))

sns.heatmap(cm_svm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_predict_svm")

plt.ylabel("y_true_svm")

plt.show()
from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)

nb_score = nb.score(x_test,y_test)

scores_accuracy.append(["NB",nb_score])



print("print accuracy of naive bayes algo: ",nb.score(x_test,y_test))
# NB confusion matrix

y_predict_nb = nb.predict(x_test)

y_true_nb = y_test

cm_nb = confusion_matrix (y_true_nb,y_predict_nb)

f, ax =plt.subplots(figsize=(5,5))

sns.heatmap(cm_nb,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_predict_nb")

plt.ylabel("y_true_nb")

plt.show()
# check score regularly

# del scores_accuracy[index] if you needed

#del scores_accuracy[1]

scores_accuracy
# Decision Tree Classification

# Decision Tree With Sklearn

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)



dt_score = dt.score(x_test,y_test)

scores_accuracy.append(["DT",dt_score])



print("print accuracy of decision tree algo: ",dt.score(x_test,y_test))
# DT confusion matrix

y_predict_dt = dt.predict(x_test)

y_true_dt = y_test

cm_dt = confusion_matrix (y_true_dt,y_predict_dt)

f, ax =plt.subplots(figsize=(5,5))

sns.heatmap(cm_dt,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_predict_dt")

plt.ylabel("y_true_dt")

plt.show()
# Randon Forest Classification

#Random Forest With Sklearn



from sklearn.ensemble import RandomForestClassifier

rf= RandomForestClassifier(n_estimators=100,random_state=1) # n_estimators= number of trees

rf.fit(x_train,y_train)



rf_score = rf.score(x_test,y_test)

scores_accuracy.append(["RF",rf_score])



print("random forest result: ", rf.score(x_test,y_test))
# Rf confusion matrix

y_predict_rf = rf.predict(x_test)

y_true_rf = y_test

cm_rf = confusion_matrix (y_true_rf,y_predict_rf)

f, ax =plt.subplots(figsize=(5,5))

sns.heatmap(cm_rf,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)

plt.xlabel("y_predict_rf")

plt.ylabel("y_true_rf")

plt.show()
scores_accuracy
algorithms=("LR","DT","RF","KNN","NB","SVM")

scores = (lr_score,dt_score,rf_score,knn_score,nb_score,svm_score)

y_pos = np.arange(1,7)

colors = ("red","gray","purple","green","orange","blue")

plt.figure(figsize=(18,10))

plt.bar(y_pos,scores,color=colors)

plt.xticks(y_pos,algorithms,fontsize=18)

plt.yticks(np.arange(0.00, 1.01, step=0.05))

plt.grid()

plt.suptitle("Bar Chart Comparison of Models",fontsize=15)

plt.show()

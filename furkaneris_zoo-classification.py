# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings 

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
plt.style.use("ggplot")#plot style
zoo = pd.read_csv("../input/zoo.csv")#read data

zoo.head()
zoo.info()
print(zoo.class_type.value_counts())

plt.figure(figsize = (10,8))

sns.countplot(zoo.class_type)

plt.show()
data = zoo.copy()

data.drop("animal_name",axis = 1,inplace = True)
x = data.drop("class_type",axis = 1)# input data

y = data.class_type.values# target data
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.4,random_state = 42)

print("x_train shape : ",x_train.shape)

print("x_test shape : ",x_test.shape)

print("y_train shape : ",y_train.shape)

print("y_test shape : ",y_test.shape)
from sklearn.svm import SVC

svm = SVC(random_state = 42,kernel = "linear")

svm.fit(x_train,y_train)

y_pred_svm = svm.predict(x_test)

print("Train Accurary : ",svm.score(x_train,y_train))

print("Test Accuray : ",svm.score(x_test,y_test))
from sklearn.metrics import confusion_matrix,classification_report

cm_svm = confusion_matrix(y_test,y_pred_svm)

cr_svm = classification_report(y_test,y_pred_svm)

print("confusion matrix : \n",cm_svm)

print("classification report : \n",cr_svm)
plt.figure(figsize = (10,8))

sns.heatmap(cm_svm,annot = True,cmap = "Blues",xticklabels = np.arange(1,8),yticklabels = np.arange(1,8))

plt.show()
from sklearn.neighbors import KNeighborsClassifier

scr_max = 0

knn_test_score_list = []

knn_train_score_list = []



for i in range(1,x_train.shape[0]+1):

    knn = KNeighborsClassifier(n_neighbors = i)

    knn.fit(x_train,y_train)

    knn_test_scr = knn.score(x_test,y_test)

    knn_test_score_list.append(knn_test_scr)

    knn_train_scr = knn.score(x_train,y_train)

    knn_train_score_list.append(knn_train_scr)

    if knn_test_scr >= scr_max:

        scr_max = knn_test_scr

        index = i



print("Best K value = ",index)

print("Best score = ",scr_max)



plt.figure(figsize = (15,10))

plt.plot(range(1,x_train.shape[0]+1),knn_test_score_list,label = "test")

plt.plot(range(1,x_train.shape[0]+1),knn_train_score_list,label = "train")

plt.legend()

plt.xlabel("K Values")

plt.ylabel("Scores")

plt.show()
knn = KNeighborsClassifier(n_neighbors = 1)

knn.fit(x_train,y_train)

y_pred_knn = knn.predict(x_test)

cr_knn = classification_report(y_test,y_pred_knn)

cm_knn = confusion_matrix(y_test,y_pred_knn)

print("confusion matrix : \n",cm_knn)

print("classification report : \n",cr_knn)
plt.figure(figsize = (10,8))

sns.heatmap(cm_knn,annot = True,xticklabels = np.arange(1,8),yticklabels = np.arange(1,8))

plt.show()
from sklearn.tree import DecisionTreeClassifier

dec_tree = DecisionTreeClassifier(random_state = 42)

dec_tree.fit(x_train,y_train)

y_pred_tree = dec_tree.predict(x_test)

print("Test Accurary : ",dec_tree.score(x_test,y_test))

print("Train Accurary : ",dec_tree.score(x_train,y_train))
cm_tree = confusion_matrix(y_test,y_pred_tree)

cr_tree = classification_report(y_test,y_pred_tree)

print("confusion matrix : \n",cm_tree)

print("classification report : \n",cr_tree)
plt.figure(figsize = (10,8))

sns.heatmap(cm_tree,annot = True,xticklabels = np.arange(1,8),yticklabels = np.arange(1,8),cmap = "Greens")

plt.show()
from sklearn.ensemble import RandomForestClassifier

s_max = 0

rf_train_score_list = []

rf_test_score_list = []



for i in range(1,x_train.shape[0]+1):

    rf = RandomForestClassifier(n_estimators = i,random_state = 42)

    rf.fit(x_train,y_train)

    test_score = rf.score(x_test,y_test)

    rf_test_score_list.append(test_score)

    train_score = rf.score(x_train,y_train)

    rf_train_score_list.append(train_score)

    if test_score >= s_max :

        s_max = test_score

        index = i



print("Best Score = ",s_max)

print("Best n_estimators = ",index)



plt.figure(figsize = (10,8))

plt.plot(range(1,x_train.shape[0]+1),rf_test_score_list,label = "test")

plt.plot(range(1,x_train.shape[0]+1),rf_train_score_list,label = "train")

plt.legend()

plt.xlabel("n estimators")

plt.ylabel("Scores")

plt.show()
rf = RandomForestClassifier(n_estimators = 60,random_state = 42)

rf.fit(x_train,y_train)

y_pred_rf = rf.predict(x_test)

cm_rf = confusion_matrix(y_test,y_pred_rf)

cr_rf = classification_report(y_test,y_pred_rf)

print("confusion matrix : \n",cm_rf)

print("classification report : \n",cr_rf)
plt.figure(figsize = (10,8))

sns.heatmap(cm_rf,annot = True,xticklabels = np.arange(1,8),yticklabels = np.arange(1,8),cmap = "Greens")

plt.show()
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(x_train,y_train)

y_pred_lr = log_reg.predict(x_test)

print("Test Accurary : ",log_reg.score(x_test,y_test))

print("Train Accurary : ",log_reg.score(x_train,y_train))
cm_lr = confusion_matrix(y_test,y_pred_lr)

cr_lr = classification_report(y_test,y_pred_lr)

print("confusion matrix : \n",cm_lr)

print("classification report : \n",cr_lr)
plt.figure(figsize = (10,8))

sns.heatmap(cm_lr,annot = True,xticklabels = np.arange(1,8),yticklabels = np.arange(1,8),cmap = "Reds")

plt.show()
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

y_pred_nb = nb.predict(x_test)

print("Test Accurary : ",nb.score(x_test,y_test))

print("Train Accurary : ",nb.score(x_train,y_train))
cm_nb = confusion_matrix(y_test,y_pred_nb)

cr_nb = classification_report(y_test,y_pred_nb)

print("confusion matrix : \n",cm_nb)

print("classification report : \n",cr_nb)
plt.figure(figsize = (10,8))

sns.heatmap(cm_nb,annot = True,xticklabels = np.arange(1,8),yticklabels = np.arange(1,8),cmap = "Reds")

plt.show()
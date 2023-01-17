# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Import library
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import data
data = pd.read_csv('../input/column_2C_weka.csv')
# data info
data.info()
# data head
data.head()
# before cleanin data and virtualize
abnormal = data[data['class'] == 'Abnormal']
normal = data[data['class'] == 'Normal']
plt.scatter(abnormal.pelvic_incidence,abnormal.pelvic_radius,color='red',label='Abnormal',alpha=0.4)
plt.scatter(normal.pelvic_incidence,normal.pelvic_radius,color='green',label='Normal',alpha=0.4)
plt.xlabel('pelvic_incidence')
plt.ylabel('pelvic_radius')
plt.legend()
plt.show()
# classification and normalization
data['class'] = [0 if x == 'Abnormal' else 1 for x in data['class']] # clasifications
y = data['class']
x_data = data.drop(['class'],axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data)) # normalizations
# data train and test splite
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.4, random_state = 1)
#print(x_train, x_test, y_train, y_test )
# logistic regression
lr = LogisticRegression()
lr.fit(x_train,y_train)
print("test accuracy {}".format(lr.score(x_test,y_test)))
# knn
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
print ('k: {} values score: {}'.format(4,knn.score(x_test, y_test)))
# Find the best k value and virtualization
score_list = []

for x in range(1,15):
    knn_n = KNeighborsClassifier(n_neighbors=x)
    knn_n.fit(x_train,y_train)
    score_list.append(knn_n.score(x_test, y_test))
    print ('k: {} values score: {}'.format(x,knn_n.score(x_test, y_test)))

plt.plot(range(1,15), score_list)
plt.xlabel('k values')
plt.ylabel('accurasi')
plt.show()
# SVM
svm = SVC(random_state=1)
svm.fit(x_train, y_train)
print ('accuracy of svmalgo: ',svm.score(x_test, y_test))
# Naive Bayes
gnb = GaussianNB()
gnb.fit(x_train, y_train)
print ('accuracy of naive bayes: ',gnb.score(x_test, y_test))
# Decision Classification
dtc = DecisionTreeClassifier()
dtc.fit(x_train, y_train)
print ('accuracy of naive decision tree: ',dtc.score(x_test, y_test))
# Random Forest Classification
rf = RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train,y_train)
print ('Random forest algo result: ',rf.score(x_test,y_test))
# Confusion matrix
y_predict = rf.predict(x_test)
y_true = y_test
cm = confusion_matrix(y_true,y_predict)
print ('confusion matrix: ',cm)
# Confusion matrix Virtualize
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor='red',fmt='.0f',ax=ax)
plt.xlabel('y_pred')
plt.ylabel('y_true')
plt.show()
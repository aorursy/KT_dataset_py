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
import numpy as np

import pandas as pd

import sklearn

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
data = pd.read_csv("../input/mushrooms.csv")
data.shape
data.head()
#ALl the variables are in string format. Convert categorical variables to integer using label encoder

from sklearn.preprocessing import LabelEncoder

lbl = LabelEncoder()
for col in data.columns:

    data[col]=lbl.fit_transform(data[col])    
#check the data after label encoding

data.head()
#split the x and y variables

y=data['class']

x=data.iloc[:,1:23]
#check shape of new variables

x.shape
y.shape
#check data

x.head
y.head
#I want to use PCA on this data. First normalise the data using StandardScalar so that the data is now between -1 and 1

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

x = sc.fit_transform(x)
#see the Standardised data

print(x)
#using principal component analysis

#Even though the number of variables is not too high, I would still like to use PCA to see which variables describe the maximum variance in data

from sklearn.decomposition import PCA

pca = PCA()

x_pca = pca.fit_transform(x)
#plot a Scree plot of the Principal Components

plt.figure(figsize=(16,11))

plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')

plt.grid()
#from the graph, first 17 components describe the maximum variance(more than 90% of the data). We shall use them for our subsequent analysis.

new_pca = PCA(n_components=17)
x_new = new_pca.fit_transform(x)
#using KMeans to plot the clusters. We know that we habe 2 classes of the target variable. So n_clusters=2

from sklearn.cluster import KMeans

k_means = KMeans(n_clusters=2)
k_means.fit_predict(x_new )
#plot the clusters.

colors = ['r','g']

for i in range(len(x_new)):

    plt.scatter(x_new[i][0], x_new[i][1], c=colors[k_means.labels_[i]], s=10)

plt.show()
#2 distinct clusters are created. Data points are far apart 

x_new.shape
#separate the train and test data

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_new, y, test_size = 0.25, random_state = 6)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
#using Logistic regression to build the first model

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(x_train, y_train)

lr_predict =lr.predict(x_test)
lr_predict_prob = lr.predict_proba(x_test)
print(lr_predict)

print(lr_predict_prob[:,1])
#import metrics

from sklearn.metrics import confusion_matrix, accuracy_score
lr_conf_matrix = confusion_matrix(y_test, lr_predict)

lr_accuracy = accuracy_score(y_test, lr_predict)
print(lr_conf_matrix)

print(lr_accuracy)
#roc curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test,lr_predict_prob[:,1] )
#auc score

from sklearn.metrics import auc

lr_auc = auc(fpr, tpr)

print(lr_auc)
#plotting ROC curve

plt.figure(figsize=(10,9))

plt.plot(fpr, tpr, label = 'AUC= %0.2f' % lr_auc )

plt.plot([0,1],[0,1], linestyle = '--')

plt.legend()
#Using Naive Bayes

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()

gnb.fit(x_train, y_train)

gnb_predict = gnb.predict(x_test)

gnb_predict_prob = gnb.predict_proba(x_test)
print(gnb_predict)

print(gnb_predict_prob)
gnb_conf_matrix = confusion_matrix(y_test, gnb_predict)

gnb_accuracy_score = accuracy_score(y_test, gnb_predict)
print(gnb_conf_matrix)

print(gnb_accuracy_score)
#calculate ROC and AUC

fpr, tpr, thresholds = roc_curve(y_test, gnb_predict_prob[:,1])

#print auc

gnb_auc = auc(fpr, tpr)

print(gnb_auc)
#plot ROC curve

plt.figure(figsize=(10,9))

plt.plot(fpr, tpr, label = 'AUC %0.2f' % gnb_auc)

plt.plot([0,1],[0,1], linestyle = '--')

plt.legend()
#lets use Decision Trees to classify 

#use the number of trees as 10 first

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=10)
dt.fit(x_train,y_train)

dt_predict = dt.predict(x_test)

dt_predict_prob = dt.predict_proba(x_test)
from sklearn.metrics import confusion_matrix, accuracy_score
dt_conf_matrix = confusion_matrix(y_test, dt_predict)

dt_accuracy_score = accuracy_score(y_test, dt_predict)
print(dt_conf_matrix)

print(dt_accuracy_score)
#calculate auc and plot roc

from sklearn.metrics import roc_curve, auc

fpr, tpr, thresholds = roc_curve(y_test, dt_predict_prob[:,1])

dt_auc = auc(fpr, tpr)

print(dt_auc)
#plot ROC curve

plt.figure(figsize=(10,9))

plt.plot(fpr, tpr, label = 'AUC %0.2f' % dt_auc)

plt.plot([0,1],[0,1], linestyle = '--')

plt.xlabel('False Positive rate')

plt.ylabel('True Positive rate')

plt.legend()

plt.grid()

#using random forest

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=10) #10 trees

rf.fit(x_train, y_train)

rf_predict = rf.predict(x_test)

rf_predict_prob = rf.predict_proba(x_test)
rf_conf_matrix = confusion_matrix(y_test,rf_predict)

rf_accuracy_score = accuracy_score(y_test, rf_predict)
print(rf_conf_matrix)

print(rf_accuracy_score)

#random forest has a higher accuracy score than the decision tree

#Decision tree = 99.3

#Random forest = 99.9
fpr, tpr, thresholds = roc_curve(y_test, rf_predict_prob[:,1])

rf_auc = auc(fpr, tpr)

print(rf_auc)
#plot the ROC curve

plt.figure(figsize=(10,9))

plt.plot(fpr, tpr, label = 'AUC: %0.2f' % rf_auc)

plt.plot([1,0],[1,0], linestyle = '--')

plt.legend(loc=0)

plt.xlabel('False Positive rate')

plt.ylabel('True Positive rate')

plt.grid()
#how would an unsupervised algo like MeanShift or DBScan work? Let's find out

from sklearn.cluster import MeanShift

ms = MeanShift()

ms.fit(x_new)
#print the labels and the cluster centers (I will be calling them centroids)

ms_labels = ms.labels_

ms_centroids = ms.cluster_centers_

print(ms_labels)

print(ms_centroids)
#np.unique will give us one count of each label. 

n_clusters = len(np.unique(ms_labels))

print(n_clusters)
#let's plot the clusters and see how different they are from our original cluster of KMeans'

plt.figure(figsize=(10,9))

colors = ['r','g','y','b']

for i in range(len(x_new)):

    plt.scatter(x_new[i][0], x_new[i][1], c=colors[ms_labels[i]], s=5)

#print cluster centers

#Cluster centers are x's in blue

plt.scatter(ms_centroids[:,0], ms_centroids[:,1], marker='x')

plt.show()

#Considerably different!
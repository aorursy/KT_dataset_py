"""Team Members
Hrishikesh S. 01FB16ECS139
Krishna Sidharth Sagere 01FB16ECS169
Sankarshana 01FB16ECS342
"""
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Absenteeism_at_work.csv')
data.describe()
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
# create training and testing vars
y = data['Absenteeism time in hours']
X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.2)
print("Training dataset shape = X-train : ",X_train.shape," Y-Train : ", y_train.shape)
print("Test dataset shape = X-test : ",X_test.shape, "Y-test: " ,y_test.shape)
print("Training dataset features\n", X_train.head())
print("Training dataset labels\n", y_train.head())
n_row = X_train.shape[0]
n_col = X_train.shape[1]
print(n_row)
print(n_col)
from collections import Counter
targets = data['Absenteeism time in hours']
count = Counter(targets)
KEYS = list(count.keys())
VALUES = list(count.values())
print("All the classes = ", KEYS)
print("Frequency of each of the given classes = ", VALUES)
print("Number of classes = ", len(KEYS))
"""Data Pre-processing"""
#train set
data_list = []
for i in range(n_row):
    temp = []
    for j in range(n_col-1):
        k = X_train.iloc[i,j]
        temp.append(k)
    data_list.append(temp)
print("Row 1 of data_list = ", data_list[0])
print("Row 2 of data_list = ", data_list[1])
#test set
n_row = X_train.shape[0]
n_col = X_train.shape[1]
print(n_row)
print(n_col)
test_data_list = []
for i in range(n_row):
    temp = []
    for j in range(n_col-1):
        k = X_train.iloc[i,j]
        temp.append(k)
    test_data_list.append(temp)
print("Row 1 of data_list = ", test_data_list[0])
print("Row 2 of data_list = ", test_data_list[1])
n_row = data.shape[0]
n_col = data.shape[1]
print(n_row)
print(n_col)
"""K-means clustering"""
import sklearn.cluster
import math
from scipy import spatial

kmeans = sklearn.cluster.KMeans(n_clusters = len(KEYS),
                               random_state = 0,
                               n_init = 200,
                               max_iter = 500)
kmeans.fit(data_list)
clust_labels = kmeans.predict(test_data_list)
print(kmeans.labels_)
print(y_test)
import seaborn as sns
import matplotlib.pyplot as plt
df = data 
df.head()
df.describe()
sns.boxplot(df['Absenteeism time in hours'])
median = np.median(df['Absenteeism time in hours'])
q75, q25 = np.percentile(df['Absenteeism time in hours'], [75 ,25])
iqr = q75 - q25
print("Lower outlier bound:",q25 - (1.5*iqr))
print("Upper outlier bound:",q75 + (1.5*iqr))
#dropping the following outliers above 17
df= df[df['Absenteeism time in hours']<=17]
df= df[df['Absenteeism time in hours']>=-7]
#Splitting data into training and testing
from sklearn.model_selection import train_test_split
y=df['Absenteeism time in hours']
X=df.drop('Absenteeism time in hours',axis=1)#Extracting only the features
X.describe()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
print(df.shape)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
print("Number of unique ouput classes after preprocessing:",((np.unique(y_train))))

 #Calculating the correlation of the above described variables
cor = df.corr()
plt.figure(figsize=(10,10))
sns.heatmap(cor, square = True) 
#Preprocessing (the next step)
#Dropping the following attributes due to multi-collinearity
X_train=X_train.drop('Service time',axis=1) #removing service time column
X_test=X_test.drop('Service time',axis=1)    
 #Normalizing features
from sklearn import preprocessing
scaled_X_train = preprocessing.scale(X_train)
scaled_X_test = preprocessing.scale(X_test)
scaled_X_train.shape
"""Model 1 : KNN"""
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report

#We kept ID attribute as we observed that ID was repeated and had a pattern with labels
knn = KNeighborsClassifier(n_neighbors=19)
knn.fit(scaled_X_train, y_train)
y_pred = knn.predict(scaled_X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("******      *********\n")
print(classification_report(y_test, y_pred))
"""Model 2 : SVM (SV classifier)"""
from sklearn import metrics, svm
from sklearn.svm import SVC

svm=svm.SVC()
svm.fit(scaled_X_train, y_train)
y_pred = svm.predict(scaled_X_test)
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("********    *********\n")
print(classification_report(y_test, y_pred))
"""Conclusion


KNN (Kth nearest neighbours)classification   ----48.57%

 disadvantage is 
It doesn't handle categorical variables very well


SVM (support vector machines)  ----- 52.85% Accuracy
The kernel trick is real strength of SVM. With an appropriate kernel function, 
we can solve any complex problem.
Unlike in neural networks, SVM is not solved for local optima.
It scales relatively well to high dimensional data.
SVM models have generalization in practice, the risk of overfitting is less in SVM.
They are relatively easy to calibrate, as opposed to other kernels.
It has localized and finite response along the entire x-axis."""




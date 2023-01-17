#Basic Headers
import numpy as np
import pandas as pd

#Understanding the system
import os
print(os.name)
print(os.getcwd())

#Looking at the dataset.
df = pd.read_csv("../input/Absenteeism_at_work.csv")
print(df, "\n\n\n")
print(df.describe(), "\n\n\n")
print(df.info())
#Mean Shift Clustering
from sklearn.cluster import MeanShift
from sklearn.cluster import MeanShift, estimate_bandwidth
X = df

#Performing MeanShift Clustering
#bandwidth = estimate_bandwidth(X)
ms = MeanShift(bandwidth = 2.001, bin_seeding=True).fit(X) #Returns labels for each row(check using len(clustering))
labels = ms.labels_  #Retrive the labels for each datapoint
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("Estimated cluster centers : \n", cluster_centers, "\n")
print("Number of estimated clusters : %d" % n_clusters_, "\n")
print("Labels are : ", labels_unique)
#DBSCAN
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# #############################################################################
# Compute DBSCAN
db = DBSCAN(eps = 8.5, min_samples = 5, metric = 'euclidean').fit(X)
#db = DBSCAN().fit(X)
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
#KNN
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

X = df.iloc[:, :-1].values  
y = df.iloc[:, 14].values

from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)  

from sklearn.neighbors import KNeighborsClassifier
# Calculating error for K values between 1 and 50
error = []
for i in range(1, 50):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 50), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')
plt.show()

index_min = np.argmin(error)
classifier = KNeighborsClassifier(n_neighbors = index_min+1)  
classifier.fit(X_train, y_train)  

y_pred = classifier.predict(X_test)  

count = 0
for i in range(len(y_pred)):
    if(y_pred[i] == y_test[i]):
        count += 1

print("Accuracy = ", count/len(y_pred), "\n\n")
        
from sklearn.metrics import classification_report, confusion_matrix 
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred),"\n\n Report:")  
print(classification_report(y_test, y_pred)) 
#Random Forest
# Import train_test_split function
from sklearn.model_selection import train_test_split

X = df.iloc[:, :-1].values  
y = df.iloc[:, 14].values

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

#Scaling the data
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

#Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier

# Calculating error for n_estimators values between 300 and 500
error = []
for i in range(300, 500):  
    clf = RandomForestClassifier(n_estimators = i)
    clf.fit(X_train, y_train)
    pred_i = clf.predict(X_test)
    error.append(np.mean(pred_i != y_test))
    
plt.figure(figsize=(12, 6))  
plt.plot(range(300, 500), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate Plot')  
plt.xlabel('n_estimators Value')  
plt.ylabel('Mean Error')
plt.show()

#Create a Gaussian Classifier
index_min = np.argmin(error)
clf=RandomForestClassifier(n_estimators = index_min+300)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_train,y_train)

y_pred=clf.predict(X_test)

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred), "\n\n")

from sklearn.metrics import classification_report, confusion_matrix 
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred),"\n\n Report:")  
print(classification_report(y_test, y_pred)) 
#SVM
import pandas as pd
from sklearn.svm import SVC
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("../input/Absenteeism_at_work.csv")

X = df.iloc[:, :-1].values  
y = df.iloc[:, 14].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

#Scaling the data
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

#####Training, prediction
svclassifier = SVC(kernel='sigmoid')  
svclassifier.fit(X_train, y_train) 
y_pred = svclassifier.predict(X_test)  

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred), "\n\n")

from sklearn.metrics import classification_report, confusion_matrix 
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred),"\n\n Report:")  
print(classification_report(y_test, y_pred)) 
#Some comments on Unsupervised Clustering Algorithms:

'''Here we have used two clustering algorithms, Mean Shift Clustering and DBSCAN.
These methods were chosen because the the number of clusters are identified by the algorithms themselves.
However we must understand that clustering algorithms are used to gain more insight about the data and find
structure/patterns within the datapoints if there is any.

Clustering methods are not used for prediction or classification purposes and hence there is no meaning in evaluating
the performance of such models in tasks such as prediction of hours of absenteeism.
Even if we were to come up with a way to evaluate the models, their accuracies would be much greater than the accuracy of
any other classification algorithm as there is a significant decrease in the number of classes while clustering. Also
splitting the data for a clustering algorithm into training and testing sets is not meaningful.

Here due to a large number of attributes/features in the dataset it is hard to visualize the clusters. This inturn makes it
difficult to tune the parameters of the model being fit. Untill somekind of dimensionality reduction is not applied on the
data it would be hard to come up with a good clustering algorithm.

As the focus of the assignment was on predection, we have not focused much on the clustering techniques. Nevertheless, we have
tried performing some basic clustering just to get a feel of how to go about using different algorithms.'''
#Conclusion:

'''Random Forest builds multiple decision trees and merges them together to get a more stable and accurate result.
The major advantage of Random Forest is that it can be used for both regression and classification problems. Random Forest
can handle categorical variables well and high dimensional spaces. Random Forest adds additional randomness to the model,
while growing the trees. Instead of searching for the most important feature while splitting a node, it searches for the best
feature among a random subset of features. This results in wide diversity that generally results in a better model. Random Forest
will also not overfit the model, as there are enough trees in the forest. Random Forest works very well with a large number of training
examples, unlike SVMs, which are inefficient to train in such a case (due to the large time required to train a large number of samples).

SVMs can be used with a non linear kernel when the problem cannot be linearly separable. SVMs can be used for text classification.
KNN is robust to noisy training data and is effective in case of a large number of training examples. However, in KNN, we would have to
determine the value of K and the type of distance to be used, which is computationally taxing as we would have to find the distances for each query
to come up with he most optimum distance. Unlike KNN or SVMs, Random Forest works “out of the box” and is therefore popular. '''
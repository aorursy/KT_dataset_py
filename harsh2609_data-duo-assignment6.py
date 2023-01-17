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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score  
import random
from sklearn import metrics
import sklearn
from math import sqrt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
df=pd.read_csv("../input/Absenteeism_at_work.csv")

#Dropping outliers as they affect KNN,KMEANS AND SVM
sns.boxplot(df['Absenteeism time in hours'])
median = np.median(df['Absenteeism time in hours'])
q75, q25 = np.percentile(df['Absenteeism time in hours'], [75 ,25])
iqr = q75 - q25
print("Lower outlier bound:",q25 - (1.5*iqr))
print("Upper outlier bound:",q75 + (1.5*iqr))
#dropping the following outliers above 17
df= df[df['Absenteeism time in hours']<=17]
df= df[df['Absenteeism time in hours']>=-7]


#Separating Lables and Attributes
X = df.iloc[:, :-1].values  
Y = df.iloc[:, 14].values  

#Spliting the dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20,random_state=850)

Y_frame=pd.DataFrame(Y_train)

#Scaling the data
scaler = StandardScaler()  
scaler.fit(X_train)
X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test)
#Performing KNN

#Selecting K: Iterate over a range of values
min=100
K =range(1,41)
for k in K:
    classifier = KNeighborsClassifier(n_neighbors=k)  
    classifier.fit(X_train,Y_train) 
    Y_pred=classifier.predict(X_test) 
    error = np.mean(Y_pred!=Y_test)
    if(error<min):
        min =error
        clust_num=k
#print(clust_num,"--------------->",min)
    
    
classifier = KNeighborsClassifier(n_neighbors=clust_num)  
classifier.fit(X_train,Y_train) 
Y_pred=classifier.predict(X_test) 
error = np.mean(Y_pred!=Y_test)
#print(error)

#print(classification_report(Y_test, Y_pred))
print("Accuracy:",accuracy_score(Y_test,Y_pred))
print("Rmse:",sqrt(mean_squared_error(Y_test, Y_pred)))
#print(Y_frame)
#print(Y_train)
#KMEANS
distortions = []
K = range(5,20)
for k in K:
    kmeanModel = KMeans(n_clusters=k,random_state=123).fit(X_train)
    kmeanModel.fit(X_train)
    distortions.append(sum(np.min(cdist(X_train, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X_train.shape[0])

# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()
kmeans=KMeans(n_clusters=11,random_state=246)
kmeans.fit(X_train)
mean_predict=kmeans.predict(X_test)

labels=dict()
cluster=dict()
#print(X_train[0])
label=kmeans.predict(X_train[0].reshape(1,-1))
#print(label)
for i in range(len(X_train)):
    label=kmeans.predict(X_train[i].reshape(1,-1))
    if(label[0] not in labels.keys()):
        labels[label[0]]=[]
    labels[label[0]].append(Y_frame.iloc[i])
for key in labels:
    cluster[key]=pd.Series(labels[key]).value_counts().index[0]
print(cluster)

def accuracy(actual,predict):
    count=0
    if(len(actual)!=len(predict)):
        print("Shape Error")
    for i in range(len(predict)):
        ele=predict[i]
        #print(cluster[ele][0])
        #print(actual.iloc[i,0])
        #print("-------------------------------->")
        if(actual.iloc[i,0]==cluster[ele][0]):
            count=count+1
    return(float(count/len(actual)))
#print(classification_report(Y_test,mean_predict))
print("Accuracy:",accuracy(pd.DataFrame(Y_test),mean_predict))
print("Rmse:",sqrt(mean_squared_error(Y_test, mean_predict)))
#SVM
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, Y_train) 
Y_pred_svm = svclassifier.predict(X_test) 
#print(classification_report(Y_test,Y_pred_svm))
print("Rmse:",sqrt(mean_squared_error(Y_test, Y_pred_svm)))
print("Accuracy:",accuracy_score(Y_test,Y_pred_svm))
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/xAPI-Edu-Data.csv')
dataset.head(5)
df = dataset[['gender','PlaceofBirth','StageID','Topic','raisedhands','VisITedResources','AnnouncementsView','Discussion', 'ParentAnsweringSurvey','ParentschoolSatisfaction','StudentAbsenceDays', 'Class']]
df.head()
df.groupby(['ParentschoolSatisfaction'])['Class'].value_counts(normalize=True)

df.groupby(['ParentAnsweringSurvey'])['ParentschoolSatisfaction'].value_counts(normalize=True)
df.groupby(['ParentAnsweringSurvey'])['Class'].value_counts(normalize=True)
df2 = dataset[['gender','raisedhands','VisITedResources','AnnouncementsView','Discussion','StudentAbsenceDays', 'Class']]
df2.head()
df2['raisedhands'] = pd.cut(df2.raisedhands, bins=3, labels=np.arange(3), right=False)
df2.groupby(['raisedhands'])['Class'].value_counts(normalize=True)
df2['VisITedResources'] = pd.cut(df2.VisITedResources, bins=3, labels=np.arange(3), right=False)
df2.groupby(['VisITedResources'])['Class'].value_counts(normalize=True)
df2['AnnouncementsView'] = pd.cut(df2.AnnouncementsView, bins=3, labels=np.arange(3), right=False)
df2.groupby(['AnnouncementsView'])['Class'].value_counts(normalize=True)
df2['Discussion'] = pd.cut(df2.Discussion, bins=3, labels=np.arange(3), right=False)
df2.groupby(['Discussion'])['Class'].value_counts(normalize=True)
df2.groupby(['StudentAbsenceDays'])['Class'].value_counts(normalize=True)
df2 = dataset[['gender','raisedhands','VisITedResources','AnnouncementsView','Discussion','StudentAbsenceDays', 'Class']]
df2.tail()

correlation = df2[['raisedhands','VisITedResources','AnnouncementsView','Discussion']].corr(method='pearson')
correlation
df2 = pd.concat([df2,pd.get_dummies(df2['gender'], prefix='gender_')], axis=1)
df2 = pd.concat([df2,pd.get_dummies(df2['StudentAbsenceDays'], prefix='absence_')], axis=1)
df2 = pd.concat([df2,pd.get_dummies(df2['Class'], prefix='class_')], axis=1)

df2.drop(['gender'], axis = 1,inplace=True)
df2.drop(['StudentAbsenceDays'], axis = 1,inplace=True)
df2.drop(['Class'], axis = 1,inplace=True)

df2.head()
from sklearn.cluster import KMeans
from sklearn import preprocessing
X = df2[['raisedhands', 'VisITedResources']].values
#NORMALIZE OUR ARRAY
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(X)
#GET X AXIS
X = pd.DataFrame(x_scaled).values
X[:5]
wcss = []
 
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++')
    kmeans.fit(X)
    #print (i,kmeans.inertia_)
    wcss.append(kmeans.inertia_)  
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('N of Clusters')
plt.ylabel('WSS') #within cluster sum of squares
plt.show()
kmeans = KMeans(n_clusters = 3, init = 'k-means++')
kmeans.fit(X)
k_means_labels = kmeans.labels_
k_means_cluster_centers = kmeans.cluster_centers_
import matplotlib.pyplot as plt

plt.scatter(X[:, 0], X[:,1], s = 10, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 30, c = 'red',label = 'Centroids')
plt.title('Students Clustering')
plt.xlabel('RaisedHands')
plt.ylabel('VisITedResources')
plt.legend()

plt.show()
df3 = dataset[['raisedhands','VisITedResources','AnnouncementsView','Discussion','Class']]
df3.head()
y = df3['Class'].values
y[0:5]
X = df3[['raisedhands', 'VisITedResources', 'AnnouncementsView', 'Discussion']].values
X= preprocessing.StandardScaler().fit(X).transform(X)
X[:5]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
#REBUILDING THE MODEL WITH BEST K
k = 4
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh
from sklearn.tree import DecisionTreeClassifier
Dtree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
Dtree.fit(X_train,y_train)
Dtree
from sklearn import svm
supMac = svm.SVC(kernel='rbf', gamma='auto')
supMac.fit(X_train, y_train) 
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear', multi_class='auto').fit(X_train,y_train)
LR
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
knn_yhat = neigh.predict(X_test)
print("KNN Jaccard index: %.2f" % jaccard_similarity_score(y_test, knn_yhat))
print("KNN F1-score: %.2f" % f1_score(y_test, knn_yhat, average='weighted') )
dtree_yhat = Dtree.predict(X_test)
print("Decision Tree Jaccard index: %.2f" % jaccard_similarity_score(y_test, dtree_yhat))
print("Decision Tree F1-score: %.2f" % f1_score(y_test, dtree_yhat, average='weighted') )
svm_yhat = supMac.predict(X_test)
print("SVM Jaccard index: %.2f" % jaccard_similarity_score(y_test, svm_yhat))
print("SVM F1-score: %.2f" % f1_score(y_test, svm_yhat, average='weighted') )
LR_yhat = LR.predict(X_test)
LR_yhat_prob = LR.predict_proba(X_test)
print("LR Jaccard index: %.2f" % jaccard_similarity_score(y_test, LR_yhat))
print("LR F1-score: %.2f" % f1_score(y_test, LR_yhat, average='weighted') )
print("LR LogLoss: %.2f" % log_loss(y_test, LR_yhat_prob))

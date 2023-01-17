import numpy as np # linear algebra 
import matplotlib.pyplot as plt
import scipy.linalg as la

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import statistics

        
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.model_selection import KFold 

from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn import svm
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# import pandas as pd #pandas lib  
url="/kaggle/input/irisdata/Iris.csv" # data 
# load dataset into Pandas DataFrame

df=pd.read_csv(url, names=['Id', 'sepal length','sepal width','petal length','petal width','target']) # 4 features 5 target  reduce 4 features into 3
# df=df.drop(['Id'])
features=['sepal length', 'sepal width', 'petal length', 'petal width'] # naming features in data array

# Separating out the target
y=df.loc[:,['target']].values
df
df.describe()
y2=df.loc[1:,'target'].values #y series into arry

x2=df.loc[1:,features].values #data frame into 2 array

x2
y2

kf = KFold(n_splits=9)
kf.get_n_splits(x2)

for train_index, test_index in kf.split(x2):
    X_train, X_test = x2[train_index], x2[test_index]
for ytrain_index, ytest_index in kf.split(y2):
    y_train, y_test = y2[ytrain_index], y2[ytest_index]
train_index.shape + test_index.shape
count1 = df.loc[1:,'Id'].values 
countx = []
for j in range (x2[:,0].size):
    countx.append(j)
# countx

county = []
for j in range (y2[:].size):
    county.append(j)
# county
countx
print('Plotting Sepel Length')
plt.scatter(x2[:,0], countx)
print('Plotting Sepel Width')
plt.scatter(x2[:,1], countx)
print('Plotting Petal Length')
plt.scatter(x2[:,2], countx)
print('Plotting Petal Width')
plt.scatter(x2[:,3], countx)
plt.scatter(x2[:,0], countx)
plt.scatter(x2[:,1], countx)
plt.scatter(x2[:,2], countx)
plt.scatter(x2[:,3], countx)
knn_model=KNeighborsClassifier(n_neighbors=5, algorithm='brute', p=2) # p=2 means euclidean dist

# Train the model using the training sets
knn_model.fit(X_train,y_train)

#Predict Output
knn_pred=knn_model.predict(X_test)
print(accuracy_score(y_test, knn_pred)*100)

naive_model = GaussianNB()

# Train the model using the training sets
naive_model.fit(X_train,y_train)

#Predict Output
naive_pred= naive_model.predict(X_test)
print(accuracy_score(y_test, naive_pred)*100)

X_train1, X_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size= 0.33)

xt_count = []
for x in range (X_train1[:,0].size):
    xt_count.append(x)

pca=PCA(n_components=1)
x=pca.fit_transform(X_train1, y_train1)

plt.scatter(xt_count,X_train1[:,0])
plt.scatter(xt_count,X_train1[:,1])
plt.scatter(xt_count,X_train1[:,2])
plt.scatter(xt_count,X_train1[:,3])
plt.scatter(xt_count,x, marker='*')


X_train1, X_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size= 0.33)

clf=LDA(n_components=1)
x=clf.fit_transform(X_train1, y_train1)

plt.scatter(xt_count,X_train1[:,0])
plt.scatter(xt_count,X_train1[:,1])
plt.scatter(xt_count,X_train1[:,2])
plt.scatter(xt_count,X_train1[:,3])
plt.scatter(xt_count,x, marker='*')


print(accuracy_score(y_test1, clf.predict(X_test1))*100)
lr_count = []
for x in range (X_test1[:,0].size):
    lr_count.append(x)

# lr_count
plt.scatter(lr_count,y_test1)
plt.scatter(lr_count,clf.predict(X_test1))
# # Splitting the Dataset 
X_train1, X_test1, y_train1, y_test1 = train_test_split(x2[:,1:2], x2[:,0], test_size= 0.33)

# # Instantiating LinearRegression() Model
lr = LinearRegression()

# # Training/Fitting the Model
lr.fit(X_train1, y_train1)

# # Making Predictions
lr.predict(X_test1)
lr_pred = lr.predict(X_test1)
# print(accuracy_score(y_test1, lr_pred)*100)
X_train1
plt.scatter(lr_count, y_test1)
#pred
plt.scatter(lr_count,lr_pred)
# X_train, X_test, y_train, y_test=train_test_split(x2, y2, test_size=0.3, random_state=42)
lr1_count = []
for x in range (X_test1[:,0].size):
    lr1_count.append(x)
# lr1_count
X_train1, X_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size= 0.33)
lr1=LogisticRegression()

lr1.fit(X_train1, y_train1)

lr1.predict(X_test1)
lr1_pred=lr1.predict(X_test1)

# print(pred)
plt.scatter(lr1_count,y_test1)
#pred
plt.scatter(lr1_count,lr1_pred)
print(accuracy_score(y_test1, lr1_pred)*100)
X_train.shape
tst_count = []
for x in range (X_test[:,0].size):
    tst_count.append(x)
# tst_count

tr_count = []
for z in range (X_train[:,0].size):
    tr_count.append(z)
# tr_count

svm_model=SVC(kernel='linear', max_iter=-1)
svm_model.fit(X_train, y_train)
svm_pred=svm_model.predict(X_test)
print(accuracy_score(y_test, svm_pred)*100)
plt.scatter(tst_count, svm_pred)

ax=plt.gca()
xlim=ax.get_xlim()

ax.scatter(tst_count, X_test[:,0])

w=svm_model.coef_[0]
a=-w[0]/w[1]
xx=np.linspace(xlim[0], xlim[1])
yy=a*xx+(svm_model.intercept_[0]/w[1])
plt.plot(xx, yy, 'red')
a
dtc_model=DecisionTreeClassifier(criterion='gini',max_leaf_nodes=3) #max_leaf_nodes = None , then we have high accuracy
dtc_model.fit(X_train,y_train)
dtc_pred=dtc_model.predict(X_test)

# print(pred)

plt.scatter(tst_count,y_test)
#pred
plt.scatter(tst_count,dtc_pred)
print(accuracy_score(y_test, dtc_pred)*100)
rfc_model=RandomForestClassifier(n_estimators=100, criterion='gini')

rfc_model.fit(X_train,y_train)
rfc_pred=rfc_model.predict(X_test)
plt.scatter(tst_count,y_test)
plt.scatter(tst_count,rfc_pred)
print(accuracy_score(y_test, rfc_pred)*100)

cluster=KMeans(n_clusters =3,random_state=1, max_iter=10, algorithm='auto')
cluster.fit(X_train)
labels=cluster.labels_
labels.size
print(labels)

# labels.tostring()
# for i in range (134):

#     if labels[i] == '1':
#         print(i)
#         labels[i]='Iris-setosa'
#     if labels[i] == '0':
#         labels[i]='Iris-versicolor'
#     if labels[i] == '2':
#         labels[i]='Iris-virginica'
# labels
# # print(accuracy_score(y_test, labels)*100)
print(accuracy_score(y_test1, lr1_pred)*100, accuracy_score(y_test, knn_pred)*100, accuracy_score(y_test, naive_pred)*100,
      accuracy_score(y_test, svm_pred)*100, accuracy_score(y_test, dtc_pred)*100, accuracy_score(y_test, rfc_pred)*100)
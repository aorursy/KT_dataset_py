# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#reading dataset
#Q.2 II.	Feature Extraction (for dataset B)
#problem 1
#applying pca to reduce dimension
dfB=pd.read_csv("/kaggle/input/handwritten-digits-recognition/DataB.csv")
X_dfb = dfB.iloc[:, 1:785].values


from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_dfb)
X_pca=pca.transform(X_dfb) 

print("eigen vectors:" ,pca.components_)
print("eigen values: " ,pca.explained_variance_)

#plot 1st and 2nd pca components
import matplotlib.pyplot as plt
Xax=X_pca[:,0]
Yax=X_pca[:,1]
labels=dfB["gnd"]
cdict={0:'red',1:'green',2:'black',3:'blue',4:'yellow'}
labl={0:0,1:1,2:2,3:3,4:4}
# class 0,1,2,3,4 has each different class
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l])
# for loop ends
plt.xlabel("First Principal Component",fontsize=14)
plt.ylabel("Second Principal Component",fontsize=14)
plt.legend()
plt.show()
#plot 5th and 6th pca components
Xax=X_pca[:,4]
Yax=X_pca[:,5]
labels=dfB["gnd"]
cdict={0:'red',1:'green',2:'black',3:'blue',4:'yellow'}
labl={0:0,1:1,2:2,3:3,4:4}
# class 0,1,2,3,4 has each different class
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l])
# for loop ends
plt.xlabel("5th Principal Component",fontsize=14)
plt.ylabel("6th Principal Component",fontsize=14)
plt.legend()
plt.show()
#problem 4 #
#taking components into dataframe
#lets use naive bayes classifier
#Import Gaussian Naive Bayes model
#  for 1st 2 components
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#features=df_pca.iloc[:,1:2]
features=X_pca[:,0:1]
label=dfB["gnd"]
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
gnb = GaussianNB()
y_pred2 = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred2).sum()))

#plot classification error for first 2 components
Xax=X_test.flatten()
Yax=y_pred2
labels=y_test
cdict={0:'red',1:'green',2:'black',3:'blue',4:'yellow'}
labl={0:0,1:1,2:2,3:3,4:4}
# class 0,1,2,3,4 has each different class
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l])
# for loop ends
plt.xlabel("Original data",fontsize=14)
plt.ylabel("predicted data",fontsize=14)
plt.legend()
plt.show()

#taking first 4  pcacomponents
features=X_pca[:,0:3]
label=dfB["gnd"]
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
gnb = GaussianNB()
y_pred4 = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred4).sum()))

#plotting claassification error for first 4 pca components classification error of naive bayes

Xax=X_test.flatten()
Yax=y_pred4
labels=y_test
cdict={0:'red',1:'green',2:'black',3:'blue',4:'yellow'}
labl={0:0,1:1,2:2,3:3,4:4}
# class 0,1,2,3,4 has each different class
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l])
# for loop ends
plt.xlabel("Original data",fontsize=14)
plt.ylabel("predicted data",fontsize=14)
plt.legend()
plt.show()

#taking first 10  pcacomponents
features=X_pca[:,0:9]
label=dfB["gnd"]
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
gnb = GaussianNB()
y_pred10 = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred10).sum()))

#keep repeating for 30,60,200,500,784 then plot error

#plotting claassification error for first 10 pca components classification error of naive bayes

Xax=X_test.flatten()
Yax=y_pred4
labels=y_test
cdict={0:'red',1:'green',2:'black',3:'blue',4:'yellow'}
labl={0:0,1:1,2:2,3:3,4:4}
# class 0,1,2,3,4 has each different class
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l])
# for loop ends
plt.xlabel("Original data",fontsize=14)
plt.ylabel("predicted data",fontsize=14)
plt.legend()
plt.show()

#taking first 30  pcacomponents
features=X_pca[:,0:29]
label=dfB["gnd"]
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
gnb = GaussianNB()
y_pred30 = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred30).sum()))

#keep repeating for 30,60,200,500,784 then plot error

#plotting claassification error for first 30 pca components classification error of naive bayes

Xax=X_test.flatten()
Yax=y_pred30
labels=y_test
cdict={0:'red',1:'green',2:'black',3:'blue',4:'yellow'}
labl={0:0,1:1,2:2,3:3,4:4}
# class 0,1,2,3,4 has each different class
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l])
# for loop ends
plt.xlabel("Original data",fontsize=14)
plt.ylabel("predicted data",fontsize=14)
plt.legend()
plt.show()

#taking first 60  pcacomponents
features=X_pca[:,0:59]  #change here for next components 0:199,0:499,0:783
label=dfB["gnd"]
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=42)
gnb = GaussianNB()
y_pred60 = gnb.fit(X_train, y_train).predict(X_test)
print("Number of mislabeled points out of a total %d points : %d" % (X_test.shape[0], (y_test != y_pred60).sum()))

#keep repeating for 30,60,200,500,784 then plot error

#plotting claassification error for first 30 pca components classification error of naive bayes

Xax=X_test.flatten()
Yax=y_pred60
labels=y_test
cdict={0:'red',1:'green',2:'black',3:'blue',4:'yellow'}
labl={0:0,1:1,2:2,3:3,4:4}
# class 0,1,2,3,4 has each different class
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l])
# for loop ends
plt.xlabel("Original data",fontsize=14)
plt.ylabel("predicted data",fontsize=14)
plt.legend()
plt.show()
# keep repeating like this
#problem 5
#using LDA to  reduce dimensionality
#using scikit learn lib
X = dfB.iloc[:, 1:785].values
y = dfB.iloc[:, 785].values
#need feature scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X)
X_data=sc.transform(X)
#perform LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X,y)
x_new = lda.transform(X)
print(x_new)
#Verifying that the lda.scalings_ are the eigenvectors

print(lda.scalings_)
print(lda.transform(np.identity(1)))
#plotting 1st and 2nd component of LDA
Xax=x_new[:,0]
Yax=x_new[:,1]
labels=dfB["gnd"]
cdict={0:'red',1:'green',2:'black',3:'blue',4:'yellow'}
labl={0:0,1:1,2:2,3:3,4:4}
# class 0,1,2,3,4 has each different class
fig,ax=plt.subplots(figsize=(7,5))
fig.patch.set_facecolor('white')
for l in np.unique(labels):
 ix=np.where(labels==l)
 ax.scatter(Xax[ix],Yax[ix],c=cdict[l],s=40,
           label=labl[l])
# for loop ends
plt.xlabel("First LDA Componet",fontsize=14)
plt.ylabel("Second LDA Component",fontsize=14)
plt.legend()
plt.show()

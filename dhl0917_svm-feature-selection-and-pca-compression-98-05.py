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
data=pd.read_csv('../input/data.csv')

data.head()
X_ori=data

# y is the labels

y_ori=X_ori.diagnosis

# Drop the unrelated colomns

# X is the instances

dropping_col = ['id','diagnosis','Unnamed: 32']

X_ori=X_ori.drop(dropping_col,axis=1)

print(X_ori.shape)

X_ori.head()
# Normalize data

# Normalization (z-score)

X=(X_ori - X_ori.mean()) / (X_ori.std())

y=[1 if label=='M' else 0 for label in y_ori]

#data_nm = (data_ins-data_ins.min())/(data_ins.max()-data_ins.min())

X.head()
# DataFrame to np

X=X.values

y=np.array(y)

attributes=['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst', 'concave points_worst',

       'symmetry_worst', 'fractal_dimension_worst']
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split

import time

# SVM_linear_kernel

def SVM(X_train,y_train,X_test,y_test,C=0.1):       # X,y are numpy

    clf = SVC(kernel='linear',C=C)

    clf.fit(X_train, y_train)  

    # train accuracy

    pred_train=clf.predict(X_train)

    res_train=[pred_train==y_train]

    acc_train=np.mean(np.array(res_train).astype(np.int))

    # test accuracy

    pred_test=clf.predict(X_test)

    res_test= [pred_test==y_test]

    acc_test=np.mean(np.array(res_test).astype(np.int))

    return acc_train,acc_test



def SVM_iter(X,y,k=100,C=0.1):

    SVM_train_acc=[]

    SVM_test_acc=[]

    for _ in range(k):

        # SVM

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        acc_train,acc_test=SVM(X_train,y_train, X_test,y_test,C)

        SVM_train_acc.append(acc_train)

        SVM_test_acc.append(acc_test)

    return np.array(SVM_train_acc).mean(),np.array(SVM_test_acc).mean()
# SVM

# start=time.time()

# train_acc=[]

# test_acc=[]

# for i in range(100):

#     X_train, X_test, y_train, y_test = train_test_split(X_ori, y_ori, test_size=0.3)

#     acc_train,acc_test=SVM(X_train,y_train, X_test,y_test)

#     train_acc.append(acc_train)

#     test_acc.append(acc_test)

# print(np.array(train_acc).mean())

# print(np.array(test_acc).mean())

# print('Original data operation time: %f'%(time.time()-start))



print()

start=time.time()

train_acc=[]

test_acc=[]

for i in range(100):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    acc_train,acc_test=SVM(X_train,y_train, X_test,y_test)

    train_acc.append(acc_train)

    test_acc.append(acc_test)

print(np.array(train_acc).mean())

print(np.array(test_acc).mean())

print('Normalized data operation time: %f'%(time.time()-start))
from sklearn.feature_selection import RFE

selector_SVM = RFE(SVC(kernel='linear',C=0.1),1, step=1)

selector_SVM = selector_SVM.fit(X, y)

print(selector_SVM.ranking_)

print([attributes[i] for i in np.argsort(selector_SVM.ranking_)])
def SVM_Select(X,y):

    train_acc=[]

    test_acc=[]

    for k in range(X.shape[1]):

        selector_SVM = RFE(SVC(kernel='linear',C=0.1),k+1, step=1)

        X_trans = selector_SVM.fit(X, y).transform(X)

        acc_train,acc_test=SVM_iter(X_trans,y)

        train_acc.append(acc_train)

        test_acc.append(acc_test)

    return train_acc,test_acc
import matplotlib.pyplot as plt

SVM_train_acc,SVM_test_acc=SVM_Select(X,y)

plt.plot(SVM_test_acc)

plt.ylim([0.8,1])
print(SVM_test_acc)

plt.plot([i+1 for i in range(30)],SVM_train_acc,c='red',label='training accuracy')

plt.plot([i+1 for i in range(30)],SVM_test_acc,c='green',label='testing accuracy')

plt.scatter([i+1 for i in range(30)],SVM_train_acc,c='red',s=10)

plt.scatter([i+1 for i in range(30)],SVM_test_acc,c='green',s=10)

plt.title('Training & Testing Accuracy for SVM')

plt.xlabel('features')

plt.ylabel('accuracy')

plt.ylim([0.8,1])

plt.legend(loc='best')

plt.show()
from sklearn.metrics import confusion_matrix

# SVM

def SVM_CM(X,y):       # X,y are numpy

    matrix=np.array([[0,0],[0,0]])

    for i in range(100):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        clf = SVC(kernel='linear',C=0.1)

        clf.fit(X_train, y_train)  

        pred_test=clf.predict(X_test)

        matrix+=confusion_matrix(y_test, pred_test)

    

    tn, fp, fn, tp=matrix.ravel()

    print(tn, fp, fn, tp)

    accuracy=(tn+tp)/(tn+fp+fn+tp)

    precision=(tp)/(tp+fp)

    recall=tp/(tp+fn)

    specificity=tn/(tn+fp)

    f1=(2*precision*recall)/(precision+recall)

    matrix=(matrix/100+0.5).astype(int)

    return matrix,np.array([[accuracy,precision,recall,specificity,f1]])
import seaborn as sns

# SVM selection -> 14 features

selector_SVM = RFE(SVC(kernel='linear',C=0.1),14, step=1)

X_SVM = selector_SVM.fit(X, y).transform(X)

res,metrics_SVM_trans=SVM_CM(X_SVM,y)

sns.heatmap(res,annot=True,fmt="d")

plt.show()

print(metrics_SVM_trans)
from sklearn.decomposition import PCA

# feature selection -> PCA

pca = PCA(n_components=0.99)

pca.fit(X_SVM)

X_pca_SVM=pca.transform(X_SVM)

print(X_SVM.shape)

print(X_pca_SVM.shape)
rain_acc=[]

test_acc=[]

for i in range(100):

    X_train, X_test, y_train, y_test = train_test_split(X_pca_SVM, y, test_size=0.3)

    acc_train,acc_test=SVM(X_train,y_train, X_test,y_test)

    train_acc.append(acc_train)

    test_acc.append(acc_test)

print(np.array(train_acc).mean())

print(np.array(test_acc).mean())
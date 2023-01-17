import pandas as pd # verinin organizasyonu için

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn import metrics

from sklearn.model_selection import cross_val_predict

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_validate

from sklearn import svm

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.cluster import KMeans



#Grafik çizdirme kütüphanesi

import matplotlib.pyplot as plt



import os #Sistem 

import warnings #uyarılar

#print(os.listdir("../input/"))

warnings.filterwarnings("ignore")
from sklearn import datasets

iris =datasets.load_iris()
X=iris.data

Y=iris.target
plt.scatter(X[:,0], X[:,1], c=Y, cmap='gist_rainbow')

plt.xlabel('Spea1 Length', fontsize=18)

plt.ylabel('Sepal Width', fontsize=18)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.3,random_state=1)

def computing(cm):

    Eba = cm[1,0]

    Eca = cm[2,0]

    Eab = cm[0,1]

    Ecb = cm[2,1]

    Eac = cm[0,2]

    Ebc = cm[1,2]

    TPa = cm[0,0]

    TPb = cm[1,1]

    TPc = cm[2,2]

    FNa = Eba+Eca

    FNb = Eab+Ecb

    FNc = Eac+Ebc

    FPa = Eab+Eac

    FPb = Eba+Ebc

    FPc = Eca+Ecb

    TNa = TPb+Ebc+Ecb+TPc

    TNb = TPa+Eac+Eca+TPc

    TNc = TPa+Eab+Eba+TPb

    Total = TPa+Eab+Eac+Eba+TPb+Ebc+Eca+Ecb+TPc

    accuracy = (TPa+TPb+TPc)/Total

    sensitivityA = (TPa)/(TPa+FNa)

    sensitivityB = (TPb)/(TPb+FNb)

    sensitivityC = (TPc)/(TPc+FNc)

    specificityA = (TNa)/(TNa+FPa)

    specificityB = (TNb)/(TNb+FPb)

    specificityC = (TNc)/(TNc+FPc)

    print("accuracy: ",accuracy)

    print("sensitivityA: ",sensitivityA)

    print("sensitivityB: ",sensitivityB)

    print("sensitivityC: ",sensitivityC)

    print("specificityA: ",specificityA)

    print("specificityB: ",specificityB)

    print("specificityC: ",specificityC)

    

    matrisim=[["accuracy: ",accuracy],["sensitivityA: ",sensitivityA],

          ["sensitivityB: ",sensitivityB],["sensitivityC: ",sensitivityC],

          ["specificityA: ",specificityA],["specificityB: ",specificityB],

          ["specificityC: ",specificityC]

          ]

    return matrisim



def classifier(model,name,n1,n2,n3):

    print("-------------")

    print("model name: ",str(name))

    print("-------------")

    fig=plt.gcf()

    fig.set_size_inches(10,5)

    plt.subplot(n1,n2,n3)

    plt.title('train')

    model.fit(x_train,y_train)

    y_pred0=cross_val_predict(model,x_train,y_train,cv=10)

    cm=confusion_matrix(y_train,y_pred0)

    sns.heatmap(cm,annot=True,fmt="d")

    print("accuracy_score")

    print(metrics.accuracy_score(y_train, y_pred0))

    print("sensitivity")

    print(metrics.recall_score(y_train, y_pred0, average='macro'))

    print("precision")

    print(metrics.precision_score(y_train, y_pred0, average='macro'))



    

    plt.subplot(n1,n2,n3+1)

    plt.title('test')

    model.fit(x_test,y_test)

    y_pred00=cross_val_predict(model,x_test,y_test,cv=10)

    cm2=confusion_matrix(y_test,y_pred00)

    sns.heatmap(cm2,annot=True,fmt="d")



    plt.subplot(n1,n2,n3+2)

    plt.title('validation all')

    model.fit(X,Y)

    y_pred2=cross_val_predict(model,X,Y,cv=10)

    conf_mat2=confusion_matrix(Y,y_pred2)

    sns.heatmap(conf_mat2,annot=True,fmt="d")

    # plt.show()



#    a='iris'+str(name)+'.png'

#    fig.savefig(a,dpi=100)





    cv1 = cross_validate(model, x_train, y_train, cv=10)

    cv2 = cross_validate(model, x_test, y_test, cv=10)

    cv3 = cross_validate(model, X, Y, cv=10)



    print('train '+str(name)+'accuracy is: ',cv1['test_score'].mean())

    print('test '+str(name)+' accuracy is: ',cv2['test_score'].mean())

    print('validation all'+str(name)+'accuracy is: ',cv3['test_score'].mean())

    print('')

    

    matris1 = computing(cm)

    matris2 = computing(cm2)

    matris3 = computing(conf_mat2)

    

    return matris1,matris2,matris3
knn=KNeighborsClassifier(n_neighbors=8)

kmeans = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)
matris1,matris2,matris3=classifier(kmeans,"kmeans",4,3,1)

matris11,matris22,matris33=classifier(knn,"knn",4,3,2)
centers = kmeans.cluster_centers_

print(centers)
new_labels = kmeans.labels_

fig, axes = plt.subplots(1, 2, figsize=(16,8))

axes[0].scatter(X[:, 0], X[:, 1], c=Y, cmap='gist_rainbow',

edgecolor='k', s=150)

axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='jet',

edgecolor='k', s=150)

axes[0].set_xlabel('Sepal length', fontsize=18)

axes[0].set_ylabel('Sepal width', fontsize=18)

axes[1].set_xlabel('Sepal length', fontsize=18)

axes[1].set_ylabel('Sepal width', fontsize=18)

axes[0].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)

axes[1].tick_params(direction='in', length=10, width=5, colors='k', labelsize=20)

axes[0].set_title('Actual', fontsize=18)

axes[1].set_title('Predicted', fontsize=18)
plt.scatter(X[:,0], X[:,1]);
plt.scatter(X[:,0], X[:,1], c=kmeans.labels_);
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_);

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=100, color="red"); # Show the centres
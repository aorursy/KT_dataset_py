

import time

t0 = time.time()



import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

from subprocess import check_output



dataset = pd.read_csv("../input/train.csv")



features=dataset.drop(['rn', 'activity'], axis = 1)

labels=dataset['activity']







Labels_keys = labels.unique().tolist()

Labels = np.array(Labels_keys)



print(Labels)

dict = {}

for i in range(len(Labels)):

    dict[Labels[i]]=i



print(dataset.isnull().sum())



features=(features-features.min())/(features.max()-features.min())
x = np.array(features).astype(float)

y = np.array([dict[i] for i in labels]).astype(int)

n_cluster = len(Labels)





m=len(x[0])

n=len(x)

def split(x,y,test_size):

    tot=len(x)

    t1=int(test_size*tot)

    arr = np.arange(tot)

    np.random.shuffle(arr)

    idtrain=arr[t1:]

    idtest=arr[:t1]

    print(len(idtrain),len(idtest),tot)

    print(idtrain,idtest)

    return x[idtrain],x[idtest],y[idtrain],y[idtest]



#Splitting the data into Train and Test

xtrain, xtest, ytrain, ytest = split(x,y,1/3)





def l2_norm(a, b):

    s=0

    for p in range(len(a)):

            s= s + (a[p] - b[p])**2

    return s



def kmeans(x,n_cluster):

    

    m=len(x[0])

    n=len(x)

    pred=np.zeros((n), dtype=int)



    kcase=0

    while(True):

        kcase+=1

        print("kcase =",kcase)

        if(kcase>1):

            centre = np.zeros((n_cluster,m) )

            n_points= np.zeros(n_cluster)

            for i in range(n):

                n_points[pred[i]]+=1

                centre[pred[i]]+=x[i]

            for i in range(n_cluster):

                if(n_points[i]>0):

                    centre[i]/=n_points[i]



                    flag=False



            for i in range(n):

                d=[np.linalg.norm(x[i]-centre[j]) for j in range(n_cluster)]

                minj=np.argmin(d)

                if(pred[i]!=minj):

                    pred[i]=minj

                    flag=True

            if(flag==False):

                break

        else:

            centre = np.zeros((n_cluster,m) )



            for j in range(m):

                minj = min(x[:, j])

                maxj = max(x[:, j])

                rangej = float(maxj-minj)

                centre[: ,j]=minj+rangej*np.random.rand(n_cluster) 

        

            for i in range(n):

                d=[np.linalg.norm(x[i]-centre[j]) for j in range(n_cluster)]

                minj=np.argmin(d)

                pred[i]=minj

            

    return pred



colors=['b','g','c','r','m','y','k','w']



pred=kmeans(x,n_cluster)

for i in range(n_cluster):

    plt.scatter(x[np.where(pred==i)][:,0], x[np.where(pred==i)][:,1], c=colors[i]) 

hmap=np.zeros((n_cluster,n_cluster))

for i in range(n): hmap[y[i]][pred[i]]+=1



print(hmap)

        
sns.heatmap(pd.DataFrame(hmap), annot=True, cmap="YlGnBu" ,fmt='g')

plt.xlabel('Predicted label')

plt.ylabel('True label')


t1 = time.time()

print(t1-t0, "seconds wall time")
def cov_mat(x):

    m = x.shape[0]

    x = x - np.mean(x, axis=0)

    return 1 / m * np.matmul(x.T, x)



def PCA(x, n_components):

    cov_matrix = cov_mat(x)

    eigval, eigvec = np.linalg.eig(cov_matrix)

    

    idx = eigval.argsort()[::-1]

    eigvec = eigvec[:, idx]

    eigvec = eigvec[:, :n_components]

    

    ans = np.matmul(x, eigvec)

    return ans

x_pca = PCA(x, 10)

x_pca=x_pca.real
x_pca


x1 = x_pca[:, 0]

x2 = x_pca[:, 1]
pred_pca=kmeans(x_pca,n_cluster)
for i in range(n_cluster):

    plt.scatter(x1[np.where(pred_pca==i)], x2[np.where(pred_pca==i)], c=colors[i]) 
hmap=np.zeros((n_cluster,n_cluster))

for i in range(n):

    hmap[y[i]][pred_pca[i]]+=1



print(hmap)
sns.heatmap(pd.DataFrame(hmap), annot=True, cmap="YlGnBu" ,fmt='g')

plt.xlabel('Predicted label')

plt.ylabel('True label')
choose_id=np.argmax(hmap,axis=1)
acc=np.zeros(n_cluster)

for _k in range(n_cluster):

    TN = TP = FN = FP = 0

    for i in range(n_cluster): #predict

        for j in range(n_cluster): #labels

            val=hmap[j][i]

            if (i==_k and j==_k): TN=TN+val

            if (i==_k and j!=_k): FN=FN+val

            if (i!=_k and j==_k): FP=FP+val

            if (i!=_k and j!=_k): TP=TP+val

    acc[_k] = (TP+TN)/(TP+TN+FP+FN)
acc
plt.plot(range(n_cluster),acc,'s-',color = 'r')

plt.xlabel("k")

plt.ylabel("accuracy")

plt.legend(loc = "best")

plt.show()
t1 = time.time()

print(t1-t0, "seconds wall time")
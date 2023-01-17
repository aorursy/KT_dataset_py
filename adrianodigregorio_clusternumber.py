# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
Mall_Customers=pd.read_csv('../input/mall-customers/Mall_Customers.csv')

Iris = pd.read_csv("../input/iris/Iris.csv")

Seed_Data = pd.read_csv("../input/seed-from-uci/Seed_Data.csv")

X_C1=[]

with open('/kaggle/input/uefclusters2/s1.txt') as f:

    for line in f:

        x, y = line.split()

        X_C1.append([int(x),int(y)])

X_C2=[]

with open('/kaggle/input/uefclusters2/s2.txt') as f:

    for line in f:

        x, y = line.split()

        X_C2.append([int(x),int(y)])

X_C3=[]

with open('/kaggle/input/uefclusters2/s3.txt') as f:

    for line in f:

        x, y = line.split()

        X_C3.append([int(x),int(y)])

X_C4=[]

with open('/kaggle/input/uefclusters2/s4.txt') as f:

    for line in f:

        x, y = line.split()

        X_C4.append([int(x),int(y)])



X_M=Mall_Customers.values[:,(3,4)]

X_I=Iris.values[:,(1,2,3,4)]

X_S=Seed_Data.values[:,range(0,7)]



from sklearn.cluster import KMeans

from sklearn.metrics import silhouette_score

def metricGenerator(X):

    sil=[0,0]

    iner=[]

    N=30

    for k in range(1, N):

            km = KMeans(init="k-means++", n_clusters=k, n_init=20)

            km.fit(X)

            iner.append(km.inertia_)

            if (k>1): 

                sil.append(silhouette_score(X,km.labels_))

    n=iner[0]

    for k in range(0,N-1):

        iner[k] = iner[k]/n

    return ([iner,sil])
mall=metricGenerator(X_M)

import pylab as pl

pl.plot(mall[0])

pl.xlabel("numero di cluster")

pl.ylabel("inerzia")

pl.grid()

pl.show()



pl.plot(mall[1])

pl.grid()

pl.xlabel("numero di cluster")

pl.ylabel("silhouette score")

pl.show()
def dDer(lis):

    ret=[]

    for k in range (0,len(lis)-1):

        ret.append(lis[k]-lis[k+1])

    return(ret)
pl.plot(dDer(dDer(mall[0])))

pl.xlabel("numero di cluster")

pl.ylabel("derivata seconda")

pl.grid()

pl.show()
def findElbow(lis,slg):

        dd=dDer(dDer(lis))

        for k in range(0,len(dd)-1):

            if (dd[k]**2+dd[k+1]**2+dd[k+2]**2)/3<slg: 

                return(k)

        return(len(dd)-1)

            
print("silhouette:",mall[1].index(max(mall[1]))," gomito:",findElbow(mall[0],10**(-4))+1)
pl.scatter(X_M[:,0],X_M[:,1],c=KMeans(init="k-means++", n_clusters=5, n_init=20).fit(X_M).labels_)

pl.xlabel("annual income")

pl.ylabel("spending score")

pl.show()
S1=metricGenerator(X_C1)
pl.plot(S1[0])

pl.xlabel("numero di cluster")

pl.ylabel("inerzia")

pl.grid()

pl.show()

pl.plot(dDer(dDer(S1[0])))

pl.xlabel("numero di cluster")

pl.ylabel("derivata seconda dell'inerzia")

pl.grid()

pl.show()

print(" gomito:",findElbow(S1[0],10**(-4))+1)
def findElbowCandidates(lis,slg,slg2):

    dd=dDer(dDer(lis))

    el=[]

    f=slg2

    for k in range(0,len(dd)-2):

        if f<slg2:

            f=(dd[k]**2+dd[k+1]**2+dd[k+2]**2)/3

        else:

            if (dd[k]**2+dd[k+1]**2+dd[k+2]**2)/3<slg: 

                el.append([k,(dd[k]**2+dd[k+1]**2+dd[k+2]**2)/3,f])

            f=(dd[k]**2+dd[k+1]**2+dd[k+2]**2)/3

    return(el)

    
def findBestElbow(lis,slg,slg2):

    el=findElbowCandidates(lis,slg,slg2)

    el_v=[]

    for k in range(0,len(el)):

        el_v.append(el[k][2]/el[k][1])

    return(el[el_v.index(max(el_v))][0])

print("silhouette:",S1[1].index(max(S1[1]))," gomito semplice:",findElbow(S1[0],10**(-4))+1,  "gomito migliorato:", findBestElbow(S1[0],10**(-3),10**(-6))+1)
def bestClusterNumber(lis):

    be = findBestElbow(lis[0],10**(-3),10**(-6))+1

    si = lis[1].index(max(lis[1]))

    se = findElbow(lis[0],10**(-4))+1

    return([be,si,se,max(se,si),max(se,si,be)])
bestClusterNumber(S1)
def printBCN(name,target,res):

    print("dataset",name,"formato da",target,"clusters:      gomito migliorato:", res[0]," silhouette:",res[1]," gomito semplice:",res[2], " pool2:",res[3]," pool3:",res[4] )

def findBCN(name,target,data):

    printBCN(name,target,bestClusterNumber(metricGenerator(data)))
findBCN("Mall_Customers",5, X_M)

findBCN("Iris Species",3, X_I)

findBCN("Seed_from_UCI",3, X_S)

findBCN("S1",15, X_C1)

findBCN("S2",15, X_C2)

findBCN("S3",15, X_C3)

findBCN("S4",15, X_C4)
from sklearn.datasets import make_blobs

X, Y= make_blobs(n_samples=500, centers=4, cluster_std=0.8, random_state=0)

pl.scatter(X[:,0],X[:,1],c=Y)

pl.show()


rand_counter=0

ind=[]

per=[]

for cl_n in range(3,9):

    ind2=[]

    per2=[]

    for std in range (2,7):

        ind1=[]

        per1=[]

        for rand in range(1,5):

            

            X, Y= make_blobs(n_samples=500, centers=cl_n, cluster_std=std/5, random_state=rand_counter)

            Z=bestClusterNumber(metricGenerator(X))

            ind1.append(Z)

            per1.append(list(map(lambda x:x-cl_n, Z)))

            #print(rand,std,cl_n," -> ", Z)

            rand_counter+=1

        ind2.append(ind1)

        per2.append(per1)

    ind.append(ind2)

    per.append(per2)

ind=np.asarray(ind)

per=np.asarray(per)

print(per)
def avErr (lis):

    return(sum(map(abs,lis))/len(lis))

def avMissErr(lis):

    return(sum(map(abs,lis))/sum(map(lambda x:x!=0,lis)))

def avHitR (lis):

    return(sum(map(lambda x:x==0,lis))/len(lis))

def totOv (lis):

    return(sum(map(lambda x:x>0,lis))/len(lis))

def flt(lis):

    return([y for x in lis for y in x])

def flt2(lis):

    return(flt(flt(lis)))

def printscore(name,lis):

    print("Performance",name,": Errore medio:",avErr(lis),"Predizioni corrette:", avHitR(lis),"Errore medio sulle predizoni errate:",avMissErr(lis) , "Stime per ecesso:", totOv(lis) )

def score(lis):

    printscore("gomito migliorato",flt2(lis[:,:,:,0]))

    printscore("silhouette",flt2(lis[:,:,:,1]))

    printscore("gomito semplice",flt2(lis[:,:,:,2]))

    printscore("pooling di 2",flt2(lis[:,:,:,3]))

    printscore("pooling di 3",flt2(lis[:,:,:,4]))
score(per)
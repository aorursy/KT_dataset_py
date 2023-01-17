# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

dlist = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        dlist.append(os.path.join(dirname,filename))



# Any results you write to the current directory are saved as output.
train_pd = pd.read_csv(dlist[1])

train_pd.shape

train_pd.head()
test_pd = pd.read_csv(dlist[0])

test_pd.shape

test_pd.head(3)
train_pd.info()
test_pd.info()
train_pd.mode()
train_pd["Sex"][train_pd["Sex"]=="male"]=0

train_pd["Sex"][train_pd["Sex"]=="female"]=1

train_pd["Embarked"][train_pd["Embarked"]=="C"]=0

train_pd["Embarked"][train_pd["Embarked"]=="Q"]=1

train_pd["Embarked"][train_pd["Embarked"]=="S"]=2

train_pd["Cabin"][train_pd["Cabin"].str.startswith("A",na=False)]=0

train_pd["Cabin"][train_pd["Cabin"].str.startswith("B",na=False)]=1

train_pd["Cabin"][train_pd["Cabin"].str.startswith("C",na=False)]=2

train_pd["Cabin"][train_pd["Cabin"].str.startswith("D",na=False)]=3

train_pd["Cabin"][train_pd["Cabin"].str.startswith("E",na=False)]=4

train_pd["Cabin"][train_pd["Cabin"].str.startswith("F",na=False)]=5

train_pd["Cabin"][train_pd["Cabin"].str.startswith("G",na=False)]=6

train_pd["Cabin"][train_pd["Cabin"].str.startswith("T",na=False)]=7

train_pd.head()
train_pd.mean()
train_pd["Age"]=train_pd["Age"].fillna(24)

train_pd["Embarked"]=train_pd["Embarked"].fillna(2)

train_pd["Cabin"]=train_pd["Cabin"].fillna(8)
train_pd.mode()
train_pd["Sex"].astype(np.int64)

train_pd.info()
print(train_pd.corr())
test_pd["Sex"][test_pd["Sex"]=="male"]=0

test_pd["Sex"][test_pd["Sex"]=="female"]=1

test_pd["Embarked"][test_pd["Embarked"]=="C"]=0

test_pd["Embarked"][test_pd["Embarked"]=="Q"]=1

test_pd["Embarked"][test_pd["Embarked"]=="S"]=2

test_pd["Age"]=test_pd["Age"].fillna(24)

test_pd["Embarked"]=test_pd["Embarked"].fillna(2)

test_pd["Cabin"]=test_pd["Cabin"].fillna(8)

test_pd["Fare"]=test_pd["Fare"].fillna(32)

test_pd["Cabin"][test_pd["Cabin"].str.startswith("A",na=False)]=0

test_pd["Cabin"][test_pd["Cabin"].str.startswith("B",na=False)]=1

test_pd["Cabin"][test_pd["Cabin"].str.startswith("C",na=False)]=2

test_pd["Cabin"][test_pd["Cabin"].str.startswith("D",na=False)]=3

test_pd["Cabin"][test_pd["Cabin"].str.startswith("E",na=False)]=4

test_pd["Cabin"][test_pd["Cabin"].str.startswith("F",na=False)]=5

test_pd["Cabin"][test_pd["Cabin"].str.startswith("G",na=False)]=6

test_pd["Cabin"][test_pd["Cabin"].str.startswith("T",na=False)]=7

test_pd.head()
train_num = train_pd.to_numpy()

train=np.delete(train_num,0,1)

train=np.delete(train,2,1)

train=np.delete(train,6,1)

train_ss=train[:,0:1]

train=np.delete(train,0,1)

#train=np.delete(train,2,1)

#train=np.delete(train,2,1)

#train=np.delete(train,2,1)



print(train[:5,:])

print(train_ss[:5,:])



train_s= np.zeros((891,2))

for cnt in range(891):

    idx = train_ss[cnt,0]

    train_s[cnt,idx] = 1 



print(train_s[:5,:])
test_num = test_pd.to_numpy()

test=np.delete(test_num,0,1)

test=np.delete(test,1,1)

test=np.delete(test,5,1)

#test=np.delete(test,2,1)

#test=np.delete(test,2,1)

#test=np.delete(test,2,1)

#test=np.delete(test,3,1)

print(test[:5,:])
#シグモイド

def Sigmoid(x):

    y=1/(1+np.exp(-x))

    return y



train.shape
#ネットワーク

def FNN(wv,M,K,x):

    N,D=x.shape

    w=wv[:M*(D+1)]

    w=w.reshape(M,(D+1))

    v=wv[M*(D+1):]

    v=v.reshape((K,M+1))

    b=np.zeros((N,M+1))

    z=np.zeros((N,M+1))

    a=np.zeros((N,K))

    y=np.zeros((N,K))

    for n in range(N):

        for m in range(M):

            b[n,m]=np.dot(w[m,:],np.r_[x[n,:],1])

            z[n,m]=Sigmoid(b[n,m])

        z[n,M]=1

        wkz=0

        for k in range(K):

            a[n,k]=np.dot(v[k,:],z[n,:])

            wkz=wkz + np.exp(a[n,k])

        for k in range(K):

            y[n,k]=np.exp(a[n,k])/wkz

    return y,a,z,b



#test

WV=np.ones(57) #M*(D+1)+K*(M+1)

M=5 #D=8

K=2

FNN(WV,M,K,train)
#交差エントロピー誤差

def CE_FNN(wv,M,K,x,t):

    N,D = x.shape

    y,a,z,b=FNN(wv,M,K,x)

    ce=-np.dot(np.log(y.reshape(-1)),t.reshape(-1))/N

    return ce



#test

WV=np.ones(57) #M*(D+1)+K*(M+1)

M=5 #D=8

K=2

CE_FNN(WV,M,K,train,train_s)
import matplotlib.pyplot as plt

%matplotlib inline



#数値微分

def dCE_FNN_num(wv,M,K,x,t):

    epsilon=0.001

    dwv=np.zeros_like(wv)

    for iwv in range(len(wv)):

        wv_modified = wv.copy()

        wv_modified[iwv]=wv[iwv]-epsilon

        mse1=CE_FNN(wv_modified,M,K,x,t)

        wv_modified[iwv]=wv[iwv]+epsilon

        mse2=CE_FNN(wv_modified,M,K,x,t)

        dwv[iwv]=(mse2-mse1)/(2*epsilon)

    return dwv



#dwvの表示

def Show_WV(wv,M):

    N=wv.shape[0]

    plt.bar(range(1,M*9+1),wv[:M*9],align="center",color='black')

    plt.bar(range(M*9+1,N+1),wv[M*9:],align="center",color='cornflowerblue')

    plt.xticks(range(1,N+1))

    plt.xlim(0,N+1)

    

#test

M=5 

K=2

nWV=M*9+K*(M+1)

np.random.seed(1)

WV=np.random.normal(0,1,nWV)

dWV=dCE_FNN_num(WV,M,K,train[:2,:],train_s[:2,:])

print(dWV)

plt.figure(1,figsize=(5,3))

Show_WV(dWV,M)

plt.show()
#数値微分を使った勾配法

import time



def Fit_FNN_num(wv_init,M,K,train,train_s,n,alpha):

    wvt=wv_init

    err_train=np.zeros(n)

    wv_hist=np.zeros((n,len(wv_init)))

    for i in range(n):

        wvt=wvt-alpha*dCE_FNN_num(wvt,M,K,train,train_s)

        err_train[i]=CE_FNN(wvt,M,K,train,train_s)

        wv_hist[i,:]=wvt

    return wvt,wv_hist,err_train



#メイン

startTime=time.time()

M=5

K=2

np.random.seed(1)

WV_init=np.random.normal(0,0.01,M*9+K*(M+1))

N_step=100

alpha=0.5

WV,WV_hist,Err_train=Fit_FNN_num(WV_init,M,K,train,train_s,N_step,alpha)

calculation_time=time.time()-startTime

print("Caluculation time:{0:.3f}sec".format(calculation_time))
#学習誤差の表示

plt.figure(1,figsize=(3,3))

plt.plot(Err_train,'black',label='training')

plt.legend()

plt.show()
#重みの時間発展の表示

plt.figure(1,figsize=(3,3))

plt.plot(WV_hist[:,:M*6],'black')

plt.plot(WV_hist[:,M*6:],'cornflowerblue')

plt.show()
test_ss,A,Z,B=FNN(WV,5,2,test)

test_sss=np.round(test_ss)

test_s=np.zeros((418,1))



for i in range(418):

    if test_sss[i,0]==1:

        test_s[i,0]=0

    else:

        test_s[i,0]=1



a=np.zeros((418,1)) 

for i in range(418):

    a[i,0]=i+892



answer=np.zeros((418,2))

for i in range(418):

    answer[i,0]=a[i,0]

    answer[i,1]=test_s[i,0]



answer
np.round(answer)

answer=pd.DataFrame(answer,columns=['PassengerId','Survived'])

answer=answer.round().astype(int)

answer.head(50)
answer.to_csv("result.csv",index=False,header=True)
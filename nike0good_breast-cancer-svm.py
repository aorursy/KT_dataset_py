import time

t0 = time.time()



import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import random

sns.set()

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df=pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
df.head(5)
df.isnull().sum()
df.describe()
x=df[df.columns.difference(['id', 'diagnosis','Unnamed: 32'])]

#x=df[["radius_mean","texture_mean","smoothness_mean","compactness_mean","concavity_mean"]]

y=df["diagnosis"]
x=(x-x.min())/(x.max()-x.min())
x
x.describe()
x = np.array(x).astype(float)

y = np.array(y)

y =(y=='M')

y=y*1
y=y*2-1

y


def split(x,y,test_size):

    tot=len(x)

    t1=int(test_size*tot)

    arr = np.arange(tot)

    np.random.shuffle(arr)

    idtrain=arr[t1:]

    idtest=arr[:t1]

    return x[idtrain],x[idtest],y[idtrain],y[idtest]



x = np.array(x).astype(float)

y = np.array(y)



#Splitting the data into Train and Test

xtrain, xtest, ytrain, ytest = split(x,y,1/3)
def select_j(i ,m):

    if(m<=1): 

        return NULL

    j = random.randint(0, m-2)

    if(j>=i):

        j+=1

    return j



def clip(x,L,H):

    return max(L,min(x,H))



def simple_smo(dataset, labels, C, max_iter): #0 <= alpha_i <= C

    

    dataset = np.array(dataset)

    m, n = dataset.shape

    

    d1 = np.matrix(dataset)

    K = np.matmul(d1,d1.T)

    K = np.array(K)



    alphas = np.zeros(m)

    b = it = 0

    

    def f(i):

        s1=0

        for j in range(m):

            s1+=K[i][j]*labels[j]*alphas[j]

        return s1+b

    

    def calc_W( x, y,alphas):

        ans=0.

        for i in range(m):

            for j in range(m):

                ans+=y[i]*y[j]*alphas[i]*alphas[j]*K[i][j]

        ans = sum(alphas)-ans/2.0

        return ans

    def delta_W(ai,aj,fi,fj,i,j):

        t1=t2=0

        for k in range(m):

            if(k!=i) and(k!=j):

                t1+=labels[k]*alphas[k]*K[i][k]

                t2+=labels[k]*alphas[k]*K[j][k]

        t0=t1*labels[i]*ai+t2*labels[j]*aj

        p = ai+aj-(ai*ai*K[i][i]+aj*aj*K[j][j])*0.5 -(ai*aj*K[i][j]*labels[i]*labels[j]) - t0

        return p

    while it < max_iter:

        tot=0

        for i in range(m):

            #Select i,j randomly

            j = select_j(i, m)



            fx_i = f(i)

            E_i = fx_i - labels[i]

            

            fx_j = f(j)

            E_j = fx_j - labels[j]

            

            #Update alphas[i] and alphas[j]

            

            a_i_old, a_j_old = alphas[i], alphas[j]

            

            if (labels[i]!=labels[j]):

                L = max(0, alphas[j] - alphas[i])

                H = min(C, C + alphas[j] - alphas[i])

            else:

                L = max(0, alphas[j] + alphas[i] - C)

                H = min(C, alphas[j] + alphas[i])

            

            if(L>=H):

                continue

            

            eta = K[i][i]+K[j][j]-2*K[i][j]

            if (eta<=0.0001):

                continue

            

            a_j_new = a_j_old + labels[j]*(E_i - E_j)/eta

            a_j_new = clip(a_j_new, L, H)

            

            if abs(a_j_new - a_j_old) < 0.01:

                continue



            a_i_new = a_i_old + labels[i]*labels[j]*(a_j_old - a_j_new)

            

            '''

            delta=delta_W(a_i_new,a_j_new,fx_i,fx_j,i,j)-delta_W(a_i_old,a_j_old,fx_i,fx_j,i,j)

            if(delta<0):

                print(delta,"ero")

                continue

            #w1=calc_W(dataset,labels,alphas)

            '''

            alphas[i], alphas[j] = a_i_new, a_j_new

            '''

            w2=calc_W(dataset,labels,alphas)

            if (w2<w1):

                print("e",delta,w2,w1)

            else:

                if(delta!=w2-w1): print(delta-(w2-w1))

            '''

            #Update b

            b_i=b-E_i-labels[i]*(a_i_new-a_i_old)*K[i][i]-labels[j]*(a_j_new-a_j_old)*K[i][j]

            b_j=b-E_j-labels[i]*(a_i_new-a_i_old)*K[i][j]-labels[j]*(a_j_new-a_j_old)*K[j][j]

            if (0<a_i_new<C):

                b = b_i

            elif (0<a_j_new<C):

                b = b_j

            else:

                b = (b_i + b_j)*0.5

            tot+=1

        if tot==0:

            it+=1

        else:

            it=0

        res=calc_W(dataset,labels,alphas)

        print('tot=',tot,'W=',res)

        

    return alphas, b







alphas, b = simple_smo(xtrain,ytrain,0.6, 20)



def caluelate_w(data_mat, label_mat, alphas):



    t1=np.tile(label_mat.reshape(1, -1).T, (1, data_mat.shape[1] ))

    t1=t1*data_mat

         

    w = np.dot(t1.T, alphas)

   

    return w

w = caluelate_w(xtrain, ytrain, alphas)
w
def prediction(test, w, b):

    d1=np.matrix(test)

    yy=np.matmul(d1,np.matrix(w).T)

    yy=yy+b

    yy=(yy>0)*2-1

    return yy



pred = prediction(xtest,w,b)
pred
def work(pred, ytest):



    TN = TP = FN = FP = 0

    for i,j in zip(pred,ytest):

        if(i==-1 and j==-1): TN=TN+1

        if(i==-1 and j==1): FN=FN+1

        if(i==1 and j==-1): FP=FP+1

        if(i==1 and j==1): TP=TP+1



    confusion_matrix=[[TN,FP],[FN,TP]]



    sns.heatmap(pd.DataFrame(confusion_matrix), annot=True, cmap="YlGnBu" ,fmt='g')

    plt.ylabel('True label: diagnosis=M')

    plt.xlabel('Predicted label: diagnosis=M')

    print("accuracy=(TP+TN)/(TP+TN+FP+FN)=" ,(TP+TN)/(TP+TN+FP+FN))

work(pred,ytest)
t1 = time.time()

print(t1-t0, "seconds wall time")
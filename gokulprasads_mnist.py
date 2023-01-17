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
import time 

data = pd.read_csv('../input/digit-recognizer/train.csv')

df=data.values

np.random.shuffle(df)

Y=df[0:37000,0,None].T

X=df[0:37000,1:].T



Ytest=df[37000:,0,None].T

Xtest=df[37000:,1:].T









mean=np.mean(X,1,keepdims=True)

std=np.std(X,1,keepdims=True)

X=X/255

Xtest=Xtest/255



features=X.shape[0]

cachetest={"a0":Xtest}

cachetrain={"a0":X}

mtrain=X.shape[1]

mtest=Xtest.shape[1]
onehot=np.array([0,1,2,3,4,5,6,7,8,9]).reshape(10,1)

y=np.zeros((10,Y.shape[1]))

for i in range(Y.shape[1]):

    y[:,i,None]=(onehot==Y[0,i])

    
def initparam(a,b):

    w=np.random.randn(a,b)

    b=np.zeros((a,1))

    return w,b
def initadam(a,b):

    w=np.zeros((a,b))

    b=np.zeros((a,1))

    return w,b
def sigmoid(z):

    ans=1/(1+np.exp(-z))

    return ans

def lrelu(x):

    ans = np.where(x > 0, x, x * 0.01)  

    return ans



def lreluDer(x):

    ans = np.where(x > 0, 1, 0.01)  

    return ans

def sigder(z):

    return z*(1-z)

def dropout(m,layerdim):

    d={"0":np.random.rand(layerdim[0],m)}

    d["0"]=d["0"]<keep_prob[0]

    for i in range(layers-1):

        d[str(i+1)]=np.random.rand(layerdim[i+1],m)

        d[str(i+1)]=d[str(i+1)]<keep_prob[i+1]

    return d
def trainacc(param):

    ytrain,cach =forward(cachetrain,param)

    pred=np.argmax(ytrain, axis=0).reshape(1,Y.shape[1])

    accuracy=np.sum((pred==Y))*100/mtrain

    return round(accuracy,2)
def testacc(param):

    ytest,cache=forward(cachetest,param)

    pred=np.argmax(ytest, axis=0)

    accuracy=np.sum((pred==Ytest))*100/mtest

    return round(accuracy,2)


    

#FORWARDPROPAGATION

def forwardprop(cache,param,d):

    cache["a0"]*=d["0"]

    cache["a0"]/=keep_prob[0]

    for i in range(layers-1):



        cache["a"+str(i+1)]=lrelu((np.dot(param["w"+str(i+1)],cache["a"+str(i)]))+param["b"+str(i+1)])

        cache["a"+str(i+1)]*=d[str(i+1)]

        cache["a"+str(i+1)]/=keep_prob[i+1]

    cache["a"+str(layers)]=sigmoid((np.dot(param["w"+str(layers)],cache["a"+str(layers-1)]))+param["b"+str(layers)])



    yn=cache["a"+str(layers)]

    

    return yn,cache



def forward(cache,param):

   

    for i in range(layers-1):



        cache["a"+str(i+1)]=lrelu((np.dot(param["w"+str(i+1)],cache["a"+str(i)]))+param["b"+str(i+1)])

        

    cache["a"+str(layers)]=sigmoid((np.dot(param["w"+str(layers)],cache["a"+str(layers-1)]))+param["b"+str(layers)])



    yn=cache["a"+str(layers)]

    

    return yn,cache



def forward_h(cache,param,layers):

   

    for i in range(layers-1):



        cache["a"+str(i+1)]=sigmoid((np.dot(param["w"+str(i+1)],cache["a"+str(i)]))+param["b"+str(i+1)])

        

    cache["a"+str(layers)]=((np.dot(param["w"+str(layers)],cache["a"+str(layers-1)]))+param["b"+str(layers)])



    yn_h=cache["a"+str(layers)]

    

    return yn_h,cache

   

def computeregcost(y,yn,lambd,param,m,t,k,costs,alpha):

    reg=0

    for i in range(layers):

        reg+= np.sum((param["w"+str(i+1)])**2)

        

    cost=-np.sum((y*np.log(yn)+(1-y)*np.log(1-yn )))/m + reg*(lambd/(2*m))



    if t%c==0:

        print(str(round(cost,5))," ",str(int((k/iterations)*100)),"%"," ",str(trainacc(param)),"%"," ",str(testacc(param)),"%",str(alpha))

        costs.append(cost)



def compute(y,yn,lambd):

    reg_h=0

    for i in range(layers):

        reg_h+= np.sum((param_h["w"+str(i+1)])**2)

        

    cost_h= np.sum((y-yn)**2)/(2*m_h) + reg_h*(lambd/(2*m_h))

    print(cost_h)      





    



#BACKPROPAGATION

def backprop(y,yn,cache,param,lambd,m,t,d):

    grad={"dz"+str(layers):yn-y}





    for i in range(layers):

        grad["dw"+str(layers-i)]=np.dot(grad["dz"+str(layers-i)],cache["a"+str(layers-i-1)].T)/m + (lambd/m)*param["w"+str(layers-i)]

        



        grad["db"+str(layers-i)]=np.sum(grad["dz"+str(layers-i)],1,keepdims=True)/m



        if i<layers-1:

            grad["dz"+str(layers-i-1)]=np.dot(param["w"+str(layers-i)].T,grad["dz"+str(layers-i)])*lreluDer(cache["a"+str(layers-i-1)])

            grad["dz"+str(layers-i-1)]*=d[str(layers-i-1)]

            grad["dz"+str(layers-i-1)]/=keep_prob[layers-i-1]

    return grad



def backward(y,yn,cache,param,lambd,layers,m):

    grad_h={"dz"+str(layers):yn-y}





    for i in range(layers):

        grad_h["dw"+str(layers-i)]=np.dot(grad_h["dz"+str(layers-i)],cache["a"+str(layers-i-1)].T)/m + (lambd/m)*param["w"+str(layers-i)]

        



        grad_h["db"+str(layers-i)]=np.sum(grad_h["dz"+str(layers-i)],1,keepdims=True)/m



        if i<layers-1:

            grad_h["dz"+str(layers-i-1)]=np.dot(param["w"+str(layers-i)].T,grad_h["dz"+str(layers-i)])*sigder(cache["a"+str(layers-i-1)])

           

    return grad_h

#UPDATE PARAMETERS  

def updateparam(u,v,alpha,grad,param,t):

    for i in range(layers):

        

        

        u["w"+str(i+1)] = (beta_1 * u["w"+str(i+1)] + (1 - beta_1) * grad["dw"+str(i+1)])/(1 - np.power(beta_1, t)) 

        v["w"+str(i+1)] = (beta_2 * v["w"+str(i+1)] + (1 - beta_2) * np.power(grad["dw"+str(i+1)], 2))/(1 - np.power(beta_2, t))

        u["b"+str(i+1)] = (beta_1 * u["b"+str(i+1)] + (1 - beta_1) * grad["db"+str(i+1)])/(1 - np.power(beta_1, t)) 

        v["b"+str(i+1)] = (beta_2 * v["b"+str(i+1)] + (1 - beta_2) * np.power(grad["db"+str(i+1)], 2))/(1 - np.power(beta_2, t))

       

        

        

        param["w"+str(i+1)]=param["w"+str(i+1)] - alpha * u["w"+str(i+1)] / (np.sqrt(v["w"+str(i+1)]) + epsilon)

        param["b"+str(i+1)]=param["b"+str(i+1)] - alpha * u["b"+str(i+1)] / (np.sqrt(v["b"+str(i+1)]) + epsilon)

        

        #param["w"+str(i+1)]=param["w"+str(i+1)] - alpha * u["w"+str(i+1)] 

        #param["b"+str(i+1)]=param["b"+str(i+1)] - alpha * u["b"+str(i+1)] 

        

        #param["w"+str(i+1)]=param["w"+str(i+1)] - alpha * grad["dw"+str(i+1)]

        #param["b"+str(i+1)]=param["b"+str(i+1)] - alpha * grad["db"+str(i+1)]

    return param,u,v



def update(alpha,grad,param):

    for i in range(layers):    

        param["w"+str(i+1)]=param["w"+str(i+1)] - alpha * grad["dw"+str(i+1)]

        param["b"+str(i+1)]=param["b"+str(i+1)] - alpha * grad["db"+str(i+1)]

    return param
def model(u,v,y,cache,param,lambd,d,m,t,k,costs,alpha):

    yn,cache=forwardprop(cache,param,d)



    computeregcost(y,yn,lambd,param,m,t,k,costs,alpha)



    grad=backprop(y,yn,cache,param,lambd,m,t,d)



    param,u,v=updateparam(u,v,alpha,grad,param,t)
def run(alpha): 

    start_time = time.time() 

    param={}

    for i in range(layers):

        param["b"+str(i+1)]=initparam(layerdim[i+1],layerdim[i])[1]

        param["w"+str(i+1)]=initparam(layerdim[i+1],layerdim[i])[0]*((2/layerdim[i])**0.5)



    

    u={}

    v={}

    for i in range(layers):

        u["b"+str(i+1)]=initadam(layerdim[i+1],layerdim[i])[1]

        u["w"+str(i+1)]=initadam(layerdim[i+1],layerdim[i])[0]

        v["b"+str(i+1)]=initadam(layerdim[i+1],layerdim[i])[1]

        v["w"+str(i+1)]=initadam(layerdim[i+1],layerdim[i])[0]



    

    

    costs=[]

    n=X.shape[1]//batch_size

    lbatch=X.shape[1]- (n*batch_size)

    t=0

    for k in range(iterations):

        for j in range(n):

            cache={"a0":X[:,batch_size*j:batch_size*(j+1)]}

            y_batch=y[:,batch_size*j:batch_size*(j+1)]

            t+=1

            m=cache["a0"].shape[1]

            d=dropout(m,layerdim)

            model(u,v,y_batch,cache,param,lambd,d,m,t,k,costs,alpha)





        if X.shape[1]%batch_size!=0 :

            cache={"a0":X[:,batch_size*(j+1):]} 

            y_batch=y[:,batch_size*(j+1):]

            t+=1

            m=cache["a0"].shape[1]

            d=dropout(m,layerdim)

            model(u,v,y_batch,cache,param,lambd/(batch_size/m),d,m,t,k,costs,alpha)



        if k%25==0 :

            alpha/=(2**1)











    print("train accuracy = "+str(trainacc(param)))





    print("test accuracy = "+str(testacc(param)))



    from matplotlib import pyplot as plt

    axis=(np.linspace(0,t,len(costs)))/(X.shape[1]/batch_size +1)

    plt.plot(axis,costs)



    print("--- %s seconds ---" % (time.time() - start_time))

    

    return testacc(param)
"""batch_size=128

layers=4

layerdim=[features,200,200,200,10]

alpha=0.001

iterations=50

lambd=0.05

c=500

#DROPOUT

keep_prob=[1,0.8,0.7,1]

#ADAM

epsilon= 0.000001

beta_1=0.8

beta_2=0.9



run(alpha)"""
Xh=np.zeros((8,70))

Yh=np.zeros((1,70))
for i in range(70):   

    batch_size=128

    layers=4



    n1=np.random.randint(50,350)

    n2=np.random.randint(50,350)

    n3=np.random.randint(50,350)

    layerdim=[features,n1,n2,n3,10]



    alpha=10**(-np.random.uniform(low=2.0, high=5.0, size=None))

    lambd=10**(-np.random.uniform(low=0.0, high=3.0, size=None))

    iterations=50

    c=500



    d1=np.random.uniform(low=0.5, high=1, size=None)

    d2=np.random.uniform(low=0.5, high=1, size=None)

    d3=np.random.uniform(low=0.5, high=1, size=None)

    keep_prob=[1,d1,d2,d3]



    epsilon= 0.000001

    beta_1=0.8

    beta_2=0.9

    

    

    Xh[:,i]=np.array([n1,n2,n3,alpha,lambd,d1,d2,d3])

    Yh[0,i]=run(alpha)
x1=pd.DataFrame(Xh)

y1=pd.DataFrame(Yh)

x1.to_csv('x1.csv', index=False)

y1.to_csv('y1.csv', index=False)
"""data = pd.read_csv('../input/digit-recognizer/test.csv')

Xt=data.values.T



Xt=(Xt)/255



cache={"a0":Xt}

yn,cache=forward(cache,param)

pred=np.argmax(yn, axis=0)





sub=np.zeros((pred.shape[0],2))

for i in range(pred.shape[0]):

    sub[i,0]=int(i+1)

    sub[i,1]=int(pred[i])



a=pd.DataFrame.from_dict({ "ImageId": sub[:,0],"Label": sub[:,1]}).astype(int)

a.to_csv('mnist.csv', index=False)"""
def model_h(y_h,cache_h,param_h,lambd_h):

    yn_h,cache_h=forward_h(cache_h,param_h,layers_h)



    compute(y_h,yn_h,lambd_h)



    grad_h=backward(y_h,yn,cache_h,param_h,lambd_h,layers_h,m_h)



    param_h=update(alpha_h,grad_h,param_h)
"""Xh=

Yh



m_h=Xh.shape[1]



lambd_h=0

layers_h=2

layerdim_h=[Xh.shape[0],10,1]



param_h={}

for i in range(layers_h):

    param_h["b"+str(i+1)]=initparam(layerdim_h[i+1],layerdim_h[i])[1]

    param_h["w"+str(i+1)]=initparam(layerdim_h[i+1],layerdim_h[i])[0]*((2/layerdim_h[i])**0.5)





cache_h={"a0":Xh}   

model_h(y_h,cache_h,param_h,lambd_h)"""
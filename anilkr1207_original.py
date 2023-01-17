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
import tensorflow as tf

import keras

import random

import numpy as n

import pandas as pd

from scipy.sparse import csr_matrix

from multiprocessing import Pool,Process

import time

import math

from keras import layers
df=pd.read_csv("../input/dummyf/dummy.csv",index_col=0)

ch=n.asarray(df["cards"])

mr=n.asarray(df["merchants"])

tc=n.unique(ch)

tm=n.unique(mr)

t=n.append(tc,tm)

#Targ targets of every node cardholder 1, merchant 0

Targ=[1]*len(tc)+[0]*len(tm)

#dic originalid->index

#rdic index->originalid

dic={x:ind for ind,x in enumerate(t)}

rdic={ind:x for ind,x in enumerate(t)}

data=[1]*len(ch)

data=data*2

row=[]

col=[]

#adl adjacency list representation of graph

#deg degree of every node

#tr csr representation of graph

adl=[[]]*len(t)

deg=[0]*len(t)

edgc={}

for i in range(len(ch)):

    row.append(dic[ch[i]])

    row.append(dic[mr[i]])

    deg[dic[ch[i]]]+=1

    col.append(dic[mr[i]])

    col.append(dic[ch[i]])

    deg[dic[mr[i]]]+=1

    adl[dic[ch[i]]].append(dic[mr[i]])

    adl[dic[mr[i]]].append(dic[ch[i]])

    edgc[(dic[ch[i]],dic[mr[i]])]=0



tr=csr_matrix((data,(row,col)),shape=[len(t),len(t)])
global deg,adl,tr
#call this if you need normalized degree representation i.e D^-1*A

def nrmad(deg,adl):

    ndeg=[0]*len(deg)

    for i in range(len(adl)):

        for j in adl[i]:

            ndeg[i]+=(1/deg[j])

    return ndeg
"""

#get random walk neighbours for every node 

#rwl random walk length

#n if set as 1 all neighbours will be of same category,0.5 would be the normal random walk n 

def trwc(adl,deg,x,rwl):

    ans=n.zeros((len(deg),rwl),dtype="int32")

    x=int(rwl*x)

    for i in range(len(deg)):

        ind=i

        a=[]

        b=[]

        while len(a)+len(b)<=int(2*rwl+2):

            e=random.choice(adl[ind])

            if (len(a)+len(b))%2==0:

                a.append(e)

            else:

                b.append(e)

            ind=e

        c=0

        for j in range(x):

            ans[i][c]=b[j]

            c+=1

        for j in range(rwl-x):

            ans[i][c]=a[j]

            c+=1

    return ans

"""
import multiprocessing 

cores=multiprocessing.cpu_count()

st=0

sz=int(len(deg)/cores)

en=sz

l=[]

while en<len(deg):

    l.append([st,en,0.5,10])

    st=en

    if en+sz>=len(deg):

        l.append([st,len(deg),0.5,10])

    en+=sz

#get random walk neighbours for every node 

#rwl random walk length

#n if set as 1 all neighbours will be of same category,0.5 would be the normal random walk n 

def trwc(k):

    x=k[2]

    rwl=k[3]

    ans=n.zeros((int(k[1]-k[0]),rwl),dtype="int32")

    x=int(rwl*x)

    st=k[0]

    en=k[1]

    for i in range(st,en):

        ind=i

        a=[]

        b=[]

        while len(a)+len(b)<=int(2*rwl+2):

            e=random.choice(adl[ind])

            if (len(a)+len(b))%2==0:

                a.append(e)

            else:

                b.append(e)

            ind=e

        c=0

        for j in range(x):

            ans[i-st][c]=b[j]

            c+=1

        for j in range(rwl-x):

            ans[i-st][c]=a[j]

            c+=1

    return ans

st=time.time()

p=Pool()

res=p.map(trwc,l)

rwneigh=res[0]

for i in range(1,len(res)):

    rwneigh=n.append(rwneigh,res[i],axis=0)

p.close()

p.join()

del res

print(time.time()-st)

#rwneigh for every node the neighbours obtained in the random walk for final loss calculation
#call nrmad if you want to use normalized degrees 
global pedgccumsum,pedgcind
#for edge sampler

for (i,j),k in edgc.items():

    edgc[(i,j)]+=(1/deg[i])

    edgc[(i,j)]+=(1/deg[j])

pedgccumsum=[]

pedgcind=[]

for (i,j),k in edgc.items():

    pedgcind.append([i,j])

    pedgccumsum.append(k)

pedgccumsum=n.cumsum(pedgccumsum)
#pnd number of times a node appeared p(v)

#ped number of times a edge appeared p(u,v)
def nodesampler(k):

    pnd=n.zeros(len(deg))

    cdeg=n.cumsum(deg)

    tv=[]

    for i in range(k):

        x=random.randint(cdeg[0],cdeg[-1])

        ind=n.searchsorted(cdeg,x)

        pnd[ind]+=1

        tv.append(ind)

    A=n.zeros((k,k))

    ped={}

    dic={ind:n for ind,n in enumerate(tv)}

    for i in tv:

        for j in tv:

            arr=tr.indices[indptr[i]:indptr[i+1]]

            ind=n.searchsorted(arr,j)

            if arr[ind]==x:

                A[i][j]=1

                if (i,j) in ped:

                    ped[(i,j)]+=1

                else:

                    ped[(i,j)]=1

    return A,pnd,ped,dic
def edgesampler(k):

    pnd=n.zeros(len(deg))

    tv=[]

    ped={}

    for i in range(k):

        x=random.uniform(pedgccumsum[0],pedgccumsum[-1])

        ind=n.searchsorted(pedgccumsum,x)

        a=pedgcind[ind][0]

        b=pedgcind[ind][1]

        tv.append(a)

        tv.append(b)

        pnd[a]+=1

        pnd[b]+=1

        if (a,b) in ped:

            ped[(a,b)]+=1

        else:

            ped[(a,b)]=1

        if (b,a) in ped:

            ped[(b,a)]+=1

        else:

            ped[(b,a)]=1

    c=len(n.unique(tv))

    tv=n.unique(tv)

    dic={ind:n for ind,n in enumerate(tv)}

    rdic={n:ind for ind,n in enumerate(tv)}

    A=n.zeros((c,c))

    for (i,j),k in ped.items():

        A[rdic[i]][rdic[j]]=1

    return A,pnd,ped,dic
#Sampling Pool

p=Pool()

m=10

l=[10]*m

res1=p.map(edgesampler,l)

p.close()

p.join()
allnp=[]

allep=[]

allA=[]

alldic=[]

for a,b,c,d in res1:

    allnp.append(b)

    allep.append(c)

    allA.append(a)

    alldic.append(d)
P=n.zeros(len(deg))

for i in allnp:

    P+=i

EP={}

for i in allep:

    for (j,k),o in i.items():

        if (j,k) in EP:

            EP[(j,k)]+=o

        else:

            EP[(j,k)]=o
#creating X for all subgraphs

def featinp(args):

    A=args[0]

    dic=args[1]

    s=len(A)

    X=n.zeros((s,len(deg)))

    for i in range(s):

        X[i][dic[i]]=1

    return X

p=Pool()

l=[]

for i in range(len(alldic)):

    l.append([allA[i],alldic[i]])

res2=p.map(featinp,l)

p.close()

p.join()

AllX=[]

for i in res2:

    AllX.append(i)
from keras import layers

import tensorflow as tf

class gcnlayer(layers.Layer):

    def __init__(self,dim,A,dic):

        super(gcnlayer,self).__init__()

        self.l=layers.Dense(dim,activation="tanh")

        self.A=A

        self.units=dim

        self.dic=dic

    def call(self,inputs):

        A=self.A

        dic=self.dic

        inputs=tf.keras.backend.eval(inputs)

        #inputs=n.array(inputs)

        #inputs=tf.make_ndarray(inputs)

        ans=n.zeros((len(A),inputs.shape[1]))

        for i in range(len(A)):

            for j in range(len(A[0])):

                if A[i][j]==1:

                    print(f"{i} {j}")

                    tmp=inputs[j]*P[dic[i]]

                    tmp=tmp/EP[(dic[i],dic[j])]

                    ans[i]+=tmp

        ans=tf.convert_to_tensor(ans,dtype="float32")

        return self.l(ans)

    def get_config(self):

        return {"units":self.units}
from keras import layers

import tensorflow as tf

class gcnlayer(layers.Layer):

    def __init__(self,dim,A,dic):

        super(gcnlayer,self).__init__()

        self.l=layers.Dense(dim,activation="tanh")

        self.A=A

        self.units=dim

        self.dic=dic

    def call(self,inputs):

        A=self.A

        dic=self.dic

        #inputs=tf.keras.backend.eval(inputs)

        #inputs=n.array(inputs)

        #inputs=tf.make_ndarray(inputs)

        inputs=tf.cast(inputs,dtype=tf.float64)

        ans=tf.zeros((len(A),inputs.shape[1]),dtype=tf.float64)

        gtmp=tf.Variable(n.zeros((inputs.shape[1])),dtype=tf.float64)

        for i in range(len(A)):

            for j in range(len(A[0])):

                if A[i][j]==1:

                    #print(f"{i} {j}")

                    tmp=(tf.gather(inputs,j,axis=0))*P[dic[i]]

                    tmp=tmp/EP[(dic[i],dic[j])]

                    gtmp.assign_add(tmp)

        #gtmp=tf.convert_to_tensor(gtmp)

        gtmp=tf.expand_dims(gtmp,axis=0)

        ans=tf.tensor_scatter_nd_update(ans,tf.constant([[i]],dtype=tf.int32),gtmp)

        ans=tf.cast(ans,dtype=tf.float32)

        return self.l(ans)

    def get_config(self):

        config={}

        config["layers"]=[]

        config["layers"].append()

        config["layer"]=-1

        if self.units==5000:

            config["layer"]=1

        if self.units==1000:

            config["layer"]=2

        if self.units==300:

            config["layer"]=3

        return config
class model(keras.Model):

    def __init__(self,A,dic,number):

        super(model,self).__init__()

        self.l1=gcnlayer(5000,A,dic)

        self.l2=gcnlayer(1000,A,dic)

        self.l3=gcnlayer(300,A,dic)

        self.A=A

        self.number=number

    def call(self,inputs):

        y=self.l1(inputs)

        y=self.l2(y)

        return self.l3(y)

    def compute_output_shape(self,input_shape):

        A=self.A

        return (len(A),300)

    def get_config(self):

        #config=super(model,self).get_config()

        config={}

        config["output_dim"]=300

        config["model_number"]=self.number

        return config
def forwardprop(args):

    inputs=args[0]

    mod=args[1]

    return mod(inputs)
tmpm=model(allA[0],alldic[0],0)

l=[]

l.append(tf.convert_to_tensor(allA[0]))

l.append(tmpm)

y=forwardprop(l)
tmpm.get_config()
allm=[]

for i in range(m):

    tmp=model(allA[i],alldic[i],i)

    #tmp.save(f"mod{i}")

    allm.append(tmp)
"""

#Forward Propogation Pool

p=Pool()

l=[]

for i in range(len(allA)):

    l.append([allX[i],allA[i]])

res3=p.map(forwardprop,l)

p.close()

p.join()

"""
finr=tf.zeros((len(deg),300),dtype=tf.float64)
"""

for i in range(len(res3)):

    arr=res3[i]

    dic=alldic[i]

    for j in range(len(arr)):

        finr[dic[j]]+=arr[j]

finr=finr/len(deg)

"""
"""

#Attention Layer

class attenlayer(keras.layers.Layer):

    def __init__(self):

        self.l=layers.Dense(300,activation="tanh")

    def call(self,inputs):

        return self.l(inputs)

att=attenlayer()

atty=att(finr)

for i in range(len(res3)):

    arr=res3[i]

    dic=alldic[i]

    for j in range(len(arr)):

        x=n.dot(atty[dic[j]],arr[j])

        x=x*-1

        x=1/(1+math.exp(x))

        finr[dic[j]]+=x*arr[j]

"""
global finr,rwneigh
def err(args):

    y=args[0]

    dic=args[1]

    ans=0

    for i in range(len(y)):

        ind=dic[i]

        for j in rwneigh[ind]:

            tmp=n.dot(finr[ind],finr[j])

            tmp=1/(1+math.exp(-1*tmp))

            tmp=-1*math.log(tmp)

            ans+=tmp

    return ans
"""

#Error calculation Pool

p=Pool()

l=[]

for i in len(res3):

    l.append([res3[i],alldic[i]])

res4=p.map(err,l)

p.close()

p.join()

"""
class atten(keras.Model):

    def __init__(self):

        super().__init__()

        self.l=layers.Dense(300,activation="tanh")

    def compute_output_shape(self,input_shape):

        return (len(deg),300)

    def call(self,res3):

        #inputs=args[0]

        #res3=args[1]

        atty=self.l(conrep)

        atty=tf.cast(atty,dtype=tf.float64)

        tfinr=tf.zeros((len(deg),300),dtype=tf.float64)

        for i in range(len(res3)):

            arr=res3[i]

            arr=tf.cast(arr,dtype=tf.float64)

            dic=alldic[i]

            for j in range(arr.shape[0]):

                a=tf.gather(atty,dic[j],axis=0)

                b=tf.gather(arr,j,axis=0)

                x=tf.tensordot(a,b,axes=1)

                x=tf.cast(x,dtype=tf.float64)

                x=tf.math.sigmoid(x)

                updates=b*x

                updates=tf.expand_dims(updates,axis=0)

                tfinr=tf.tensor_scatter_nd_add(tfinr,tf.constant([[dic[j]]],dtype=tf.int32),updates)

        return tfinr
ATT=atten()
flag=[]

for dic in alldic:

    for i in range(len(dic)):

        flag.append(dic[i])

flag=n.unique(flag)
def erratt():

    ans=tf.Variable(0,dtype=tf.float64)

    for i in flag:

        for j in rwneigh[i]:

            a=tf.gather(finr,i,axis=0)

            b=tf.gather(finr,j,axis=0)

            tmp=tf.tensordot(a,b,axes=1)

            tmp=tf.math.sigmoid(tmp)

            tmp=-1*tf.math.log(tmp)

            ans.assign_add(tmp)

    ans=tf.convert_to_tensor(ans)

    print(ans)

    return ans
erratt()
"""

def backprop(args):

    mod=args[0]

    loss=args[1]

    tape=tf.GradientTape(persistent=True)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    grads=tape.gradient(loss,mod.trainable_weights)

    optimizer.apply_gradients(zip(grads,mod.trainable_weights))

"""
m_his=[]
conrep=tf.zeros((len(deg),300),dtype=tf.float64)
global conrep
finr
for epochs in range(10):

    with tf.GradientTape(persistent=True) as tape:

        #Forward Propogation Pool

        st=time.time()

        """

        p=Pool()

        l=[]



        for i in range(len(allA)):

            l.append([tf.convert_to_tensor(AllX[i]),allm[i]])

        res3=p.map(forwardprop,l)

        p.close()

        p.join()

        """

        res3=[]

        for i in range(len(allA)):

            res3.append(forwardprop([tf.convert_to_tensor(AllX[i]),allm[i]]))

        print(f"count {epochs}")

        print(f"forwardprop {time.time()-st}")

        st=time.time()

        conrep=tf.zeros((len(deg),300),dtype=tf.float64)

        for i in range(len(res3)):

            arr=res3[i]

            dic=alldic[i]

            arr=tf.cast(arr,dtype=tf.float64)

            for j in range(arr.shape[0]):

                tmp=tf.gather(arr,j,axis=0)

                tmp=tf.expand_dims(tmp,axis=0)

                conrep=tf.tensor_scatter_nd_add(conrep,tf.constant([[dic[j]]],dtype=tf.int32),tmp)

        conrep=conrep/len(deg)

        #Attention

        conrep=tf.cast(conrep,dtype=tf.float32)

        finr=ATT(res3)

        print(f"attention {time.time()-st}")

        """

        #Error calculation Pool for subgraphs

        p=Pool()

        l=[]

        for i in len(res3):

            l.append([res3[i],alldic[i]])

        res4=p.map(err,l)

        p.close()

        p.join()

        """

        st=time.time()

        #Error calculation for attention layer

        #att_loss=erratt()

        ans=tf.convert_to_tensor(0,dtype=tf.float64)

        for i in flag:

            for j in rwneigh[i]:

                a=tf.gather(finr,i,axis=0)

                b=tf.gather(finr,j,axis=0)

                tmp=tf.tensordot(a,b,axes=1)

                tmp=tf.math.sigmoid(tmp)

                tmp=-1*tf.math.log(tmp)

                ans=tf.math.add(ans,tmp)

        att_loss=tf.convert_to_tensor(ans)

        print(att_loss)

        print(f"losscal {time.time()-st}")

    st=time.time()

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    grads=tape.gradient(att_loss,ATT.trainable_weights)

    optimizer.apply_gradients(zip(grads,ATT.trainable_weights))

    print(time.time()-st)

    """

    #Error back_prop for subgraphs Pool

    p=Pool()

    l=[]

    #total loss over all sub graphs

    total_loss=0

    for i in range(len(allm)):

        l.append([m,att_loss])

        #total_loss+=res4[i]

    p.map(backprop,l)

    p.close()

    p.join()

    """

    for i in range(len(allm)):

        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

        grads=tape.gradient(att_loss,allm[i].trainable_weights)

        optimizer.apply_gradients(zip(grads,allm[i].trainable_weights))

    print(time.time()-st)

    m_his.append([i,att_loss])
"""

a=tf.constant(n.ones((300)))

b=tf.constant(n.ones((300)))

st=time.time()

for i in range(1000):

    for j in range(10):

        tf.tensordot(a,b,axes=1)

print(time.time()-st)

"""
att_loss
grads
class tmpmodel(keras.Model):

    def __init__(self):

        super().__init__()

        self.l=layers.Dense(300,activation="tanh")

    def __call__(self,inputs):

        return self.l(inputs)
def forward(args):

    inputs=args[0]

    m=args[1]

    return m(inputs)
tmpm=tmpmodel()
global fin
fin=tf.zeros((3,300),dtype=tf.float64)
class tatten(keras.Model):

    def __init__(self):

        super().__init__()

        self.l=layers.Dense(300,activation="tanh")

    def compute_output_shape(self,input_shape):

        return (3,300)

    def call(self,y):

        atty=self.l(crep)

        atty=tf.cast(atty,dtype=tf.float64)

        tfinr=tf.zeros((3,300),dtype=tf.float64)

        y=tf.cast(y,dtype=tf.float64)

        for i in range(y.shape[0]):

            a=tf.gather(atty,i,axis=0)

            b=tf.gather(y,i,axis=0)

            x=tf.tensordot(a,b,axes=1)

            x=tf.cast(x,dtype=tf.float64)

            x=tf.math.sigmoid(x)

            updates=b*x

            updates=tf.expand_dims(updates,axis=0)

            tfinr=tf.tensor_scatter_nd_add(tfinr,tf.constant([[i]],dtype=tf.int32),updates)

        return tfinr
tmpatt=tatten()
global crep
crep=tf.zeros((3,300),dtype=tf.float64)
import tensorflow as tf

import keras

import random

import numpy as n

import pandas as pd

from scipy.sparse import csr_matrix

from multiprocessing import Pool,Process

import time

import math

from keras import layers

class tmpmodel(keras.Model):

    def __init__(self):

        super().__init__()

        self.l=layers.Dense(300,activation="tanh")

    def __call__(self,inputs):

        return self.l(inputs)

def forward(args):

    inputs=args[0]

    m=args[1]

    return m(inputs)

tmpm=tmpmodel()

global fin

fin=tf.zeros((3,300),dtype=tf.float64)

class tatten(keras.Model):

    def __init__(self):

        super().__init__()

        self.l=layers.Dense(300,activation="tanh")

    def compute_output_shape(self,input_shape):

        return (3,300)

    def call(self,y):

        atty=self.l(crep)

        atty=tf.cast(atty,dtype=tf.float64)

        tfinr=tf.zeros((3,300),dtype=tf.float64)

        y=tf.cast(y,dtype=tf.float64)

        for i in range(y.shape[0]):

            a=tf.gather(atty,i,axis=0)

            b=tf.gather(y,i,axis=0)

            x=tf.tensordot(a,b,axes=1)

            x=tf.cast(x,dtype=tf.float64)

            x=tf.math.sigmoid(x)

            updates=b*x

            updates=tf.expand_dims(updates,axis=0)

            tfinr=tf.tensor_scatter_nd_add(tfinr,tf.constant([[i]],dtype=tf.int32),updates)

        return tfinr

tmpatt=tatten()

global crep

crep=tf.zeros((3,300),dtype=tf.float64)

for epochs in range(10):

    with tf.GradientTape(persistent=True) as Tape:

        y=forward([tf.convert_to_tensor([[1],[2],[3]],dtype=tf.float32),tmpm])

        crep=tf.zeros((3,300),dtype=tf.float64)

        for i in range(3):

            tmp=tf.ones((1,300),dtype=tf.float64)

            crep=tf.tensor_scatter_nd_add(crep,tf.constant([[i]],dtype=tf.int32),tmp)

            tmp=tf.gather(y,i,axis=0)

            tmp=tf.expand_dims(tmp,axis=0)

            tmp=tf.cast(tmp,dtype=tf.float64)

            crep=tf.tensor_scatter_nd_add(crep,tf.constant([[i]],dtype=tf.int32),tmp)

        crep=crep/3

        #Attention

        crep=tf.cast(crep,dtype=tf.float32)

        fin=tmpatt(y)

        #Error calculation for attention layer

        ans=tf.convert_to_tensor(0,dtype=tf.float64)

        for i in range(y.shape[0]):

            for j in range(y.shape[0]):

                if i!=j:

                    a=tf.gather(fin,i,axis=0)

                    b=tf.gather(fin,j,axis=0)

                    tmp=tf.tensordot(a,b,axes=1)

                    tmp=tf.math.sigmoid(tmp)

                    tmp=-1*tf.math.log(tmp)

                    ans=tf.math.add(ans,tmp)

        att_loss=tf.convert_to_tensor(ans)

        print(att_loss)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    grads=Tape.gradient(att_loss,tmpatt.trainable_weights)

    optimizer.apply_gradients(zip(grads,tmpatt.trainable_weights))

    grads=Tape.gradient(att_loss,tmpm.trainable_weights)

    optimizer.apply_gradients(zip(grads,tmpm.trainable_weights))
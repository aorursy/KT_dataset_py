import os,time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.special import comb
from sklearn import preprocessing
import random,copy,pickle
import statsmodels.api as sm
import pexpect

dir0='/kaggle/input/liverpool-ion-switching/'
dir='/kaggle/input/6th-place-ion/'
dir1='/kaggle/working/'
groups=[[0,1],[2,6],[3,7],[5,8],[4,9]]
nchannel=[1,1,3,5,10]
df_train = pd.read_csv(dir0+"/train.csv")

signal_clean=df_train['signal'].values
batch=[1,6,7,8,9]
for i in batch:
    if i!=1:
        x=np.arange(500000)
        signal_clean[500000*i:500000*(i+1)]=df_train['signal'].values[500000*i:500000*(i+1)]-5*np.sin(np.pi*x/500000)
    else:
        x=np.arange(100000)
        signal_clean[500000:600000]=df_train['signal'].values[500000:600000]-5*np.sin(np.pi*x/500000)
plt.figure(figsize=(16,5))
plt.plot(signal_clean)
plt.show()
df_train['signal']=signal_clean
for mm in range(5):
    mm=0 # mm=0,1,2,3,or 4
    n_para=groups[mm][0]

    K=4
    zopen=np.array([0,0,1,1],dtype='int32')
    M=nchannel[mm]
    pi=np.array([0.,0.,0.,1.])
    if mm==0: pi=np.array([1.,0.,0.,0.])
    A=np.random.rand(K*K).reshape((K,K))
    for i in range(K):
        for k in range(K):
            if i==k:
                A[i,i]=1.
            else:
                A[i,k]=random.choice([0.5,0.001])
    A[3,0],A[0,3],A[3,1],A[1,3]=0,0,0,0
    A[2,0],A[0,2]=0,0
    A=A/np.sum(A,axis=0)#.reshape(K,1)

    B=np.zeros((M+1,K**M))
    for i in range(K**M):
        cpi=np.zeros(M,dtype='int32')
        tmpi=i
        for j in range(M-1,-1,-1):
            cpi[j]=tmpi/K**j
            tmpi=tmpi-cpi[j]*K**j
        B[np.sum(zopen[cpi]),i]=1

    cha=np.zeros(1,dtype='int64')
    for n in groups[mm]:
        tmp=df_train['open_channels'].values[500000*n:500000*(n+1)]
        cha=np.append(cha,tmp)
    cha=cha[1:]
    #A=np.fromfile('A004_lh=1323450.024.bin')
    cha.tofile(dir1+'cha'+'{0:03d}'.format(n_para)+'.bin')
    pi.tofile(dir1+'pi'+'{0:03d}'.format(n_para)+'.bin')
    A.tofile(dir1+'A'+'{0:03d}'.format(n_para)+'.bin')
    B.tofile(dir1+'B'+'{0:03d}'.format(n_para)+'.bin')
    #A=np.fromfile(dir1+'A'+'{0:03d}'.format(n_para)+'.bin')

    prc = pexpect.spawn("/bin/bash")
    prc.sendline("export OMP_NUM_THREADS=1")
    prc.sendline(dir+"./a.out 1 "+str(n_para)+" "+str(M)+" "+str(K)+" "+str(100000)+" "+str(10)+" "+str(200)+" "
                 +dir1+">log"+'{0:03d}'.format(n_para)+"-"+str(i)+'.txt')
    data0=''
    time.sleep(1)
    f = open(dir1+'log'+'{0:03d}'.format(n_para)+"-"+str(i)+'.txt')
    while 1==1:
        time.sleep(1)
        data1 = f.read()
        if data1!=data0 and data1!='':
            data0=copy.copy(data1)
            print(data1) 
        if 'END' in data1:
            break

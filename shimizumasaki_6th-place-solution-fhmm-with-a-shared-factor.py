import numpy as np
import pandas as pd
import pickle,time,copy
import pexpect
import matplotlib.pyplot as plt
import numpy.fft as fft
from scipy import signal
import glob
dir0='/kaggle/input/liverpool-ion-switching/'
dir='/kaggle/input/6th-place-ion/'
def write_params(M,file_A,mu=np.zeros(1),var=np.zeros(1),n_para=0,K=4,level=1):
    zopen=np.array([0,0,1,1],dtype='int32')
    B=np.zeros((M+1,K**M))
    for i in range(K**M):
        cpi=np.zeros(M,dtype='int32')
        tmpi=i
        for j in range(M-1,-1,-1):
            cpi[j]=tmpi/K**j
            tmpi=tmpi-cpi[j]*K**j
        B[np.sum(zopen[cpi]),i]=1

    A=np.fromfile(file_A)
    if level==0:
        pi=np.array([1.,0,0,0])
    else:
        pi=np.array([0,0,0,1.])
    if mu[0]==0:
        if M<10:
            aa,bb=1.2346256990189566, -2.735787099027291
        else:
            #aa,bb=1.2334311467598833, -5.459027096569211
            aa,bb=1.2334311467598833, -5.5
        mu=aa*np.arange(M+1)+bb
    if var[0]==0:
        if M==1: var=0.0617*np.ones(1)
        if M==3: var=0.0778*np.ones(1)
        if M==5: var=0.0845*np.ones(1)
        if M==10: var=0.165*np.ones(1) #もともと0.170
    pi.tofile('pi'+'{0:03d}'.format(n_para)+'.bin')
    A.tofile('A'+'{0:03d}'.format(n_para)+'.bin')
    B.tofile('B'+'{0:03d}'.format(n_para)+'.bin')
    mu.tofile('mu'+'{0:03d}'.format(n_para)+'.bin')
    var.tofile('var'+'{0:03d}'.format(n_para)+'.bin')
def clean_hum_noise_STFT(sig,pz,aa,bb,dd=1,nn=100000,nl=22500,thr=0.005):
    nperseg=nl
    hw=int(50*nperseg/10000+0.000001)
    noverlap=nperseg-10
    pp=np.max(pz,axis=1)
    cp=np.argmax(pz,axis=1)
    sig_clean=copy.copy(sig)
    x=sig-aa*cp-bb
    f, t, Z = signal.stft(x,fs=10000,nperseg=nperseg,noverlap=noverlap)
    Z1=copy.copy(Z)
    Z1=np.mean(np.abs(Z1),axis=1)
    tmp=np.where(Z1>thr)
    Z[tmp]=0
    _, sig_clean = signal.istft(Z,fs=10000,nperseg=nperseg,noverlap=noverlap)
    plt.plot(x-sig_clean)
    plt.show()
    Z0=np.mean(np.abs(Z),axis=1)
    plt.scatter(f[hw-5:hw+6],Z1[hw-5:hw+6])
    plt.scatter(f[hw-5:hw+6],Z0[hw-5:hw+6])
    plt.show()
    plt.scatter(f,Z0)
    plt.ylim([0,0.01])
    plt.show()
    return sig_clean+aa*cp+bb
df_test = pd.read_csv(dir0+"test.csv")
signal_clean=df_test['signal'].values
batch=[0,1,4,6,7,8,10]
x=np.arange(100000)
for i in batch:
    if i!=10:
        signal_clean[100000*i:100000*(i+1)]=df_test['signal'].values[100000*i:100000*(i+1)]-5*np.sin(np.pi*x/500000)
    else:
        x=np.arange(500000)
        signal_clean[100000*i:100000*(i+5)]=df_test['signal'].values[100000*i:100000*(i+5)]-5*np.sin(np.pi*x/500000)
plt.figure(figsize=(16,5))
plt.plot(signal_clean)
plt.show()
df_test['signal']=signal_clean
level=[0,2,3,0,1, 4,3, 4,0,2,0,0,0,0,0,0,0,0,0,0]
noc = [3,3,5,3,1,10,5,10,3,3,3,3,3,3,3,3,3,3,3,3]
nit = [10]*20
nl  = [15000]*20
nl[5]=22500
nl[7]=22500
thr,ratio=0.4, 1.
files_A=[dir+'A000_lh30373.221.bin',dir+'/A002_lh245363.418.bin',dir+'/A003_lh672353.458.bin'
         ,dir+'/A015_lh937508.257.bin',dir+'/A034_lh1323450.005.bin']
cp=np.array(2000000,dtype='int64')
w=np.ones(11)
for n in range(20):
    sig=df_test['signal'].values[100000*n:100000*(n+1)]
    for i in range(nit[n]):
        n_para,M,K=n,noc[n],4
        sig.tofile('sig'+'{0:03d}'.format(n_para)+'.bin') 
        if i==0:
            write_params(M,files_A[level[n]],n_para=n_para,level=level[n])
        else:
            write_params(M,files_A[level[n]],mu=mu,var=var,n_para=n_para,level=level[n])
        time.sleep(1.)
        prc = pexpect.spawn("/bin/bash")
        prc.sendline(dir+"a.out 0 "+str(n_para)+" "+str(M)+" "+str(K)+" "+str(100000)+" "+str(1)+" "+"5 >log"
                     +'{0:03d}'.format(n_para)+"-"+str(i)+".txt")
        data0=''
        time.sleep(2.)
        f = open('log'+'{0:03d}'.format(n_para)+"-"+str(i)+'.txt')
        while 1==1:
            time.sleep(0.1)
            data1 = f.read()
            if data1!=data0 and data1!='':
                data0=copy.copy(data1)
                print(n_para,data1)#,data1[0:3]) 
            if 'END' in data1:
                break
        time.sleep(1.)

        files_pz=sorted(glob.glob('pz'+'{0:03d}'.format(n_para)+'*.bin'))
        if len(files_pz)>1: print('many files!')
        pz=np.fromfile(files_pz[0]).reshape(-1,M+1)
        files_mu=sorted(glob.glob('mu'+'{0:03d}'.format(n_para)+'*_opt.bin'))
        mu=np.fromfile(files_mu[0])
        aa,bb=mu[1]-mu[0],mu[0]
        files_var=sorted(glob.glob('var'+'{0:03d}'.format(n_para)+'*_opt.bin'))
        var=np.fromfile(files_var[0])
        print(aa,bb,var,mu)
        sig=df_test['signal'].values[100000*n:100000*(n+1)]
        #sig=clean_hum_noise(sig,pz,aa,bb,thr,dd[n],ratio)
        cp0=np.argmax(pz*w[0:noc[n]+1],axis=1)
        sig=clean_hum_noise_STFT(sig,pz,aa,bb,nl=nl[n])
        print(i,np.sum(abs(cp0-cw[100000*n:100000*(n+1)])))
    cp[100000*n:100000*(n+1)]=np.argmax(pz*w[0:noc[n]+1],axis=1)
plt.plot(cp)
df=pd.read_csv(dir+'submission.csv')
df['open_channels']=cp
df.to_csv('submission.csv',index=False,float_format='%.4f')
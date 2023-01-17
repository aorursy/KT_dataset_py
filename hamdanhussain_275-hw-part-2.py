# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# Any results you write to the current directory are saved as output.
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



import pickle as pickle
        
import random, sys, keras

import pandas as pd
import numpy as np
with open("/kaggle/input/csc-275/RML2016.10a_dict.pkl",'rb') as file:
    Xd = pickle.load(file,encoding='bytes')
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'BPSK'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
    if(df['snr'][i]==16):
        ind.append(i)
    
import matplotlib.pyplot as plt
for i in range(0,1000,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='green',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in BPSK SNR 12")
    plt.legend()
    plt.show()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
#print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'8PSK'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
  if(df['snr'][i]==16):
    ind.append(i)
import matplotlib.pyplot as plt
for i in range(0,100,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='blue',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in 8PSK SNR 12")
    plt.legend()
    plt.show()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
#print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'AM-DSB'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
  if(df['snr'][i]==16):
    ind.append(i)
import matplotlib.pyplot as plt
for i in range(0,100,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='blue',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in 8 AM-DSB 12")
    plt.legend()
    plt.show()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
#print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'AM-SSB'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
  if(df['snr'][i]==16):
    ind.append(i)
import matplotlib.pyplot as plt
for i in range(0,100,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='blue',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in AM-SSB SNR 12")
    plt.legend()
    plt.show()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
#print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'BPSK'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
  if(df['snr'][i]==16):
    ind.append(i)
import matplotlib.pyplot as plt
for i in range(0,100,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='blue',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in BPSK SNR 12")
    plt.legend()
    plt.show()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
#print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'CPFSK'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
  if(df['snr'][i]==16):
    ind.append(i)
import matplotlib.pyplot as plt
for i in range(0,100,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='blue',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in CPFSK SNR 12")
    plt.legend()
    plt.show()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
#print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'GFSK'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
  if(df['snr'][i]==16):
    ind.append(i)
import matplotlib.pyplot as plt
for i in range(0,100,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='blue',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in GFSK SNR 12")
    plt.legend()
    plt.show()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
#print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'PAM4'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
  if(df['snr'][i]==16):
    ind.append(i)
import matplotlib.pyplot as plt
for i in range(0,100,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='blue',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in PAM4 SNR 12")
    plt.legend()
    plt.show()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
#print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'QAM16'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
  if(df['snr'][i]==16):
    ind.append(i)
import matplotlib.pyplot as plt
for i in range(0,100,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='blue',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in QAM16 SNR 12")
    plt.legend()
    plt.show()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
#print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'QAM64'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
  if(df['snr'][i]==16):
    ind.append(i)
import matplotlib.pyplot as plt
for i in range(0,100,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='red',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in QAM64 SNR 12")
    plt.legend()
    plt.show()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
#print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'QPSK'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
  if(df['snr'][i]==16):
    ind.append(i)
import matplotlib.pyplot as plt
for i in range(0,100,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='yellow',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in QPSK SNR 12")
    plt.legend()
    plt.show()
snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X=[]
Y=[]
lbl = []
#print(mods)
for mod in mods:
    for snr in snrs:
        if(mod==b'WBFM'and snr==12):
            test = Xd[(mod,snr)]
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
                Y.append([test[0]+1j*test[1]])
X = np.vstack(X)
Y= np.vstack(Y)
df= pd.DataFrame(lbl,columns=["mod","snr"])
df['snr'].value_counts()
ind = []
for i in range(0,df.shape[0]):
  if(df['snr'][i]==16):
    ind.append(i)
import matplotlib.pyplot as plt
for i in range(0,100,1):
    x = X[i][0]
    y= X[i][1]
    fig = plt.figure()
    plt.scatter(x,y,c='green',label=i)
    plt.xlabel("I")
    plt.ylabel("Q")
    plt.title("Data representation variance in WBFM SNR 12")
    plt.legend()
    plt.show()
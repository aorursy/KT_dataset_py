import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt

df_train = pd.read_csv("../input/train.csv",nrows=2000) 
#df_test = pd.read_csv("test.csv")
X_train = df_train.drop("label",axis=1)

X_train_n = [0]*10
for n in range(10):
    X_train_n[n] = df_train.loc[df_train.label==n].drop("label",axis=1)

plt.figure(figsize=(12,12))

for i in range(10):
    plt.subplot(10,4,i+1)
    plt.imshow(X_train_n[i].mean(axis=0).reshape(28,28),interpolation='nearest')   
    #plt.axis('tight')
plt.figtext(0.5, 0.965,"average image",size='large')

plt.show()

# use corr
nr = 0
for k in range(50):
    maxcorr=0
    for i in range(10):
        a = X_train_n[i][:10].mean(axis=0)
        corr = np.corrcoef(X_train.values[k,:],a)[0,1]
        #print i,corr
        if maxcorr <= corr:
            maxcorr = corr
            pred = i
    if pred == df_train.label[k]:
        nr += 1
    print (k,maxcorr,pred,df_train.label[k])
print (nr / 50)
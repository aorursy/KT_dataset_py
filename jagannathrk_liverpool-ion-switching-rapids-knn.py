import sys

!cp ../input/rapids/rapids.0.12.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/
import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

from scipy.stats import mode

from sklearn.metrics import f1_score, accuracy_score

from cuml.neighbors import KNeighborsClassifier, NearestNeighbors

import cuml; cuml.__version__
train = pd.read_csv('/kaggle/input/data-without-drift/train_clean.csv')

train['group'] = -1

x = [(0,500000),(1000000,1500000),(1500000,2000000),(2500000,3000000),(2000000,2500000)]

for k in range(5): train.iloc[x[k][0]:x[k][1],3] = k

    

res = 1000

plt.figure(figsize=(20,5))

plt.plot(train.time[::res],train.signal[::res])

plt.plot(train.time,train.group,color='black')

plt.title('Clean Train Data. Blue line is signal. Black line is group number.')

plt.xlabel('time'); plt.ylabel('signal')

plt.show()
test = pd.read_csv('/kaggle/input/data-without-drift/test_clean.csv')

test['group'] = -1

x = [[(0,100000),(300000,400000),(800000,900000),(1000000,2000000)],[(400000,500000)], 

     [(100000,200000),(900000,1000000)],[(200000,300000),(600000,700000)],[(500000,600000),(700000,800000)]]

for k in range(5):

    for j in range(len(x[k])): test.iloc[x[k][j][0]:x[k][j][1],2] = k

        

res = 400

plt.figure(figsize=(20,5))

plt.plot(test.time[::res],test.signal[::res])

plt.plot(test.time,test.group,color='black')

plt.title('Clean Test Data. Blue line is signal. Black line is group number.')

plt.xlabel('time'); plt.ylabel('signal')

plt.show()
step = 0.2

pt = [[],[],[],[],[]]

cuts = [[],[],[],[],[]]

for g in range(5):

    mn = train.loc[train.group==g].signal.min()

    mx = train.loc[train.group==g].signal.max()

    old = 0

    for x in np.arange(mn,mx+step,step):

        sg = train.loc[(train.group==g)&(train.signal>x-step/2)&(train.signal<x+step/2)].open_channels.values

        if len(sg)>100:

            m = mode(sg)[0][0]

            pt[g].append((x,m))

            if m!=old: cuts[g].append(x-step/2)

            old = m

    pt[g] = np.vstack(pt[g])

    

models = ['1 channel low prob','1 channel high prob','3 channel','5 channel','10 channel']

plt.figure(figsize=(15,8))

for g in range(5):

    plt.plot(pt[g][:,0],pt[g][:,1],'-o',label='Group %i (%s model)'%(g,models[g]))

plt.legend()

plt.title('Traing Data Open Channels versus Clean Signal Value',size=16)

plt.xlabel('Clean Signal Value',size=16)

plt.ylabel('Open Channels',size=16)

plt.show()
import warnings

warnings.filterwarnings("ignore")

for g in range(5):

    if g==0: res = 100

    else: res = 10



    plt.figure(figsize=(16,8))

    plt.subplot(1,2,1)

    for k in range(0,11):

        idx = np.array( train.loc[(train.open_channels==k) & (train.group==g)].index )

        if len(idx)==0: continue

        plt.scatter(train.signal[idx-1],train.signal[idx],s=0.01,label='%i open channels'%k)

    plt.xlabel('Previous Signal Value',size=14)

    plt.ylabel('Signal Value',size=14)

    lgnd = plt.legend(numpoints=1, fontsize=10)

    for k in range( len(lgnd.legendHandles) ):

        lgnd.legendHandles[k]._sizes = [30]

    

    data = test.loc[test.group==g]

    #plt.scatter(data.signal[:-1][::res],data.signal[1:][::res],s=0.1,color='black')

    xx = plt.xlim(); yy = plt.ylim()

    for k in range(len(cuts[g])):

        if (g!=4)|(k!=0): plt.plot([xx[0],xx[1]],[cuts[g][k],cuts[g][k]],':',color='black')

    plt.title('Train Data in group %i'%g,size=16)

    

    plt.subplot(1,2,2)

    plt.scatter(data.signal[:-1][::res],data.signal[1:][::res],s=0.1,color='black')

    plt.xlim(xx); plt.ylim(yy)

    for k in range(len(cuts[g])):

        if (g!=4)|(k!=0): plt.plot([xx[0],xx[1]],[cuts[g][k],cuts[g][k]],':',color='black')

        if (g==4)&(k!=0): plt.text(xx[0]+1,cuts[g][k],'%i open channels'%(k+2),size=12)

        elif g!=4: plt.text(xx[0]+1,cuts[g][k],'%i open channels'%(k+1),size=14)

    plt.xlabel('Previous Signal Value',size=14)

    plt.ylabel('Signal Value',size=14)

    plt.title('Unknown Test Data in group %i'%g,size=16)



    plt.show()
def wiggle(df, row, plt, xx=None, yy=None):

    plt.plot([-3,-2,-1,0,1,2,3],df.loc[df.index[row-3:row+4],'signal'],'-')

    sizes = np.array([1,2,3,12,3,2,1])*50

    colors = ['red','red','red','green','blue','blue','blue']

    for k in range(7):

        plt.scatter(k-3,df.loc[df.index[row+k-3],'signal'],s=sizes[k],color=colors[k])

    if xx!=None: plt.xlim(xx)

    if yy!=None: plt.ylim(yy)

    return plt.xlim(),plt.ylim()



row=2; col=4;

np.random.seed(42)

plt.figure(figsize=(4*col,4*row))

for k in range(row*col):

    plt.subplot(row,col,k+1)

    r = np.random.randint(2e6)

    wiggle(test,r,plt)

    if k%col==0: plt.ylabel('signal')

    g = test.loc[r,'group']

    plt.title('Test row %i group %i'%(r,g))

plt.tight_layout(pad=3.0)

plt.show()
%%time



KNN = 100

batch = 1000



test_pred = np.zeros((test.shape[0]),dtype=np.int8)

for g in [0,1,2,3,4]:

    print('Infering group %i'%g)

    

    # TRAIN DATA

    data = train.loc[train.group==g]

    X_train = np.zeros((len(data)-6,7))

    X_train[:,0] = 0.25*data.signal[:-6]

    X_train[:,1] = 0.5*data.signal[1:-5]

    X_train[:,2] = 1.0*data.signal[2:-4]

    X_train[:,3] = 4.0*data.signal[3:-3]

    X_train[:,4] = 1.0*data.signal[4:-2]

    X_train[:,5] = 0.5*data.signal[5:-1]

    X_train[:,6] = 0.25*data.signal[6:]

    y_train = data.open_channels[3:].values



    # TEST DATA

    data = test.loc[test.group==g]

    X_test = np.zeros((len(data)-6,7))

    X_test[:,0] = 0.25*data.signal[:-6]

    X_test[:,1] = 0.5*data.signal[1:-5]

    X_test[:,2] = 1.0*data.signal[2:-4]

    X_test[:,3] = 4.0*data.signal[3:-3]

    X_test[:,4] = 1.0*data.signal[4:-2]

    X_test[:,5] = 0.5*data.signal[5:-1]

    X_test[:,6] = 0.25*data.signal[6:]



    # HERE IS THE CORRECT WAY TO USE CUML KNN 

    #model = KNeighborsClassifier(n_neighbors=KNN)

    #model.fit(X_train,y_train)

    #y_hat = model.predict(X_test)

    #test_pred[test.group==g][1:-1] = y_hat

    #continue

    

    # WE DO THIS BECAUSE CUML v0.12.0 HAS A BUG

    model = NearestNeighbors(n_neighbors=KNN)

    model.fit(X_train)

    distances, indices = model.kneighbors(X_test)



    # FIND PREDICTIONS OURSELVES WITH STATS.MODE

    ct = indices.shape[0]

    pred = np.zeros((ct+6),dtype=np.int8)

    it = ct//batch + int(ct%batch!=0)

    #print('Processing %i batches:'%(it))

    for k in range(it):

        a = batch*k; b = batch*(k+1); b = min(ct,b)

        pred[a+3:b+3] = mode( y_train[ indices[a:b].astype(int) ], axis=1)[0][:,0]

        #print(k,', ',end='')

    #print()

    test_pred[test.group==g] = pred
data1 = test.loc[test.group==4].iloc[3:]

data1.reset_index(inplace=True)



data2 = train.loc[train.group==4].iloc[3:]

data2.reset_index(inplace=True)



for j in range(5):

    r = np.random.randint(data1.shape[0])

    distances, indices = model.kneighbors(X_test[r:r+1,])



    row=2; 

    plt.figure(figsize=(16,row*4))

    for k in range(row*4):

        if k in [1,2,3]: continue

        plt.subplot(row,4,k+1)

        if k==0: 

            xx,yy = wiggle(data1,r,plt)

            g = data1.loc[r,'group']

            rw = data1.loc[r,'index']

            plt.title('UNKNOWN Test row %i group %i'%(rw,g))

        else:

            r=indices[0,k-4].astype('int')

            wiggle(data2,r,plt,xx,yy)

            g = data2.loc[r,'group']

            rw = data2.loc[r,'index']

            t = data2.loc[r,'open_channels']

            plt.title('LABEL = %i. Train row %i group %i'%(t,rw,g))

        if k%4==0: plt.ylabel('signal')

    plt.tight_layout(pad=3.0)

    plt.show()
sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')

sub.open_channels = test_pred

sub.to_csv('submission.csv',index=False,float_format='%.4f')



res=200

plt.figure(figsize=(20,5))

plt.plot(sub.time[::res],sub.open_channels[::res])

plt.show()
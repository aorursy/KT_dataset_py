import sys

!cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz

!cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null

sys.path = ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib/python3.6"] + sys.path

sys.path = ["/opt/conda/envs/rapids/lib"] + sys.path 

!cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/

sys.path
import numpy as np 

import pandas as pd 

import cuml

import matplotlib.pyplot as plt

from scipy.stats import mode

from sklearn.metrics import f1_score, accuracy_score

from cuml.neighbors import KNeighborsClassifier, NearestNeighbors

import cuml; cuml.__version__

from scipy import signal

from scipy.fft import fftshift

from tqdm import tqdm_notebook as tqdm



# visualize

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import matplotlib.style as style

import seaborn as sns

from matplotlib.ticker import ScalarFormatter

sns.set_context("talk")

style.use('fivethirtyeight')

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
plt.figure(figsize=(20, 10))

plt.plot(train["time"], train["signal"])

plt.title("Signal data", fontsize=20)

plt.xlabel("Time", fontsize=18)

plt.ylabel("Signal", fontsize=18)

plt.show()

fig = make_subplots(rows=5, cols=2, subplot_titles=["Batch #{}".format(i) for i in range(10)])

i = 0



for row in range(1, 6):

    for col in range(1, 3):   

        

        data = train.iloc[(i * 500000):((i+1) * 500000 + 1)]['open_channels'].value_counts(sort=False).values

        fig.add_trace(go.Bar(x=list(range(11)), y=data), row=row, col=col)

        

        i += 1



fig.update_layout(title_text="Target distribution in different batches", height=1200, showlegend=True)

fig.show()
#train = train.copy()

train['batches'] = (train.index // 500_000) + 1

#train
from itertools import cycle



color_pal = [x['color'] for x in plt.rcParams['axes.prop_cycle']]

color_cycle = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])



fig, axs = plt.subplots(10, 2, figsize=(20, 30))

axs = axs.flatten()

i = 0

for b, d in train.groupby('batches'):    

    d.set_index('time')['signal'].plot(style='.',

                                       title=f'train batch {b:0.0f} - signal',

                                       ax=axs[i],

                                       alpha=0.2,

                                      color=next(color_cycle))

    d.set_index('time')['open_channels'].plot(style='.',

                                              title=f'train batch {b:0.0f} - open_channels',

                                              ax=axs[i+1],

                                              alpha=0.5,

                                      color=next(color_cycle))

    i += 2

plt.tight_layout()
plt.figure(figsize=(20,5))

plt.plot(train.time, train.signal)

plt.plot(train.time, train.open_channels,alpha=0.7)

plt.show()
KNN = 200

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



    ct = indices.shape[0]

    pred = np.zeros((ct+6),dtype=np.int8)

    it = ct//batch + int(ct%batch!=0)

    print('Processing %i batches:'%(it))

    for k in range(it):

        a = batch*k; b = batch*(k+1); b = min(ct,b)

        pred[a+3:b+3] = np.median( y_train[ indices[a:b].astype(int) ], axis=1)

        #print(k,', ',end='')

    #print()

    test_pred[test.group==g] = pred
#missing = np.mean(X_test,axis=0)

#X_test = np.vstack((X_test,missing))

#X_test.shape
 # HERE IS THE CORRECT WAY TO USE CUML KNN 

##for g in [0,1,2,3,4]:    

#    model = KNeighborsClassifier(n_neighbors=KNN)

 #   model.fit(X_train,y_train)

  #  y_hat = model.predict(X_test)

   # test_pred[test.group==g][1:-1] = y_hat

    #continue




    # HERE IS THE CORRECT WAY TO USE CUML KNN 

#model = KNeighborsClassifier(n_neighbors=KNN)

#model.fit(X_train,y_train)

#y_hat = model.predict(X_test)

#test_pred[test.group==g][1:-1] = y_hat

    

   
test_pred[test.group==g].shape
sub = pd.read_csv('../input/liverpool-ion-switching/sample_submission.csv')

sub.open_channels = test_pred

sub.to_csv('submission.csv',index=False,float_format='%.4f')



res=200

plt.figure(figsize=(20,5))

plt.plot(sub.time[::res],sub.open_channels[::res])

plt.show()
#from sklearn.metrics import classification_report,confusion_matrix

#print(classification_report(test,test_pred))

#print('\n')

#print(confusion_matrix(test,test_pred))

#from sklearn.metrics import f1_score



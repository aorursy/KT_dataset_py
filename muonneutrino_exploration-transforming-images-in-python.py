import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
train = pd.read_json('../input/train.json')

train['inc_angle'] = pd.to_numeric(train['inc_angle'],errors='coerce')
def get_stats(train,label=1):

    train['max'+str(label)] = [np.max(np.array(x)) for x in train['band_'+str(label)] ]

    train['maxpos'+str(label)] = [np.argmax(np.array(x)) for x in train['band_'+str(label)] ]

    train['min'+str(label)] = [np.min(np.array(x)) for x in train['band_'+str(label)] ]

    train['minpos'+str(label)] = [np.argmin(np.array(x)) for x in train['band_'+str(label)] ]

    train['med'+str(label)] = [np.median(np.array(x)) for x in train['band_'+str(label)] ]

    train['std'+str(label)] = [np.std(np.array(x)) for x in train['band_'+str(label)] ]

    train['mean'+str(label)] = [np.mean(np.array(x)) for x in train['band_'+str(label)] ]

    train['p25_'+str(label)] = [np.sort(np.array(x))[int(0.25*75*75)] for x in train['band_'+str(label)] ]

    train['p75_'+str(label)] = [np.sort(np.array(x))[int(0.75*75*75)] for x in train['band_'+str(label)] ]

    train['mid50_'+str(label)] = train['p75_'+str(label)]-train['p25_'+str(label)]



    return train

train = get_stats(train,1)

train = get_stats(train,2)
def plot_var(name,nbins=50):

    minval = train[name].min()

    maxval = train[name].max()

    plt.hist(train.loc[train.is_iceberg==1,name],range=[minval,maxval],

             bins=nbins,color='b',alpha=0.5,label='Boat')

    plt.hist(train.loc[train.is_iceberg==0,name],range=[minval,maxval],

             bins=nbins,color='r',alpha=0.5,label='Iceberg')

    plt.legend()

    plt.xlim([minval,maxval])

    plt.xlabel(name)

    plt.ylabel('Number')

    plt.show()
for col in ['inc_angle','min1','max1','std1','med1','mean1','mid50_1']:

    plot_var(col)

for col in ['min2','max2','std2','med2','mean2','mid50_2']:

    plot_var(col)
train_stats = train.drop(['id','is_iceberg','band_1','band_2'],axis=1)
corr = train_stats.corr()

fig = plt.figure(1, figsize=(10,10))

plt.imshow(corr,cmap='inferno')

labels = np.arange(len(train_stats.columns))

plt.xticks(labels,train_stats.columns,rotation=90)

plt.yticks(labels,train_stats.columns)

plt.title('Correlation Matrix of Global Variables')

cbar = plt.colorbar(shrink=0.85,pad=0.02)

plt.show()
icebergs = train[train.is_iceberg==1].sample(n=9,random_state=123)

ships = train[train.is_iceberg==0].sample(n=9,random_state=456)
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = np.reshape(np.array(icebergs.iloc[i,0]),(75,75))

    ax.imshow(arr,cmap='inferno')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = np.reshape(np.array(ships.iloc[i,0]),(75,75))

    ax.imshow(arr,cmap='inferno')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = np.reshape(np.array(icebergs.iloc[i,1]),(75,75))

    ax.imshow(arr,cmap='inferno')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = np.reshape(np.array(ships.iloc[i,1]),(75,75))

    ax.imshow(arr,cmap='inferno')

    

plt.show()
from scipy import signal



xder = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

yder = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

smooth = np.array([[1,1,1],[1,5,1],[1,1,1]])

xder2 = np.array([[-1,2,-1],[-3,6,-3],[-1,2,-1]])

yder2 = np.array([[-1,-3,-1],[2,6,2],[-1,-3,-1]])
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = signal.convolve2d(np.reshape(np.array(icebergs.iloc[i,0]),(75,75)),smooth,mode='valid')

    ax.imshow(arr,cmap='inferno')

    ax.set_title('Smoothed')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = signal.convolve2d(np.reshape(np.array(ships.iloc[i,0]),(75,75)),smooth,mode='valid')

    ax.imshow(arr,cmap='inferno')

    ax.set_title('Smoothed')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = signal.convolve2d(np.reshape(np.array(icebergs.iloc[i,0]),(75,75)),xder,mode='valid')

    ax.imshow(arr,cmap='inferno')

    ax.set_title('X-derivative')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = signal.convolve2d(np.reshape(np.array(ships.iloc[i,0]),(75,75)),xder,mode='valid')

    ax.imshow(arr,cmap='inferno')

    ax.set_title('X-derivative')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arrx = signal.convolve2d(np.reshape(np.array(icebergs.iloc[i,0]),(75,75)),xder,mode='valid')

    arry = signal.convolve2d(np.reshape(np.array(icebergs.iloc[i,0]),(75,75)),yder,mode='valid')

    ax.imshow(np.hypot(arrx,arry),cmap='inferno')

    ax.set_title('Gradient Magnitude')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arrx = signal.convolve2d(np.reshape(np.array(ships.iloc[i,0]),(75,75)),xder,mode='valid')

    arry = signal.convolve2d(np.reshape(np.array(ships.iloc[i,0]),(75,75)),yder,mode='valid')

    ax.imshow(np.hypot(arrx,arry),cmap='inferno')

    ax.set_title('Gradient Magnitude')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = signal.convolve2d(np.reshape(np.array(icebergs.iloc[i,0]),(75,75)),xder2,mode='valid')

    ax.imshow(arr,cmap='inferno')

    ax.set_title(r'Second X derivative')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arr = signal.convolve2d(np.reshape(np.array(ships.iloc[i,0]),(75,75)),xder2,mode='valid')

    ax.imshow(arr,cmap='inferno')

    ax.set_title(r'Second X derivative')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arrx = signal.convolve2d(np.reshape(np.array(icebergs.iloc[i,0]),(75,75)),xder2,mode='valid')

    arry = signal.convolve2d(np.reshape(np.array(icebergs.iloc[i,0]),(75,75)),yder2,mode='valid')



    ax.imshow(np.hypot(arrx,arry),cmap='inferno')

    ax.set_title('Laplacian')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arrx = signal.convolve2d(np.reshape(np.array(ships.iloc[i,0]),(75,75)),xder2,mode='valid')

    arry = signal.convolve2d(np.reshape(np.array(ships.iloc[i,0]),(75,75)),yder2,mode='valid')



    ax.imshow(np.hypot(arrx,arry),cmap='inferno')

    ax.set_title('Laplacian')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arrx = signal.convolve2d(np.reshape(np.array(icebergs.iloc[i,0]),(75,75)),xder,mode='valid')

    arry = signal.convolve2d(np.reshape(np.array(icebergs.iloc[i,0]),(75,75)),yder,mode='valid')

    arrx = signal.convolve2d(arrx,yder,mode='valid')

    arry = signal.convolve2d(arry,xder,mode='valid')

    ax.imshow(np.hypot(arrx,arry),cmap='inferno')

    ax.set_title('Curl of Gradient Magnitude')

    

plt.show()
# Plot band_1

fig = plt.figure(1,figsize=(15,15))

for i in range(9):

    ax = fig.add_subplot(3,3,i+1)

    arrx = signal.convolve2d(np.reshape(np.array(ships.iloc[i,0]),(75,75)),xder,mode='valid')

    arry = signal.convolve2d(np.reshape(np.array(ships.iloc[i,0]),(75,75)),yder,mode='valid')

    arrx = signal.convolve2d(arrx,yder,mode='valid')

    arry = signal.convolve2d(arry,xder,mode='valid')

    ax.imshow(np.hypot(arrx,arry),cmap='inferno')

    ax.set_title('Curl of Gradient Magnitude')

    

plt.show()
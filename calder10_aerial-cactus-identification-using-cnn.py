# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# Import libaries

import os

import time

import pandas as pd

import numpy as np

import cv2

from tqdm import tqdm

import matplotlib.pyplot as plt

from keras.models import Sequential,load_model 

from keras.layers import Dense,Conv2D,Dropout,MaxPooling2D,Flatten,BatchNormalization

from keras.callbacks import EarlyStopping

from keras import optimizers

from sklearn.metrics import confusion_matrix

import seaborn as sns



train_path='../input/train/train'

test_path='../input/test/test'
# Prepare train data 

label_train=pd.read_csv("../input/train.csv")

# Ordino il file in base all'id

label_train=label_train.sort_values(by=['id'])

# Creo 2 array distinti uno per gli id e l'altro per i label

id=label_train['id'].values

l=label_train['has_cactus'].values



train=[]

X=[]

Y=[]

a=0



for i in tqdm(sorted(os.listdir(train_path))):

    path=os.path.join(train_path,i)

    i=cv2.imread(path,cv2.IMREAD_COLOR)

    X.append(i)

    train.append([np.array(i),l[a]])

    a=a+1



train=np.array(train)

Y=train[:,1]

train=train[:,0]

X=np.array(X)



X.shape



X=X/255

train=train/255
# Plot the first 25 images in Training Set 

plt.figure(figsize = (30,30))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    if Y[i]==1:

        l="Has Cactus"

    elif Y[i]==0:

        l="No Cactus"

    plt.xlabel(l,fontsize=25)

    plt.imshow(train[i])

plt.suptitle("First 25 images in Training Set ",fontsize=30)
# Prepare Test data 

test_viz=[]

X_test=[]



for i in tqdm(os.listdir(test_path)):

    id=i

    path=os.path.join(test_path,i)

    i=cv2.imread(path,cv2.IMREAD_COLOR)

    X_test.append(i)

    test_viz.append([np.array(i),id])



X_test=np.array(X_test)

X_test.shape

test_viz=np.array(test_viz)

id_test=test_viz[:,1]

test_viz=test_viz[:,0]

test_viz.shape



X_test=X_test/255

test_viz=test_viz/255
# Plot the first 25 images in Test Set 

plt.figure(figsize = (30,30))

for i in range(25):

    plt.subplot(5,5,i+1)

    plt.xticks([])

    plt.yticks([])

    plt.grid(False)

    plt.imshow(test_viz[i])

plt.suptitle("First 25 images in Testing Set ",fontsize=30)
m=Sequential()

m.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu",input_shape=(32,32,3)))

m.add(Conv2D(filters=32,kernel_size=2,padding="same",activation="relu"))

m.add(Conv2D(filters=32,kernel_size=4,padding="same",activation="relu"))

m.add(MaxPooling2D(pool_size=2,strides=1))

m.add(Dropout(0.2))

m.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

m.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

m.add(Conv2D(filters=64,kernel_size=4,padding="same",activation="relu"))

m.add(MaxPooling2D(pool_size=2,strides=1))

m.add(Dropout(0.2))

m.add(Conv2D(filters=128,kernel_size=2,padding="same",activation="relu"))

m.add(Conv2D(filters=128,kernel_size=2,padding="same",activation="relu"))

m.add(Conv2D(filters=128,kernel_size=4,padding="same",activation="relu"))

m.add(MaxPooling2D(pool_size=2,strides=1))

m.add(Dropout(0.2))

m.add(Flatten())

m.add(Dense(32,activation="relu"))

m.add(Dense(1,activation="sigmoid"))

m.summary()
# training of the model

# el=EarlyStopping(min_delta=0.006,patience=5,restore_best_weights=True)

m.compile(loss="binary_crossentropy",optimizer='adam',metrics=["accuracy"])

s=time.time()

h=m.fit(X,Y,batch_size=128,validation_split=0.2,epochs=100)

e=time.time()

t=e-s

print("Addestramento completato in %d minuti e %d secondi" %(t/60,t*60))
acc=h.history['acc']

val_acc=h.history['val_acc']

loss=h.history['loss']

val_loss=h.history['val_loss']
# Trend of accuracy during the training 

plt.plot(acc)

plt.plot(val_acc)

plt.title('Cactus_identifier_net1 Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train','Validation'])

plt.show()
# Trend of loss during the training 

plt.plot(loss)

plt.plot(val_loss)

plt.title('Cactus_identifier_net1 Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train','Validation'])

plt.show()
pred=m.predict(X_test)

ids=[]

label=[]

a=0

for i in tqdm(os.listdir(test_path)):

    id=i

    ids.append(id)

    label.append(pred[a])

    a=a+1



label=np.array(label,dtype='float64')

out=pd.DataFrame({'id': ids,'has_cactus':label[:,0]})



out.to_csv('cactus_identifier_net.csv',index=False,header=True)
# Confusion matrix for the Training Set 

pred_train=m.predict(X)

p=[]

for i in pred_train:

    if i>0.5:

        p.append(1)

    elif i<0.5:

        p.append(0)

        

p=np.array(p,dtype='int')

Y=np.array(Y,dtype='int')



cm=confusion_matrix(Y,p)

cm_df = pd.DataFrame(cm,index = ['0 - No Cactus','1 - Has Cactus'],  columns = ['0 - No Catus','1 - Has Cactus'])

plt.figure(figsize=(10,10))

sns.heatmap(cm_df,annot=True,cmap="Blues_r",linewidth=0.5,square=True,fmt='g')



plt.ylabel("True Label ")

plt.xlabel("Predict Label")

plt.title("CONFUSION MATRIX FOR TRAINING SET")
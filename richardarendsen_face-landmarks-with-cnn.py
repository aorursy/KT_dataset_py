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
# load the dataset
Pface = np.moveaxis(np.load('../input/face_images.npz')['face_images'],-1,0)
LMs = pd.read_csv('../input/facial_keypoints.csv')

LMpos=LMs.columns.tolist()
print(LMs.isnull().sum())
iselect=np.nonzero(LMs.left_eye_center_x.notna() & LMs.right_eye_center_x.notna() &
         LMs.nose_tip_x.notna() & LMs.mouth_center_bottom_lip_x.notna())[0]

Spic=Pface.shape[1]
m=iselect.shape[0]
X=np.zeros((m,Spic,Spic,1))
Y=np.zeros((m,8))

X[:,:,:,0]=Pface[iselect,:,:]/255.0
Y[:,0]=LMs.left_eye_center_x[iselect]/Spic
Y[:,1]=LMs.left_eye_center_y[iselect]/Spic
Y[:,2]=LMs.right_eye_center_x[iselect]/Spic
Y[:,3]=LMs.right_eye_center_y[iselect]/Spic
Y[:,4]=LMs.nose_tip_x[iselect]/Spic
Y[:,5]=LMs.nose_tip_y[iselect]/Spic
Y[:,6]=LMs.mouth_center_bottom_lip_x[iselect]/Spic
Y[:,7]=LMs.mouth_center_bottom_lip_y[iselect]/Spic

print('# selected images = %d' %(m))
import matplotlib.pyplot as plt

n = 0
nrows = 4
ncols = 4
irand=np.random.choice(Y.shape[0],nrows*ncols)
fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=[ncols*2,nrows*2])
for row in range(nrows):
    for col in range(ncols):
        ax[row,col].imshow(X[irand[n],:,:,0], cmap='gray')
        ax[row,col].scatter(Y[irand[n],0::2]*Spic,Y[irand[n],1::2]*Spic,marker='X',c='r',s=100)
        ax[row,col].set_xticks(())
        ax[row,col].set_yticks(())
        ax[row,col].set_title('image index = %d' %(irand[n]),fontsize=10)
        n += 1

# Split the dataset
from sklearn.model_selection import train_test_split

random_seed=21
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.2, random_state=random_seed)
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import SGD

model = Sequential()
model.add(Conv2D(32, (3, 3), padding = 'same', activation='tanh', input_shape=(Spic, Spic, 1)))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='sigmoid'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error', optimizer=sgd)

model.fit(Xtrain, Ytrain, batch_size=128, epochs=10, validation_data = (Xtest, Ytest), verbose = 1)
Ytrain_pred = model.predict(Xtrain)
Ytest_pred = model.predict(Xtest)

n = 0
nrows = 4
ncols = 4
irand=np.random.choice(Ytest.shape[0],nrows*ncols)
fig, ax = plt.subplots(nrows,ncols,sharex=True,sharey=True,figsize=[ncols*2,nrows*2])
for row in range(nrows):
    for col in range(ncols):
        ax[row,col].imshow(Xtest[irand[n],:,:,0], cmap='gray')
        ax[row,col].scatter(Ytest[irand[n],0::2]*Spic,Ytest[irand[n],1::2]*Spic,marker='X',c='r',s=100)
        ax[row,col].scatter(Ytest_pred[irand[n],0::2]*Spic,Ytest_pred[irand[n],1::2]*Spic,marker='+',c='b',s=100)
        ax[row,col].set_xticks(())
        ax[row,col].set_yticks(())
        ax[row,col].set_title('image index = %d' %(irand[n]),fontsize=10)
        n += 1
plt.suptitle('x: Manual; +: CNN', fontsize=16)
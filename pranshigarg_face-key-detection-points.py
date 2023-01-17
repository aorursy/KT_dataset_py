# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_data=pd.read_csv('/kaggle/input/facial-keypoints-detection/training.zip')

test_data=pd.read_csv('/kaggle/input/facial-keypoints-detection/test.zip')
train_data.head()
train_data.isnull().sum()

null_c=train_data.columns.tolist()

null_c
for i in null_c:

    train_data[i]=train_data[i].transform(lambda x:x.fillna(x.mean()))
train_data.isnull().sum()
train_data['Image']=train_data['Image'].apply(lambda x: np.fromstring(x,sep=' '))
train_data['Image']
X_train=np.vstack(train_data['Image'].values)

X_train=X_train.reshape(-1,96,96,1)

X_train=X_train.astype(np.float32)
X_train.shape
train_data=train_data.drop(['Image'],axis=1)

Y_train=train_data.values.astype(np.float32)

Y_train.shape
import matplotlib.pyplot as plt
plt.imshow(X_train[5].reshape(96,96),cmap='gray')

plt.show()
def show_image(X, Y):

    img = np.copy(X)

    for i in range(0,Y.shape[0],2):

        if 0 < Y[i+1] < 96 and 0 < Y[i] < 96:

            img[int(Y[i+1]),int(Y[i]),0] = 255

    plt.imshow(img[:,:,0])
# Preview dataset samples

show_image(X_train[5], Y_train[5])
from keras.models import Sequential, Model

from keras.layers import Activation, Convolution2D, MaxPooling2D, Flatten, Dense, Dropout
model=Sequential()

model.add(Convolution2D(32,3,3,input_shape=(96,96,1)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Convolution2D(64,2,2))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Convolution2D(128,2,2))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

          

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.1))

model.add(Dense(30))
model.compile(optimizer='adam', 

              loss='mean_squared_error',

              metrics=['mae'])
hist=model.fit(X_train,Y_train,epochs = 100,validation_split = 0.2)
#hist=model.fit(X_train,Y_train,epochs = 50,batch_size = 256,validation_split = 0.2)
f=plt.figure()

plt.plot(hist.history['loss'],linewidth=3,label='train')

plt.plot(hist.history['val_loss'],linewidth=3,label='valid')

plt.grid()

plt.legend()

plt.xlabel('epoch')

plt.ylabel('loss')

plt.yscale('log')

plt.show()
test_data.shape
test_data['Image']=test_data['Image'].apply(lambda x: np.fromstring(x,sep=' '))

X_test=np.vstack(test_data['Image'].values)

X_test=X_test.reshape(-1,96,96,1)

X_test=X_test.astype(np.float32)

plt.imshow(X_test[5].reshape(96,96),cmap = 'gray')

plt.show()
def show_results(image_index):

    Ypred = model.predict(X_test[image_index:(image_index+1)])

    show_image(X_test[image_index], Ypred[0])
show_results(5)
lookid=pd.read_csv('/kaggle/input/facial-keypoints-detection/IdLookupTable.csv')

sample=pd.read_csv('/kaggle/input/facial-keypoints-detection/SampleSubmission.csv')

pred=model.predict(X_test)
sample.head()
lookid_list = list(lookid['FeatureName'])

imageID = list(lookid['ImageId']-1)

pre_list = list(pred)
rowid = lookid['RowId']

rowid=list(rowid)

feature = []

for f in list(lookid['FeatureName']):

    feature.append(lookid_list.index(f))
preded = []

for x,y in zip(imageID,feature):

    preded.append(pre_list[x][y])

rowid = pd.Series(rowid,name = 'RowId')

loc = pd.Series(preded,name = 'Location')

sub = pd.concat([rowid,loc],axis = 1)
sub.to_csv('pg_submission.csv',index = False)
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from IPython.display import clear_output 
from time import sleep 
import os 
train_data = pd.read_csv('../input/training/training.csv')
test_data = pd.read_csv('../input/test/test.csv')
lookid_data = pd.read_csv('../input/IdLookupTable.csv')
train_data.info()
train_data.isnull().any().value_counts()
train_data.fillna(method = 'ffill', inplace =True)
train_data.isnull().any().value_counts()
train_data.shape
train_data.tail(1)
len(train_data), len(test_data)
len(train_data['Image'][0])
len(test_data.Image[0].split(' '))
def append_X(X):
    image_list = []
    for i in range(len(X)):
        image_list.append(np.asarray(X.Image[i].split(' '),dtype = 'float').reshape(96,96))                        
    image = image_list
    return image
    
X_train = append_X(train_data) 
plt.imshow(X_train[0],cmap='gray')
plt.show()

X_test = append_X(test_data) 
plt.imshow(X_test[0],cmap='gray')
plt.show()

y = train_data.iloc[:, :-1].values
y[1,:]
print(len(X_train))


def keypoints_show(x, y=None):
    plt.imshow(x, 'gray')
    if y is not None:
        points = np.vstack(np.split(y, 15)).T
        plt.plot(points[0], points[1], 'o', color='red')
    plt.axis('off')   

    
sample_idx = np.random.choice(len(X_train))

y[sample_idx]
X_train[sample_idx]

keypoints_show(X_train[sample_idx], y[sample_idx])
y.shape
np.array(X_train).shape
X = np.stack(np.array(X_train))[...,None]
X_t = np.stack(np.array(X_test))[...,None]
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, MaxPool2D, Flatten, LeakyReLU, ELU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils.vis_utils import model_to_dot
model = Sequential()

model.add(Conv2D(filters = 256, kernel_size = (6,6), padding = 'Same', activation = 'relu', input_shape = (96,96,1)))
model.add(MaxPool2D(pool_size=(4,4), strides=(4,4)))

model.add(Conv2D(filters = 128, kernel_size = (4,4), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (4,4), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 32, kernel_size = (3,3), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters = 16, kernel_size = (2,2), padding = 'Same', activation = 'relu'))
model.add(MaxPool2D(pool_size=(1,1), strides=(1,1)))
model.add(Dropout(0.6))

model.add(Flatten())
model.add(Dense(256, activation = 'relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(30, activation = 'relu'))


model.compile(loss='mse', optimizer='adam', metrics = ['mae'])
model.fit(X,y,epochs = 50,batch_size = 128,validation_split = 0.2)
pred = model.predict(X_t)
lookid_list = list(lookid_data['FeatureName'])
imageID = list(lookid_data['ImageId']-1)
pre_list = list(pred)
rowid = lookid_data['RowId']
rowid=list(rowid)
feature = []
for f in list(lookid_data['FeatureName']):
    feature.append(lookid_list.index(f))
preded = []
for x,y in zip(imageID,feature):
    preded.append(pre_list[x][y])
rowid = pd.Series(rowid,name = 'RowId')
loc = pd.Series(preded,name = 'Location')
submission = pd.concat([rowid,loc],axis = 1)
submission.to_csv('face_key_detection_submission.csv',index = False)
y_t = model.predict(X_t)
np.array(X_test).shape
sample_idx = np.random.choice(len(X_test))
keypoints_show(X_test[sample_idx], y_t[sample_idx])
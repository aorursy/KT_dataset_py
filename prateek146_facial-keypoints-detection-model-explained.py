import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import clear_output

from tqdm import tqdm

import os

import keras


for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
Train_Dir = '/kaggle/input/facial-keypoints-detection/training.zip'

Test_Dir = '/kaggle/input/facial-keypoints-detection/test.zip'

lookid_dir = '/kaggle/input/facial-keypoints-detection/IdLookupTable.csv'

train_data = pd.read_csv(Train_Dir)  

test_data = pd.read_csv(Test_Dir)

lookid_data = pd.read_csv(lookid_dir)

os.listdir('../input')
train_data.info()
train_data.isnull().sum()
feature_8 = ['left_eye_center_x','left_eye_center_y','right_eye_center_x','right_eye_center_y','nose_tip_x','nose_tip_y','mouth_center_bottom_lip_x','mouth_center_bottom_lip_y','Image']



train_8 = train_data[feature_8].dropna().reset_index()

train_30=train_data.dropna().reset_index()
train_8.info()
train_30.info()
def str_to_int(train1):

    images = train1.Image.values

    del train1['Image']

    del train1['index']

    y=train1.values

    x = []

    for i in tqdm(images):

        q=[int(j) for j in i.split()]

        x.append(q)

    x=np.array(x)

    x=x.reshape(-1,96,96,1)

    x=x/255.0

    return([x,y])
X_train_8,Y_train_8 = str_to_int(train_8)

X_train_30,Y_train_30 = str_to_int(train_30)

print('X_train with 8 feature shape: ', X_train_8.shape)

print('y_train with 8 feature shape: ', Y_train_8.shape)

print('X_train with 30 feature shape: ', X_train_30.shape)

print('y_train with 30 feature shape: ', Y_train_30.shape)
from keras.models import Sequential

from keras.layers import Activation,Convolution2D,MaxPooling2D,BatchNormalization,Flatten,Dense,Dropout,Conv2D,MaxPool2D,ZeroPadding2D

from keras.layers.advanced_activations import LeakyReLU
def create_model(out=8):

    

    model = Sequential()



    # Input dimensions: (None, 96, 96, 1)

    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    # Input dimensions: (None, 96, 96, 32)

    model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    # Input dimensions: (None, 48, 48, 32)

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    # Input dimensions: (None, 48, 48, 64)

    model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    # Input dimensions: (None, 24, 24, 64)

    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    # Input dimensions: (None, 24, 24, 96)

    model.add(Convolution2D(96, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    # Input dimensions: (None, 12, 12, 96)

    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    # Input dimensions: (None, 12, 12, 128)

    model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    # Input dimensions: (None, 6, 6, 128)

    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    # Input dimensions: (None, 6, 6, 256)

    model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    model.add(MaxPool2D(pool_size=(2, 2)))



    # Input dimensions: (None, 3, 3, 256)

    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())

    # Input dimensions: (None, 3, 3, 512)

    model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))

    model.add(LeakyReLU(alpha = 0.1))

    model.add(BatchNormalization())



    # Input dimensions: (None, 3, 3, 512)

    model.add(Flatten())

    model.add(Dense(512,activation='relu'))

    model.add(Dropout(0.1))

    model.add(Dense(out))

    model.summary()

    

    model.compile(optimizer = 'adam' , loss = "mean_squared_error", metrics=["mae"])

    return model
#Prepare 2 models to handle 2 different datasets.

model_30 = create_model(out=30)

model_8 = create_model(out=8)
#Prepare callbacks

LR_callback = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, verbose=10, factor=.4, min_lr=.00001)

EarlyStop_callback = keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)
history = model_8.fit(X_train_8,Y_train_8,validation_split=.1,batch_size=64,epochs=50,callbacks=[LR_callback,EarlyStop_callback])
# Plot the loss and accuracy curves for training and validation

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['mae'], color='b', label="Training mae")

ax[1].plot(history.history['val_mae'], color='r',label="Validation mae")

legend = ax[1].legend(loc='best', shadow=True)
history = model_30.fit(X_train_30,Y_train_30,validation_split=.1,batch_size=64,epochs=50,callbacks=[LR_callback,EarlyStop_callback])

# Plot the loss and accuracy curves for training and validation

fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])

legend = ax[0].legend(loc='best', shadow=True)



ax[1].plot(history.history['mae'], color='b', label="Training mae")

ax[1].plot(history.history['val_mae'], color='r',label="Validation mae")

legend = ax[1].legend(loc='best', shadow=True)
images = test_data.Image.values

x = []

for i in tqdm(images):

    q=[int(j) for j in i.split()]

    x.append(q)

x=np.array(x)

x=x.reshape(-1,96,96,1)

x=x/255.0
#Pridect points for each image using 2 different model.

y_hat_30 = model_30.predict(x) 

y_hat_8 = model_8.predict(x)

print('Predictions shape', y_hat_30.shape)

print('Predictions shape', y_hat_8.shape)
feature_8_index=[0,1,2,3,20,21,28,29]

for i in range(8):

    y_hat_30[:,feature_8_index[i]] = y_hat_8[:,i]
def plot_face_pts(img, pts):

    plt.imshow(img[:,:,0], cmap='gray')

    for i in range(1,31,2):

        plt.plot(pts[i-1], pts[i], 'b.')
#Display samples of the dataset.

fig = plt.figure(figsize=(10, 7))

fig.subplots_adjust(

    left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i, f in enumerate(range(34,40)):

    ax = fig.add_subplot(2, 3, i + 1, xticks=[], yticks=[])

    plot_face_pts(x[f], y_hat_30[f])



plt.show()
#All required features in order.

required_features = list(lookid_data['FeatureName'])

#All images nmber in order.

imageID = list(lookid_data['ImageId']-1)

#Generate Directory to map feature name 'Str' into int from 0 to 29.

feature_to_num = dict(zip(required_features[0:30], range(30)))
#Generate list of required features encoded into ints.

feature_ind = []

for f in required_features:

    feature_ind.append(feature_to_num[f])
#Pick only the required predictions from y_hat_30 (filteration).

required_pred = []

for x,y in zip(imageID,feature_ind):

    required_pred.append(y_hat_30[x, y])
#Submit

rowid = lookid_data['RowId']

loc30 = pd.Series(required_pred,name = 'Location')

submission = pd.concat([rowid,loc30],axis = 1)

submission.to_csv('Predictions.csv',index = False)
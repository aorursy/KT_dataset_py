print("Contents of input/facial-keypoints-detection directory: ")
!ls ../input/facial-keypoints-detection/

print("\nExtracting .zip dataset files to working directory ...")
!unzip -u ../input/facial-keypoints-detection/test.zip
!unzip -u ../input/facial-keypoints-detection/training.zip

print("\nCurrent working directory:")
!pwd
print("\nContents of working directory:")
!ls
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output
from time import sleep
import os
Train_Dir = 'training.csv'
Test_Dir = 'test.csv'
lookid_dir = '../input/facial-keypoints-detection/IdLookupTable.csv'
train_data = pd.read_csv(Train_Dir)  
test_data = pd.read_csv(Test_Dir)
lookid_data = pd.read_csv(lookid_dir)
train_data.head().T
train_data.shape
train_data.describe()
train_data.info()
all_data_na = (train_data.isnull().sum() / len(train_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:30]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(20)
train_data.isnull().any()
null_counts = train_data.isnull().sum()
null_counts[null_counts > 0].sort_values(ascending=False)
# Set the limit
# Drop columns using that limit
# limit = len(train_data) * 0.7
# new=train_data.dropna(axis=1, thresh=limit)
# View columns in the dataset
# new.columns
type(train_data)
#train_data.fillna(method = 'ffill',inplace = True)
train_data = train_data.fillna(pd.concat([train_data.ffill(), train_data.bfill()]).groupby(level=0).mean())
train_data.isnull().any()
train_data.isnull().any().value_counts()
train_data[['Image']].describe()

imag = []
for i in range(0,7049):  
    img = train_data['Image'][i].split(' ')
    img = ['0' if x == '' else x for x in img]
    imag.append(img)
    
    

timag = []
for i in range(0,1783):
    timg = test_data['Image'][i].split(' ')
    timg = ['0' if x == '' else x for x in timg]
    
    timag.append(timg)
image_list = np.array(imag,dtype = 'float')
X_train = image_list.reshape(-1,96,96,1)
timage_list = np.array(timag,dtype = 'float')
X_test = timage_list.reshape(-1,96,96,1) 
plt.imshow(X_train[0].reshape(96,96),cmap='gray')
plt.show()
training = train_data.drop('Image',axis = 1)

y_train = []
for i in range(0,7049):
    y = training.iloc[i,:]

    y_train.append(y)
y_train = np.array(y_train,dtype = 'float')
from keras.layers.advanced_activations import LeakyReLU
# from keras.layers import Activation
from keras.models import Sequential, Model
from keras.layers import Activation, Convolution2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout, Conv2D,MaxPool2D, ZeroPadding2D
training.shape
'''

model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', activation = 'selu', use_bias=False, input_shape=(96,96,1)))

model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', activation = 'selu',use_bias=False))

model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same',activation = 'selu', use_bias=False))

model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same',activation = 'selu', use_bias=False))

model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', activation = 'selu',use_bias=False))
# model.add(BatchNormalization())

model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', activation = 'selu',use_bias=False))

model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',activation = 'selu',use_bias=False))

model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',activation = 'selu',use_bias=False))

model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', activation = 'selu',use_bias=False))

model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same',activation = 'selu', use_bias=False))

model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
model.summary()

'''
model = Sequential()

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False, input_shape=(96,96,1)))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(32, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(64, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
# model.add(BatchNormalization())
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(128, (3,3),padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(256, (3,3),padding='same',use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())

model.add(Convolution2D(512, (3,3), padding='same', use_bias=False))
model.add(LeakyReLU(alpha = 0.1))
model.add(BatchNormalization())


model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(30))
# I didn't use softmax here, because it's only used in logistic regression, 
# where the outputs are usually categorical. But here our outputs are a bunch of floating
# values, which are not categorical
model.summary()
import keras
model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['accuracy'])

'''
# Compile with adam optimizer and cross-entropy loss

model.compile(optimizer='adam', 
              loss='mean_squared_error',
              metrics=['accuracy'])
              
# using adam  
model.compile(optimizer="adam", 
              loss="categorical_crossentropy", 
              metrics=["accuracy"])

# Use sgd optimizer
model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=["accuracy"])

# Use RMSprop
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

# Compile the model
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

'''

log = model.fit(X_train,y_train,epochs = 50,batch_size = 512,validation_split = 0.2)
pred = model.predict(X_test)
# Plotting loss and accuracy curves for training and verification
fig, ax = plt.subplots(2,1)

# accuracy
ax[0].plot(log.history['accuracy'], color='b', label="Training accuracy")
ax[0].plot(log.history['val_accuracy'], color='r',label="Validation accuracy")
legend = ax[0].legend(loc='best', shadow=True)

# loss
ax[1].plot(log.history['loss'], color='b', label="Training loss")
ax[1].plot(log.history['val_loss'], color='r', label="validation loss",axes =ax[1])
legend = ax[1].legend(loc='best', shadow=True)
fig.show()
plt.savefig('FacialKeypointsDetection.png')
# RowID
rowid=list(lookid_data['RowId'])

imageID = list(lookid_data['ImageId']-1)
feature_name = list(lookid_data['FeatureName'])
pre_list = list(pred)
feature = []
for f in feature_name:
    feature.append(feature_name.index(f))
# Location
# predict using image and feature
location = []
for x,y in zip(imageID,feature):
    location.append(pre_list[x][y])
rowid = pd.Series(rowid,name = 'RowId')
loc = pd.Series(location,name = 'Location')
loc = loc.clip(0.0,96.0)
submission = pd.concat([rowid,loc],axis = 1)


submission.to_csv('FacialKeypointsDetection.csv',index = False)
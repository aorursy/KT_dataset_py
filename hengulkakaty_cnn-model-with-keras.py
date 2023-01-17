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
import cv2
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.utils import to_categorical
import matplotlib.pyplot as plt 
from random import shuffle
np.set_printoptions(suppress=True)
#np.set_printoptions(threshold=np.inf)
IMG_SIZE = 120
X_Train_orig = []
Y_Train_orig = []
for i in os.listdir('../input/train/'):
    label = i.split('.')[-3]
    if label == 'cat':
        label = 0
    elif label == 'dog':
        label = 1
    img = cv2.imread('../input/train/'+i, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_CUBIC)
    #img = img / 255
    X_Train_orig.append([np.array(img), np.array(label)])

np.save('Training_Data.npy', X_Train_orig)
shuffle(X_Train_orig)
X = np.array([i[0] for i in X_Train_orig]).reshape(-1,IMG_SIZE,IMG_SIZE,3)
Y = np.array([i[1] for i in X_Train_orig])
from sklearn.model_selection import train_test_split
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size = 0.1)
print('Shape of X_train is :', X_train.shape)
print('Shape of Y_train is :', Y_train.shape)
print('Shape of X_val is :', X_val.shape)
print('Shape of Y_val is :', Y_val.shape)
import matplotlib.pyplot as plt 
plt.figure(figsize=(20,20))   # to fix a shape for each image print
for i in range(50):          # using a for loop to display a number of images
    plt.subplot(5, 10, i+1) # we need to use this function to print an array of pictures 
    plt.imshow(X_val[i,:,:,:]) # this will call the images from train set one by one
    plt.title('DOG' if Y_val[i] == 1 else 'CAT')  # Lets also look into the labels 
    plt.axis('off') 
def Keras_Model(input_shape):    
    
    X_input = Input(input_shape)
    
    # First Layer
    X = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv0')(X_input) 
    X = BatchNormalization(axis = 3, name = 'bn0')(X)
    X = Activation('relu')(X) 
    
    X = Conv2D(64, (3, 3), strides = (1, 1), padding = 'same', name = 'conv1')(X) 
    X = BatchNormalization(axis = 3, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3, 3), name='max_pool_0')(X)
    X = Dropout(0.3)(X)
    
    # Second Layer
    X = Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', name = 'conv3')(X) 
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', name = 'conv4')(X) 
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(128, (3, 3), strides = (1, 1), padding = 'same', name = 'conv5')(X) 
    X = BatchNormalization(axis = 3, name = 'bn5')(X)
    X = Activation('relu')(X)
     
    X = MaxPooling2D((3, 3), name='max_pool_1')(X)
    X = Dropout(0.3)(X)
    
    # Fourth Convolutional Layer
    X = Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', name = 'conv6')(X) 
    X = BatchNormalization(axis = 3, name = 'bn6')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', name = 'conv7')(X) 
    X = BatchNormalization(axis = 3, name = 'bn7')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(256, (3, 3), strides = (1, 1), padding = 'same', name = 'conv8')(X) 
    X = BatchNormalization(axis = 3, name = 'bn8')(X)
    X = Activation('relu')(X)

 
    X = MaxPooling2D((3, 3), name='max_pool_2')(X)
    X = Dropout(0.3)(X)
    
    X = Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv10')(X) 
    X = BatchNormalization(axis = 3, name = 'bn10')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv11')(X) 
    X = BatchNormalization(axis = 3, name = 'bn11')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(512, (3, 3), strides = (1, 1), padding = 'same', name = 'conv12')(X) 
    X = BatchNormalization(axis = 3, name = 'bn12')(X)
    X = Activation('relu')(X)

    
    X = MaxPooling2D((3, 3), name='max_pool_3')(X)
    X = Dropout(0.3)(X)
    
    # Flatten the data.
    X = Flatten()(X)
    # Dense Layer
    X = Dense(4096, activation='relu', name='fc1')(X)
    X = Dropout(0.5)(X)
    X = Dense(1024, activation='relu', name='fc2')(X)
    X = Dropout(0.5)(X)
    X = Dense(256, activation='relu', name='fc3')(X)
    # Using softmax function to get the output
    X = Dense(1, activation='sigmoid', name='fc4')(X)
    
    model = Model(inputs = X_input, outputs = X, name='model')
    
    return model
Keras_Model = Keras_Model(X_train.shape[1:4])
from keras.optimizers import Adam
epochs = 100
batch_size = 64
lrate = 0.001
decay = lrate/epochs
optimizer = Adam(lr=lrate, epsilon=1e-08, decay = decay)
Keras_Model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
from keras.callbacks import ReduceLROnPlateau
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=1, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.0000001)
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto')
history = Keras_Model.fit(x = X_train, y = Y_train, batch_size = batch_size, 
                        epochs=epochs, verbose=1, 
                        validation_data = (X_val, Y_val),
                          shuffle = True, 
                          steps_per_epoch= None, validation_steps=None,
                                  callbacks=[learning_rate_reduction, early_stopping] )
preds = Keras_Model.evaluate(X_train, Y_train)
print ("Loss = " + str(preds[0]))
print ("Train set Accuracy = " + str(preds[1]))
preds_val = Keras_Model.evaluate(X_val, Y_val)
print ("Loss = " + str(preds_val[0]))
print ("Validation Set Accuracy = " + str(preds_val[1]))
history_dict = history.history
history_dict.keys()
val_loss = history_dict['val_loss']
val_acc = history_dict['val_acc']
loss = history_dict['loss']
acc = history_dict['acc']
epochs = range(1,len(history_dict['val_loss'])+1)
plt.plot(epochs,acc,'b-')
plt.title('Accuracy of Model')
plt.xlabel('epochs')
plt.ylabel('Accuracy')

plt.plot(epochs,val_acc,'b-', color = 'red')
plt.title('Accuracy of Model')
plt.xlabel('epochs')
plt.ylabel('Accuracy')
plt.show()
plt.plot(epochs,loss,'b-')
plt.title('loss function')
plt.xlabel('epochs')
plt.ylabel('Loss')

plt.plot(epochs,val_loss,'b-', color = 'red')
plt.title('loss function')
plt.xlabel('epochs')
plt.ylabel('val_loss')
plt.show()
predicted_val_probability = Keras_Model.predict(X_val, batch_size=64)
Y_val_pred_label = np.round(predicted_val_probability)
l = []
for i in range(len(Y_val_pred_label)):
    if Y_val[i] != Y_val_pred_label[i]:
        l.append(i)
m = []
for t in l:
    if predicted_val_probability[t] >= 0.99:
        m.append(t)
    elif predicted_val_probability[t] < 0.01:
        m.append(t)
    
len(m)
plt.figure(figsize=(35,35))
c = 1
for i in m[:]:
    plt.subplot(10,5, c)
    plt.imshow(X_val[i])
    plt.title('DOG:{}\nTrue label:{}'.format(predicted_val_probability[i], Y_val[i])
              if predicted_val_probability[i]>= 0.5 else 'CAT:{}\nTrue label:{}'.format(predicted_val_probability[i],Y_val[i]))
    plt.axis('off')
    c = c+1
X_Test_orig = []
for i in os.listdir('../input/test/'):
    label = i.split('.')[-2]
    img = cv2.imread('../input/test/'+i, cv2.IMREAD_COLOR)
    img = cv2.resize(img,(IMG_SIZE,IMG_SIZE), interpolation = cv2.INTER_CUBIC)
    #img = img / 255
    X_Test_orig.append([np.array(img), np.array(label)])

np.save('Test_Data.npy', X_Train_orig)
X_test = np.array([i[0] for i in X_Test_orig]).reshape(-1,IMG_SIZE, IMG_SIZE, 3)
Label = np.array([i[1] for i in X_Test_orig])
X_test = X_test / 255
classes = Keras_Model.predict(X_test, batch_size = batch_size)
prediction = pd.DataFrame()
prediction['id'] = Label
prediction['label'] = classes

prediction.to_csv('submission.csv', index = False)
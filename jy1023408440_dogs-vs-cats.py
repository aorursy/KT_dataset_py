# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import os, cv2, re, random
import numpy as np
import pandas as pd
from keras import backend as K


from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.optimizers import *
from keras import layers, models, optimizers
import h5py
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.
TRAIN_DIR = '../input/train/'
TEST_DIR = '../input/test/'

CHANNELS = 3
nb_classes=2
img_width=299
img_height=299

train_images = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
train_dogs =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'dog' in i]
train_cats =   [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR) if 'cat' in i]

test_images =  [TEST_DIR+i for i in os.listdir(TEST_DIR)]
#train_images_dogs_cats = train_dogs[0:50] + train_cats[0:50] 
#train_images_dogs_cats = train_images_dogs_cats[0:20]+train_images_dogs_cats[24980:25000]
train_images_dogs_cats=train_images
def read_image(file_path):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    #b,g,r = cv2.split(img)
    #img2 = cv2.merge([r,g,b])
    return cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
from tqdm import tqdm  
def prep_data(images):
 
    x = np.zeros((len(images), img_width, img_height, 3), dtype=np.uint8)
    y = np.zeros(len(images), dtype=np.uint8)
    count = len(images)
        
    for i, image_file in tqdm(enumerate(images)):
        x[i] = cv2.resize(cv2.imread(image_file), (img_width,img_height), interpolation=cv2.INTER_CUBIC)
        if 'dog' in image_file:
            y[i]=1
        elif 'cat' in image_file:
            y[i]=0
        
    return x, y
from keras.applications import *
import tensorflow as tf
from sklearn.utils import shuffle
import keras
from sklearn.model_selection import KFold,StratifiedKFold
from keras.optimizers import *
from keras.callbacks import EarlyStopping
#print(K.image_data_format())
#X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=1)

#X_train, Y_train = shuffle(X_train, Y_train)
#X_test, Y_test = prepare_data(test_images_dogs_cats)
X, Y = prep_data(train_images_dogs_cats)
#X, Y = shuffle(X, Y)
#X=np.array(X, dtype='uint8')
#Y=np.array(Y, dtype='uint8')
sfolder = StratifiedKFold(n_splits=5,random_state=0,shuffle=False)

nb_epoch = 10
batch_size = 16

## Callback for loss logging per epoch
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        
    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))   
from keras.utils import multi_gpu_model
from keras.applications.inception_resnet_v2 import *
input_x = Input((img_width, img_height, 3))
input_x = Lambda(preprocess_input)(input_x)
base_model = InceptionResNetV2(input_tensor=input_x,weights='imagenet', include_top=False)
# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

#for layer in base_model.layers:
#    layer.trainable = False
early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')   

keras.optimizers.Adadelta(lr=0.001, rho=0.95, epsilon=1e-06)
#model = multi_gpu_model(model2, gpus=4)
model.compile(optimizer='adadelta',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = LossHistory()

#historylist = []
for train_index, test_index in sfolder.split(X,Y):
    #print("TRAIN:", train_index, "valid:",test_index)
    print('----------------------------------------------------')
    X_train, X_val = X[train_index],X[test_index]
    Y_train, Y_val = Y[train_index],Y[test_index]
    
    #validdata=np.concatenate((np.array(X_val),np.array(Y_val)), axis=0)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              validation_data=(X_val,Y_val), verbose=1, shuffle=True, callbacks=[history, early_stopping])
    del X_train,Y_train,X_val,Y_val
    #historylist.append(history)
import matplotlib.pyplot as plt
loss = history.losses
val_loss = history.val_losses


plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('InceptionResNetV2 Loss Trend')
plt.plot(loss, 'blue', label='Training Loss')
plt.plot(val_loss, 'green', label='Validation Loss')
plt.xticks(range(0,nb_epoch)[0::1])
plt.legend()
plt.show()
#model.save('model_inception_resnet_v2_keras.h5')
def pre_test():
    x = np.zeros((len(test_images), img_width, img_height, 3), dtype=np.uint8)
    for i, image_file in enumerate(test_images):
        x[i] = cv2.resize(cv2.imread(image_file), (img_width,img_height), interpolation=cv2.INTER_CUBIC)
    return x
#test_images_dogs_cats = test_images[0:1000] 
#test_images_dogs_cats = test_images_dogs_cats[0:20] 
from tqdm import tqdm 
import pandas as pd
df = pd.read_csv("sample_submission.csv")

testdata=pre_test()
model_out = model.predict(testdata)
model_out=model_out.clip(min=0.005, max=0.995)

for i, fname in enumerate(test_images):
    index = int(fname[fname.rfind('/')+1:fname.rfind('.')])
    df.set_value(index-1, 'label', model_out[i])

df.to_csv('pred0507.csv', index=None)
df.head(10)

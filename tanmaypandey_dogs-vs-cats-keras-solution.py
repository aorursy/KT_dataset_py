import os, cv2, re, random
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split

TRAIN_DIR = '../input/dogs-vs-cats-redux-kernels-edition/train/'
TEST_DIR = '../input/dogs-vs-cats-redux-kernels-edition/test/'
train_images_dogs_cats = [TRAIN_DIR+i for i in os.listdir(TRAIN_DIR)] # use this for full dataset
test_images_dogs_cats = [TEST_DIR+i for i in os.listdir(TEST_DIR)]
NO_EPOCHS=10
RESNET_WEIGHTS_PATH = '../input/keras-pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
len(train_images_dogs_cats)
len(train_images_dogs_cats)
train_images_dogs_cats
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
def prepare_data(list_of_images):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """
    x = [] # images as arrays
    y=[]
    
    for image in tqdm(list_of_images):
        x.append(cv2.resize(cv2.imread(image), (224,224), interpolation=cv2.INTER_CUBIC))
        z=(re.split('\d+',image)[0][-4:-1])
        if 'cat' in z:
            y.append(0)
        else:
            y.append(1)

                
    
    return x,y
train_images_dogs_cats[0]
X ,Y= prepare_data(train_images_dogs_cats)
print(K.image_data_format())
X[0]
Y[0]
len(X)
print(type(X),type(Y))
X = np.array(X)
Y = np.array(Y)
X.shape
np.unique(Y,return_counts=True)
print(len(X),len(Y))
Y
Y
from keras.utils import to_categorical
Y1 = to_categorical(Y)
Y[:10]
Y1[:10]
# First split the data in two sets, 80% for training, 20% for Val/Test)
X_train, X_val, Y_train, Y_val = train_test_split(X,Y1, test_size=0.2, random_state=7)
len(X_val)
nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 64
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.layers import Dense,Dropout
model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights=RESNET_WEIGHTS_PATH))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
# ResNet-50 model is already trained, should not be trained
model.layers[0].trainable = True
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
print(X.shape,X_train.shape,X_val.shape)
type(Y_val)
print(np.unique(Y_train,return_counts=True),np.unique(Y_val,return_counts=True))
train_model = model.fit(X_train,Y_train ,
    batch_size=64,
    epochs=10,
    verbose=1,
    validation_data=(X_val,Y_val))
import matplotlib.pyplot as plt
hist=train_model.history
acc=hist['acc']
val_acc=hist['val_acc']
epoch=range(len(acc))
loss=hist['loss']
val_loss=hist['val_loss']
f,ax=plt.subplots(1,2,figsize=(16,8))
ax[0].plot(epoch,acc,'b',label='Training Accuracy')
ax[0].plot(epoch,val_acc,'r',label='Validation Accuracy')
ax[0].legend()
ax[1].plot(epoch,loss,'b',label='Training Loss')
ax[1].plot(epoch,val_loss,'r',label='Training Loss')
ax[1].legend()
plt.show()
import keras
model.save_weights('model_weights.h5')
model.save('model_keras.h5')
X,_=prepare_data(test_images_dogs_cats)
X = np.array(X)
y_test=model.predict(X,verbose=1)
import matplotlib.pyplot as plt
test_images_dogs_cats[0]
f,ax=plt.subplots(1,5,figsize=(10,5))
i=0
for x in test_images_dogs_cats[:5]:
    print(ax[i].imshow(cv2.imread(x))) 
    i+=1
y_test[:,1]
y_final=y_test[:,1]
# y_final=[0 if x[0]>x[1] else 1 for x in y_test ]

# y_final[:5]
len(test_images_dogs_cats)
len(X)
df_test=pd.DataFrame({'id':range(1,len(X)+1),'label':y_final})
df_test.to_csv('solution1.csv',index=False)

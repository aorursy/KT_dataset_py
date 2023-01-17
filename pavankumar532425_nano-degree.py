# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#model selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score

#preprocess.
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
import random as rn
import tensorflow as tf
# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle as Dr
from PIL import Image
print(os.listdir('../input/vgg16'))
print(os.listdir('../input/resnet50'))
X=[]
Y=[]
IMG_SIZE=150
FLOWER_DAISY_DIR='../input/flowers-recognition/flowers/flowers/daisy'
FLOWER_SUNFLOWER_DIR='../input/flowers-recognition/flowers/flowers/sunflower'
FLOWER_TULIP_DIR='../input/flowers-recognition/flowers/flowers/tulip'
FLOWER_DANDI_DIR='../input/flowers-recognition/flowers/flowers/dandelion'
FLOWER_ROSE_DIR='../input/flowers-recognition/flowers/flowers/rose'
categories = ['Daisy','Dandelion','Rose','Sunflower','Tulip']
print("labels of images are " ,categories )
def assign_label(flower_type):
    return flower_type
 
def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        try:
            label=assign_label(flower_type)
            path = os.path.join(DIR,img)
            img = cv2.imread(path,cv2.IMREAD_COLOR)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            X.append(np.array(img))
            Y.append(str(label))
        except:
            print("")
make_train_data('Daisy',FLOWER_DAISY_DIR)
print(len(X))
make_train_data('Dandelion',FLOWER_DANDI_DIR)
print(len(X))
make_train_data('Rose',FLOWER_ROSE_DIR)
print(len(X))
make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
print(len(X))
make_train_data('Tulip',FLOWER_TULIP_DIR)
print(len(X))
def find_categories_count(Y):
    categories_count={
    'Daisy':0,
    'Dandelion':0,
    'Rose':0,
    'Sunflower':0,
    'Tulip':0
    }
    for i in Y:
         categories_count[i]=categories_count[i]+1
    return categories_count
def find_class_label(data):
    match={
        0:'Daisy',
        1:'Dandelion',
        2:'Rose',
        3:'Sunflower',
        4:'Tulip'
        }
    return match[data]
print(find_categories_count(Y))
height=[]
categories_count=find_categories_count(Y)
for i in categories_count:
    height.append(categories_count[i])
plt.bar([x for x in categories],height)
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Y))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Y[l])
        
plt.tight_layout()
def covert_categories_to_numeric(data):
    y=[]
    match={
        'Daisy':0,
        'Dandelion':1,
        'Rose':2,
        'Sunflower':3,
        'Tulip':4
    }
    for i in  data:
        y.append(match[i])
    return np.array(y)
Y=covert_categories_to_numeric(Y)
print(Y)
Y=to_categorical(Y,5)
X=np.array(X)
X=X/255
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.1,random_state=48)
x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,test_size=0.1,random_state=48)
# printing statistics about the dataset
print('There are %d total image categories.' % len(categories))
print('There are %s total flowers images.\n' % X.shape[0])
print('There are %d training flowers images.' % x_train.shape[0])
print('There are %d validation flowers images.' % x_valid.shape[0])
print('There are %d testing flowers images.'% x_test.shape[0])
plt.bar(['total_data','training_data','validation_data','testing_data'],[X.shape[0],x_train.shape[0],x_valid.shape[0],x_test.shape[0]])
height=[]
data=[]
for i in np.argmax(y_train,axis=1):
    data.append(find_class_label(i))
categories_count=find_categories_count(data)
for i in categories_count:
    height.append(categories_count[i])
plt.bar([x for x in categories],height)
y_train.shape[0]
height=[]
data=[]
for i in np.argmax(y_valid,axis=1):
    data.append(find_class_label(i))
categories_count=find_categories_count(data)
for i in categories_count:
    height.append(categories_count[i])
plt.bar([x for x in categories],height)
y_valid.shape[0]
height=[]
data=[]
for i in np.argmax(y_test,axis=1):
    data.append(find_class_label(i))
categories_count=find_categories_count(data)
for i in categories_count:
    height.append(categories_count[i])
plt.bar([x for x in categories],height)
y_test.shape[0]
np.random.seed(42)
rn.seed(42)
tf.set_random_seed(42)
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
datagen.fit(x_train)
benchmark_checkpoints = ModelCheckpoint(filepath='weights.best.from_bench.hdf5', 
                               verbose=1, save_best_only=True)
bench_model = Sequential()
bench_model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', 
                        input_shape=(150,150, 3)))
bench_model.add(MaxPooling2D(pool_size=2))
bench_model.add(Flatten())
bench_model.add(Dense(5, activation='softmax'))
bench_model.summary()
bench_model.compile(loss='categorical_crossentropy', 
                    optimizer=SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True), 
                  metrics=['accuracy'])
bench_results = bench_model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=10,validation_data=(x_valid,y_valid), callbacks=[benchmark_checkpoints], verbose=1)
bench_pred=np.argmax(bench_model.predict(x_test) ,axis=1)
plt.plot(bench_results.history['loss'])
plt.plot(bench_results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()
plt.plot(bench_results.history['acc'])
plt.plot(bench_results.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()
mis_classification_count=0
for i in range(0,len(bench_pred)):
    if bench_pred[i]!=np.argmax(y_test[i]):
        mis_classification_count+=1
        
print(mis_classification_count)
test_accuracy = 100*(y_test.shape[0]-mis_classification_count)/len(bench_pred)
print('Test accuracy: %.4f%%' % test_accuracy)
from sklearn.metrics import classification_report
print(classification_report( np.argmax(y_test ,axis=1),bench_pred ,target_names =categories ))
resnet50_checkpoints = ModelCheckpoint(filepath='weights.best.from_resnet50.hdf5', 
                               verbose=1, save_best_only=True)
import keras
from keras.applications.resnet50 import ResNet50
model = Sequential()
model.add(ResNet50(include_top = False, weights = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape = (150,150,3)))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001,  momentum=0.9, nesterov=True),
              metrics=['accuracy'])
model.summary()
resnet50_results = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=20,validation_data=(x_valid,y_valid),callbacks=[resnet50_checkpoints ], verbose=1)
plt.plot(resnet50_results.history['loss'])
plt.plot(resnet50_results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()
plt.plot(resnet50_results.history['acc'])
plt.plot(resnet50_results.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()
model.load_weights('weights.best.from_resnet50.hdf5')
resnet50_pred=np.argmax(model.predict(x_test),axis=1)
mis_classification_count=0
mis_classification_index=[]
for i in range(0,len(resnet50_pred)):
    if resnet50_pred[i]!=np.argmax(y_test[i]):
        mis_classification_count+=1
        mis_classification_index.append(i)
print(mis_classification_count)      
test_accuracy = 100*(y_test.shape[0]-mis_classification_count)/len(bench_pred)
print('Test accuracy: %.4f%%' % test_accuracy)
from sklearn.metrics import classification_report
print(classification_report(np.argmax(y_test ,axis=1), resnet50_pred ,target_names =categories ))
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l= rn.randint(0,len(y_test))
        ax[i,j].imshow(x_test[l])
        ax[i,j].set_title('predicted_label: '+find_class_label(resnet50_pred[l])+'/original_label: '+find_class_label(np.argmax(y_test[l])))
        
plt.tight_layout()
VGG16_checkpoints = ModelCheckpoint(filepath='weights.best.from_VGG16.hdf5', 
                               verbose=1, save_best_only=True)
import keras
from keras.applications.vgg16 import VGG16 
model = Sequential()
model.add(VGG16(include_top = False, weights = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', input_shape = (150,150,3)))
model.add(Flatten())
model.add(Dense(512, activation = 'relu'))
model.add(Dense(5, activation = 'softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.001,  momentum=0.9, nesterov=True),
              metrics=['accuracy'])
model.summary()
VGG16_results = model.fit_generator(datagen.flow(x_train, y_train, batch_size=32),
                    steps_per_epoch=len(x_train) / 32, epochs=20,validation_data=(x_valid,y_valid),callbacks=[VGG16_checkpoints])
plt.plot(VGG16_results.history['loss'])
plt.plot(VGG16_results.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()
plt.plot(VGG16_results.history['acc'])
plt.plot(VGG16_results.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()
model.load_weights('weights.best.from_VGG16.hdf5')
VGG16_pred=np.argmax(model.predict(x_test),axis=1)
mis_classification_count=0
mis_classification_index=[]
for i in range(0,len(VGG16_pred)):
    if VGG16_pred[i]!=np.argmax(y_test[i]):
        mis_classification_count+=1
        mis_classification_index.append(i)
print(mis_classification_count)      
test_accuracy = 100*(y_test.shape[0]-mis_classification_count)/len(VGG16_pred)
print('Test accuracy: %.4f%%' % test_accuracy)
from sklearn.metrics import classification_report
print(classification_report(np.argmax(y_test ,axis=1), VGG16_pred ,target_names =categories ))
fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l= rn.randint(0,len(y_test))
        ax[i,j].imshow(x_test[l])
        ax[i,j].set_title('predicted_label: '+find_class_label(VGG16_pred[l])+'/original_label: '+find_class_label(np.argmax(y_test[l])))
        
plt.tight_layout()


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
 
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
%matplotlib inline  
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from sklearn.preprocessing import LabelBinarizer
from keras.preprocessing.image import ImageDataGenerator

#dl libraraies
import keras
from keras.models import Sequential, Model
from keras import backend as K
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical
from sklearn.metrics import classification_report,confusion_matrix
from keras.applications import VGG19
from keras.applications.vgg19 import VGG19, decode_predictions, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import ReduceLROnPlateau

import matplotlib.image as mpimg
from keras import backend as K
import matplotlib.pyplot as plt

# specifically for cnn
from keras.layers import Dense, MaxPool2D
from keras.layers import Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
 
import tensorflow as tf
import random as rn

import cv2
import os
import random
import tensorflow as tf

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2                  
import numpy as np  
from tqdm import tqdm
import os                   
from random import shuffle  
from zipfile import ZipFile
from PIL import Image
import os
print(os.listdir('../input/flowers-recognition/flowers'))
data_dir = "/kaggle/input/flowers-recognition/flowers"
img_size = 150
images_array=[]
def make_train_data(label, label_index, data_dir):
    path = os.path.join(data_dir, label)
    for img in tqdm(os.listdir(path)):
        try:
            img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
            img_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
            images_array.append([img_arr, label_index])
            data.append(np.array(img_arr))
        except Exception as e:
            print(e)
data=[]
make_train_data('daisy', 0, data_dir)
print("Daisy Images: " , len(data))
print("Total Images: " , len(images_array))
data=[]
make_train_data('sunflower', 1, data_dir)
print("Sunflower Images: " , len(data))
print("Total Images: " , len(images_array))
data=[]
make_train_data('tulip', 2, data_dir)
print("Tulip Images: " , len(data))
print("Total Images: " , len(images_array))
data=[]
make_train_data('dandelion', 3, data_dir)
print("Dandelion Images: " , len(data))
print("Total Images: " , len(images_array))
data=[]
make_train_data('rose', 4, data_dir)
print("Rose Images: " , len(data))
print("Total Images: " , len(images_array))
flowers_labels = ['daisy','sunflower', 'tulip', 'dandelion', 'rose']
l = []
for i in images_array:
    l.append(flowers_labels[i[1]])
sns.set_style('darkgrid')
sns.countplot(l)
fig,ax=plt.subplots(5,5)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (5):
        l=rn.randint(0,len(images_array))
        ax[i,j].imshow(images_array[l][0])
        ax[i,j].set_title('Flower: '+flowers_labels[images_array[l][1]])
        
plt.tight_layout()
X = []
Y = []

for feature, label in images_array:
    X.append(feature)
    Y.append(label)
X=np.array(X)
X=X/255
# Reshaping the data from 1-D to 3-D as required through input by CNN's 
X = X.reshape(-1, img_size, img_size, 3)
Y = np.array(Y)
label_binarizer = LabelBinarizer()
Y=label_binarizer.fit_transform(Y)
x_train,x_test,y_train,y_test = train_test_split(X , Y , test_size = 0.2 , random_state = 0)
# With data augmentation to prevent overfitting and handling the imbalance in dataset

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = False,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)
VGG19_model = VGG19(input_shape=(150,150,3), include_top=False, weights="imagenet")
# setting the VGG model to be untrainable.
VGG19_model.trainable = False
VGG19_model.summary()
model = Sequential()
model.add(VGG19_model)
model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.summary()
epochs=50
batch_size=128
red_lr=ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=0.0001, patience=2, verbose=1)
model.compile(optimizer = Adam(lr=1e-4) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
history = model.fit(datagen.flow(x_train,y_train, batch_size = batch_size) , epochs = epochs , 
                    validation_data = (x_test, y_test), verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0] , "%")
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
fig , ax = plt.subplots(1,2)
fig.set_size_inches(20,10)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend(['train', 'test'])

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
ax[1].legend(['train', 'test'])

plt.show()
for i in range (len(VGG19_model.layers)):
    print (i,VGG19_model.layers[i])
  
for layer in VGG19_model.layers[15:]:
    layer.trainable=True
for layer in VGG19_model.layers[0:15]:
    layer.trainable=False
model.compile(optimizer = Adam(lr=1e-4) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
history = model.fit(datagen.flow(x_train,y_train, batch_size = batch_size) , epochs = epochs , validation_data = (x_test, y_test))
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0] , "%")
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
fig , ax = plt.subplots(1,2)
fig.set_size_inches(20,10)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend(['train', 'test'])

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
ax[1].legend(['train', 'test'])

plt.show()
for i in range (len(VGG19_model.layers)):
    print (i,VGG19_model.layers[i])
  
for layer in VGG19_model.layers[11:]:
    layer.trainable=True
for layer in VGG19_model.layers[0:11]:
    layer.trainable=False
model.compile(optimizer = Adam(lr=1e-4) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
history = model.fit(datagen.flow(x_train,y_train, batch_size = batch_size) , epochs = epochs , validation_data = (x_test, y_test))
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0] , "%")
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
fig , ax = plt.subplots(1,2)
fig.set_size_inches(20,10)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend(['train', 'test'])

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
ax[1].legend(['train', 'test'])

plt.show()
for layer in VGG19_model.layers:
    layer.trainable=True
model.compile(optimizer = Adam(lr=1e-4) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
history = model.fit(datagen.flow(x_train,y_train, batch_size = batch_size) , epochs = epochs , validation_data = (x_test, y_test))
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0] , "%")
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
fig , ax = plt.subplots(1,2)
fig.set_size_inches(20,10)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend(['train', 'test'])

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
ax[1].legend(['train', 'test'])

plt.show()
for i in range (len(VGG19_model.layers)):
    print (i,VGG19_model.layers[i])
  
for layer in VGG19_model.layers[11:]:
    layer.trainable=True
for layer in VGG19_model.layers[0:11]:
    layer.trainable=False
model = Sequential()
model.add(VGG19_model)
model.add(Conv2D(filters=32, kernel_size=(2,2), padding="same", activation="relu"))
model.add(Conv2D(filters=32, kernel_size=(2,2), padding="same", activation="relu"))
model.add(MaxPool2D((2,2) , strides = 2))
model.add(BatchNormalization())
model.add(Dropout(0.20))
model.add(Flatten())
model.add(Dense(256,activation='relu'))
model.add(Dense(5,activation='softmax'))
model.summary()
model.compile(optimizer = Adam(lr=1e-4) , loss = 'categorical_crossentropy' , metrics = ['accuracy'])
history = model.fit(datagen.flow(x_train,y_train, batch_size = batch_size) , epochs = epochs , validation_data = (x_test, y_test))
print("Loss of the model is - " , model.evaluate(x_test,y_test)[0] , "%")
print("Accuracy of the model is - " , model.evaluate(x_test,y_test)[1]*100 , "%")
fig , ax = plt.subplots(1,2)
fig.set_size_inches(20,10)

ax[0].plot(history.history['accuracy'])
ax[0].plot(history.history['val_accuracy'])
ax[0].set_title('Model Accuracy')
ax[0].set_ylabel('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend(['train', 'test'])

ax[1].plot(history.history['loss'])
ax[1].plot(history.history['val_loss'])
ax[1].set_title('Model Loss')
ax[1].set_ylabel('Loss')
ax[1].set_xlabel('Epochs')
ax[1].legend(['train', 'test'])

plt.show()
# serialize weights to HDF5
model.save_weights("Flowers_Recognition_model.h5")
print("Saved model to disk")
predictions = model.predict_classes(x_test)
predictions[:50]
y_test_inv = label_binarizer.inverse_transform(y_test)
print(classification_report(y_test_inv, predictions, target_names = flowers_labels))
cm = confusion_matrix(y_test_inv,predictions)
cm
cm = pd.DataFrame(cm , index = flowers_labels , columns = flowers_labels)
cm
plt.figure(figsize = (10,10))
sns.heatmap(cm,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = flowers_labels , yticklabels = flowers_labels)
# now storing some properly as well as misclassified indexes'.
i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test_inv)):
    if(y_test_inv[i] == predictions[i]):
        prop_class.append(i)
    if(len(prop_class)==8):
        break

i=0
for i in range(len(y_test_inv)):
    if(y_test_inv[i] != predictions[i]):
        mis_class.append(i)
    if(len(mis_class)==8):
        break
count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[prop_class[count]])
        ax[i,j].set_title("Predicted Flower : "+ flowers_labels[predictions[prop_class[count]]] +"\n"+"Actual Flower : "+ flowers_labels[y_test_inv[prop_class[count]]])
        plt.tight_layout()
        count+=1
count=0
fig,ax=plt.subplots(4,2)
fig.set_size_inches(15,15)
for i in range (4):
    for j in range (2):
        ax[i,j].imshow(x_test[mis_class[count]])
        ax[i,j].set_title("Predicted Flower : "+flowers_labels[predictions[mis_class[count]]]+"\n"+"Actual Flower : "+flowers_labels[y_test_inv[mis_class[count]]])
        plt.tight_layout()
        count+=1
img_path = '../input/flowers-recognition/flowers/sunflower/151898652_b5f1c70b98_n.jpg'
org_img = cv2.imread(img_path)
plt.imshow(org_img)
plt.show()
img = image.load_img(img_path, target_size=(224, 224))
plt.imshow(img)
plt.show()
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x.shape
x = preprocess_input(x)
x
model = VGG19(weights='imagenet')
preds = model.predict(x)
cam_predictions = pd.DataFrame(decode_predictions(preds, top=3)[0],columns=['col1','category','probability']).iloc[:,1:]
argmax = np.argmax(preds[0])
output = model.output[:, argmax]
model.summary()
last_conv_layer = model.get_layer('block5_conv4')
grads = K.gradients(output, last_conv_layer.output)[0]
pooled_grads = K.mean(grads, axis=(0, 1, 2))
iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
pooled_grads_value, conv_layer_output_value = iterate([x])
for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
heatmap = np.mean(conv_layer_output_value, axis=-1)
heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()
import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
hif = .8
superimposed_img = heatmap * hif + img
import matplotlib.image as mpimg

output = 'output.jpeg'
cv2.imwrite(output, superimposed_img)
img=mpimg.imread(output)
plt.imshow(img)
plt.axis('off')
plt.title(predictions[0])

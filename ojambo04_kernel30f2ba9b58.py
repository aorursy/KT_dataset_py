# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

# for dirname, _, filenames in os.walk('/kaggle/input'):

#     for filename in filenames:

#         print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow as tf

import matplotlib.pyplot as plt

from glob import glob 

import cv2

from PIL import Image

from pathlib import Path

from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,recall_score,confusion_matrix,classification_report
print(os.listdir("/kaggle/input/chest-xray-pneumonia/chest_xray/"))
path_train = "/kaggle/input/chest-xray-pneumonia/chest_xray/train"

path_val = "/kaggle/input/chest-xray-pneumonia/chest_xray/val"

path_test = "/kaggle/input/chest-xray-pneumonia/chest_xray/test"
plt.figure(1, figsize=(15, 7))



plt.subplot(1 , 2 , 1)

img_pneumonia_l = glob(path_train+"/PNEUMONIA/*.jpeg") #Getting an image in the PNEUMONIA folder

img_pneumonia = np.asarray(plt.imread(img_pneumonia_l[0]))

plt.title('PNEUMONIA X-RAY')

plt.imshow(img_pneumonia)



plt.subplot(1 , 2 , 2)

img_normal_l = glob(path_train+"/NORMAL/*.jpeg") #Getting an image in the NORMAL folder

img_normal = np.asarray(plt.imread(img_normal_l[0]))

plt.title('NORMAL CHEST X-RAY')

plt.imshow(img_normal)



plt.show()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_image_generator = ImageDataGenerator(rescale=1./255, 

                            shear_range = 0.2,

                            zoom_range = 0.2,

                            horizontal_flip=True) # Generator for our training data



validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
train_batch = train_image_generator.flow_from_directory(path_train,

                                            target_size = (224, 224),

                                            classes = ["NORMAL", "PNEUMONIA"],

                                            batch_size=32,            

                                            class_mode='binary')



val_batch = validation_image_generator.flow_from_directory(path_val,

                                        target_size = (224, 224),

                                        classes = ["NORMAL", "PNEUMONIA"],

                                        batch_size=32,                   

                                        class_mode = "binary")



test_batch = validation_image_generator.flow_from_directory(path_test,

                                        target_size = (224, 224),

                                        classes = ["NORMAL", "PNEUMONIA"],

                                        batch_size=32,

                                        class_mode = "binary")



print(train_batch.image_shape)
IMG_SHAPE=(224, 224, 3)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,

                                               include_top=False,

                                               weights='imagenet')
base_model.trainable = False
base_model.summary()
inputs = tf.keras.Input(shape=IMG_SHAPE)



mobilenet_output = base_model(inputs)

x = tf.keras.layers.GlobalAveragePooling2D()(mobilenet_output)

x = tf.keras.layers.Dense(256)(x)

outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)



model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam',

              loss='binary_crossentropy',

              metrics=['accuracy'])
model.summary()
early_stop = tf.keras.callbacks.EarlyStopping(

    monitor='val_loss', 

    patience=3, 

    mode="auto"

)



checkpoint =  tf.keras.callbacks.ModelCheckpoint(

    filepath='best_model',

    save_best_only=True,

    save_weights_only=True,

    monitor='val_loss',

    mode='auto',

    verbose = 1

)
def create_plots(history):

    

    plt.plot(history.history['accuracy'])

    plt.plot(history.history['val_accuracy'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'], loc='lower right')

    plt.show()



    # Plot training & validation loss values

    plt.plot(history.history['loss'])

    plt.plot(history.history['val_loss'])

    plt.title('Model loss')

    plt.ylabel('Loss')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Validation'], loc='upper right')

    plt.show()
grad_model = tf.keras.models.Model([inputs], [mobilenet_output, model.outputs])
history = model.fit(train_batch,

                    callbacks=[early_stop,checkpoint],

                    epochs=5,

                    validation_data=val_batch)
create_plots(history)
test_loss, test_score = model.evaluate_generator(test_batch,steps=100)

print("Loss on test set: ", test_loss)

print("Accuracy on test set: ", test_score)
y_test = []

y_test_pred = []



i=0

for x, y in test_batch:

    if i == len(test_batch):

        break;

        

    y_pred = np.squeeze(model.predict(x)) 

    

    y_test.extend(y)

    y_test_pred.extend(y_pred)

    

    i += 1
print(len(y_test))

print(len(y_test_pred))
y_test = np.array(y_test)

y_test_pred = np.array(y_test_pred)
y_pred = np.squeeze((y_test_pred >= 0.5).astype(int))

print(y_pred)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
img_pneumonia = np.asarray(plt.imread(img_pneumonia_l[0]))

plt.title('PNEUMONIA X-RAY')

plt.imshow(img_pneumonia, cmap='gray')
img_pneumonia.shape
img = tf.keras.preprocessing.image.load_img(img_pneumonia_l[0], target_size=(224, 224))

img = tf.keras.preprocessing.image.img_to_array(img)

print(img.shape)
with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(np.expand_dims(img, axis=0))



grads = tape.gradient(predictions, conv_outputs)[0]

spatial_map = conv_outputs[0]    
print(grads.shape)

print(spatial_map.shape)
weights = np.mean(grads,axis=(0,1))
cam = np.dot(spatial_map, weights)
H = img_pneumonia.shape[0] #height

W = img_pneumonia.shape[1] #width



cam = np.maximum(cam, 0) # ReLU so we only get positive importance

cam = cv2.resize(cam, (W, H), cv2.INTER_NEAREST)

cam = cam / cam.max()
plt.imshow(img_pneumonia, cmap='gray')

plt.imshow(cam, cmap='magma', alpha=0.5)
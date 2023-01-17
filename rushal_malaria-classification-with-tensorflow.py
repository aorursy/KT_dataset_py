import pandas as pd 

import matplotlib.pyplot as plt 

import tensorflow as tf

import numpy as np

import cv2 as cv

import os

from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
data = []

labels = []

image_category=os.listdir("../input/cell_images/cell_images/")



def load_image_data(img_path,img_category):

    """

    load_image_data Function load the image for given directory

    img_path:img_path variable is also file path of image in directory

    img_category:img_category variable use for 0 or 1 for Parasitized or Uninfected 

    """

    try:

        img = cv.imread(img_path)

        img = cv.resize(img,(64,64))

        img = tf.keras.preprocessing.image.img_to_array(img)

        data.append(img)

        labels.append(img_category)

    except Exception as Error:

        print(Error)

        

    

def def_load_directory(image_type,img_category):

    """

    def_load_directory Function load the image for given directory

    image_type:image_type is variable use for define the image Parasitized or Uninfected

               image_type variable is also file path of image in directory

    img_category:img_category variable use for 0 or 1 for Parasitized or Uninfected 

    """

    image_directory_path = os.path.join("../input/cell_images/cell_images/",image_type)

    image_list = os.listdir(image_directory_path)

    for image in image_list:

        image_path = os.path.join(image_directory_path,image)

        load_image_data(image_path,img_category)

        

        

for image_type  in image_category:

    if image_type == "Parasitized":

        def_load_directory(image_type,0)

    if image_type == "Uninfected":

        def_load_directory(image_type,1)

cells = np.array(data)

labels = np.array(labels)

cells.shape,labels.shape
cells,labels = shuffle(cells,labels)
cells = cells.astype("float32")/255

labels = tf.keras.utils.to_categorical(labels)
x_train,x_test,y_train,y_test = train_test_split(cells,labels,test_size=0.33,random_state=45)

x_train.shape,x_test.shape,y_train.shape,y_test.shape
x_train = x_train.reshape(-1,64,64,3)

x_test = x_test.reshape(-1,64,64,3)
conv_layer_1 = tf.keras.layers.Conv2D(32,kernel_size=2,padding="same",

                                                activation='relu',input_shape=(64,64,3))

pool_layer_1 = tf.keras.layers.MaxPool2D(pool_size=2)



conv_layer_2 = tf.keras.layers.Conv2D(64,kernel_size=2,padding="same",activation='relu')

pool_layer_2 = tf.keras.layers.MaxPool2D(pool_size=2)



conv_layer_3 = tf.keras.layers.Conv2D(128,kernel_size=2,padding="same",activation='relu')

pool_layer_3 = tf.keras.layers.MaxPool2D(pool_size=2)



flatten_layer_1 = tf.keras.layers.Flatten()



dropout_layer_1 = tf.keras.layers.Dropout(0.5)



dense_layer_1 = tf.keras.layers.Dense(2,activation="softmax")
model = tf.keras.Sequential([conv_layer_1,pool_layer_1,

                             conv_layer_2,pool_layer_2,

                             conv_layer_3,pool_layer_3,

                             flatten_layer_1,

                             dropout_layer_1,dense_layer_1])
model.summary()
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=['accuracy'])
history = model.fit(x_train,y_train,batch_size=50,epochs=10,verbose=1,validation_split=0.1)
plt.plot(history.history['loss'],label='Loss')

plt.plot(history.history['val_loss'],label="Val Loss")

plt.legend()
accuracy  = model.evaluate(x_test,y_test)

print("Test Accuracy:-",accuracy)
def get_cell_name(label):

    if label==0:

        return "Paracitized"

    if label==1:

        return "Uninfected"



random_value = np.random.randint(0,len(x_test))#Select any random value form test dataset

test_case = x_test[random_value]

test_case = np.expand_dims(test_case,axis=0)#Expand one dimension for model predictation



true_test = y_test[random_value]#Select true of randomly selection image



pred_value = model.predict(np.array(test_case))#Predict the output of model



print("The Predict cell {} with accuracy:{}".format(get_cell_name(np.argmax(pred_value)),np.max(pred_value)))

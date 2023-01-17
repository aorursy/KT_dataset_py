

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import cv2

from tensorflow.keras.optimizers import RMSprop

import os

from os import listdir, makedirs

from os.path import join, exists, expanduser



from keras import applications

from keras.preprocessing.image import ImageDataGenerator

from keras import optimizers

from keras.models import Sequential, Model

from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense

from keras import backend as K

import tensorflow as tf
img_width, img_height = 224, 224 

train_data_dir = '../input/fruits/fruits-360/Training'

validation_data_dir = '../input/fruits/fruits-360/Test'

batch_size = 16
fruit_list = ["Kiwi", "Banana", "Orange",

                "Limes", "Lemon","Pear", "Pear 2", "Papaya","Apple Golden 1","Apple Golden 2",

              "Apple Golden 3","Apple Braeburn","Apple Red 1","Apple Red 2","Apple Red 3",

              "Apple Red Yellow 1","Apple Red Yellow 2"

              "Banana","Banana Red","Banana Lady Finger","Corn","Corn Husk","Mango","Mango Red",

              "Strawberry","Strawberry Wedge", "Pineapple", "Pomegranate",

              "Tomato 1","Tomato 2","Tomato 3","Tomato 4","Tomato Heart","Tomato not Ripened",

              "Watermelon"

             ]

print(len(fruit_list))
train_datagen = ImageDataGenerator(

    rescale=1. / 255,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True)



test_datagen = ImageDataGenerator(rescale=1. / 255)



train_generator = train_datagen.flow_from_directory(

    train_data_dir,

    target_size=(img_height, img_width),

    batch_size=batch_size,

    classes=fruit_list,

    class_mode='categorical')



validation_generator = test_datagen.flow_from_directory(

    validation_data_dir,

    target_size=(img_height, img_width),

    classes=fruit_list,

    batch_size=batch_size,

    class_mode='categorical')
train_generator.class_indices
model = tf.keras.models.Sequential([  ## initializing and making an empty model with sequential

  

    # Note the input shape is the desired size of the image 300x300 with 3 bytes color

    # This is the first convolution layer

    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(224, 224,3)), ## image input shape is 300x300x3 

                           #16 neurons in this layer





    tf.keras.layers.MaxPooling2D(2,2),    # doing max_pooling

    tf.keras.layers.Dropout(0.2),



  

    # The second convolution layer

    tf.keras.layers.Conv2D(32, (3,3), activation='relu'), # another layer with 32 neurons

    tf.keras.layers.MaxPooling2D(2,2),     # doing max_pooling

    tf.keras.layers.Dropout(0.2),





    # The third convolution layer

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # another layer with 64 neurons

    tf.keras.layers.MaxPooling2D(2,2),        # doing max_pooling

    tf.keras.layers.Dropout(0.2),







    # The fourth convolution layer

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # another layer with 64 neurons

    tf.keras.layers.MaxPooling2D(2,2),          # doing max_pooling

    tf.keras.layers.Dropout(0.2),  





    # The fifth convolution 

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), # another layer with 64 neurons

    tf.keras.layers.MaxPooling2D(2,2),        # doing max_pooling

    tf.keras.layers.Dropout(0.2),







    tf.keras.layers.Flatten(),  # reducing layers arrays 

    tf.keras.layers.Dense(512, activation='relu'), # 512 neuron hidden layer







    # Only 1 output neuron. It will contain a value from 0-1 where 0 for ('normal') clas and 

    # 1 for ('pneumonia') class

    tf.keras.layers.Dense(34, activation='softmax')



])



# to get the summary of the model

model.summary()  # summarising a model



# configure the model for traning by adding metrics
# from keras.callbacks import ModelCheckpoint

# filepath="weights.best.hdf5" # mentioning a file for saving checkpoint model in case it gets interrupted



# checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

# ## we marked filepath, monitor and mentioned to save best model only 





# callbacks_list = [checkpoint]  # customising model to save checkpoints

class MyThresholdCallback(tf.keras.callbacks.Callback):

    def __init__(self, threshold):

        super(MyThresholdCallback, self).__init__()

        self.threshold = threshold



    def on_epoch_end(self, epoch, logs=None): 

        val_acc = logs["val_accuracy"]

        if val_acc >= self.threshold:

            self.model.stop_training = True

            print("\nReq acc is reached")

my_callback = MyThresholdCallback(threshold=0.96)


model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=0.001),

              metrics = ['accuracy']) # compiling mode
hist = model.fit_generator(

    generator = train_generator,

    steps_per_epoch = 21457//(100),

    epochs = 100,

    shuffle=True,

    validation_data = validation_generator,

    callbacks=[my_callback],

    validation_steps = 7777 // 100

                   )
eval_datagen = ImageDataGenerator(rescale = 1./255)



test_generator = eval_datagen.flow_from_directory(

    '../input/fruits/fruits-360/Test',

    target_size = (224, 224),

    classes=fruit_list,

    class_mode = 'categorical'

    

)
eval_result = model.evaluate_generator(test_generator)

print('loss rate at evaluation data :', eval_result[0])

print('accuracy rate at evaluation data :', eval_result[1])
model.save_weights('34_fruits.h5')

print("Model is saved")



model.save('Complete_34_fruits.h5')
print(train_generator.class_indices)

classes = train_generator.class_indices
img = cv2.imread('../input/fruits/fruits-360/Test/Apple Golden 2/322_100.jpg')

img = cv2.resize(img,(224,224))

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

img = np.expand_dims(img,axis=0)

p=model.predict(img)


def get_key(val,dict): 

    for key, value in dict.items(): 

         if val == value: 

             return key 

  

    return "key doesn't exist"





P = p.argmax(axis=1)

print(type(P))

P = P.tolist()
print(P)

print(get_key(P[0],classes))
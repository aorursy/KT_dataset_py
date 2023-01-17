#https://www.kaggle.com/vladminzatu/cactus-detection-with-tensorflow-2-0

!pip install tensorflow==2.0.0-alpha0



from tensorflow.python.ops import control_flow_util

control_flow_util.ENABLE_CONTROL_FLOW_V2 = True
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

print(os.listdir("../input"))
#https://www.kaggle.com/abhishekrock/cat-dog-try

#import cv2

label=[]

data1=[]

counter=0

path_array = []

datagen = []

path="../input/train/train"

for file in os.listdir(path):

    #image_data=cv2.imread(os.path.join(path,file), cv2.IMREAD_GRAYSCALE)

    path_array.append(os.path.join(path,file))

    #image_data=cv2.resize(image_data,(150,150))

    if file.startswith("cat"):

        label.append('cat')

    elif file.startswith("dog"):

        label.append('dog')



print(path_array[:5])

print(label[:5])
d = {'path': path_array, 'label': label}

df_train = pd.DataFrame(data=d)

df_train.head()
#https://keras.io/preprocessing/image/

#https://medium.com/@arindambaidya168/https-medium-com-arindambaidya168-using-keras-imagedatagenerator-b94a87cdefad

#https://github.com/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb

import tensorflow as tf

from keras_preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import RMSprop

#datagen=ImageDataGenerator(rescale=1./255, validation_split=0.2)

IMAGE_HT_WID = 96

# train_datagen = ImageDataGenerator(

#         rescale=1./255,

#         shear_range=0.2,

#         zoom_range=0.2,

#         horizontal_flip=True, 

#         validation_split=0.1)



train_datagen = ImageDataGenerator(

#                                rotation_range=15,

#                                width_shift_range=0.1,

#                                height_shift_range=0.1,

#                                shear_range=0.01,

#                                zoom_range=[0.9, 1.25],

#                                horizontal_flip=True,

#                                vertical_flip=False,

#                                #data_format='channels_last',

#                               fill_mode='reflect',

#                               brightness_range=[0.5, 1.5],

                               validation_split=0.1,

                               rescale=1./255)





test_datagen = ImageDataGenerator(rescale=1./255)

#test_datagen = ImageDataGenerator()



train_generator=train_datagen.flow_from_dataframe(

                    dataframe=df_train,

                    #directory="./train/",

                    x_col="path",

                    y_col="label",

                    subset="training",

                    batch_size=50,

                    seed=42,

                    shuffle=True,

                    class_mode="categorical",

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))



valid_generator=train_datagen.flow_from_dataframe(

                    dataframe=df_train,

                    #directory="./train/",

                    x_col="path",

                    y_col="label",

                    subset="validation",

                    batch_size=50,

                    seed=42,

                    shuffle=True,

                    class_mode="categorical",

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))



from keras.models import Sequential

from keras_preprocessing.image import ImageDataGenerator

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization

from keras.layers import Conv2D, MaxPooling2D

from keras import regularizers, optimizers

import pandas as pd

import numpy as np



# model = Sequential()

# model.add(Conv2D(32, (3, 3), padding='same',

#                  input_shape=(IMAGE_HT_WID,IMAGE_HT_WID,3)))

# model.add(Activation('relu'))

# model.add(Conv2D(32, (3, 3)))

# model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))

# model.add(Conv2D(64, (3, 3), padding='same'))

# model.add(Activation('relu'))

# model.add(Conv2D(64, (3, 3)))

# model.add(Activation('relu'))

# model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))

# model.add(Flatten())

# model.add(Dense(512))

# model.add(Activation('relu'))

# model.add(Dropout(0.5))

# model.add(Dense(2, activation='softmax'))

# model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])

# #https://github.com/lmoroney/dlaicourse/blob/master/Course%202%20-%20Part%206%20-%20Lesson%203%20-%20Notebook.ipynb

# from tensorflow.keras import layers

# from tensorflow.keras import Model

# !wget --no-check-certificate \

#     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \

#     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

  

# from tensorflow.keras.applications.inception_v3 import InceptionV3



# local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



# pre_trained_model = InceptionV3(input_shape = (IMAGE_HT_WID, IMAGE_HT_WID, 3), 

#                                 include_top = False, 

#                                 weights = None)



# pre_trained_model.load_weights(local_weights_file)



# for layer in pre_trained_model.layers:

#   layer.trainable = False

  

# # pre_trained_model.summary()



# last_layer = pre_trained_model.get_layer('mixed7')

# print('last layer output shape: ', last_layer.output_shape)

# last_output = last_layer.output

# from tensorflow.keras.optimizers import RMSprop



# # Flatten the output layer to 1 dimension

# x = layers.Flatten()(last_output)

# # Add a fully connected layer with 1,024 hidden units and ReLU activation

# x = layers.Dense(1024, activation='relu')(x)

# # Add a dropout rate of 0.2

# x = layers.Dropout(0.2)(x)                  

# # Add a final sigmoid layer for classification

# x = layers.Dense(2, activation='sigmoid')(x)           



# model = Model( pre_trained_model.input, x) 



# model.compile(optimizer = RMSprop(lr=0.0001), 

#               loss = 'categorical_crossentropy', 

#               metrics = ['acc'])

# model.summary()



#https://www.tensorflow.org/tutorials/images/transfer_learning

import tensorflow as tf

from tensorflow import keras

base_model = tf.keras.applications.MobileNetV2(input_shape=(IMAGE_HT_WID, IMAGE_HT_WID, 3),

                                               include_top=False, 

                                               weights='imagenet')

base_model.trainable = False

print(base_model.summary())

model = tf.keras.Sequential([

  base_model,

  keras.layers.GlobalAveragePooling2D(),

  keras.layers.Dense(2, activation='sigmoid')

])

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), 

              loss='categorical_crossentropy', 

              metrics=['accuracy'])

print(model.summary())
EPOCHS=5

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

history = model.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=EPOCHS,

                    verbose=1

)
#model.evaluate_generator(generator=valid_generator)
import matplotlib.pyplot as plt

acc = history.history['accuracy']

val_acc = history.history['val_accuracy']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





plt.show()
test_path_array = []



testpath="../input/test1/test1"

for file in os.listdir(testpath):

   

    test_path_array.append(os.path.join(testpath,file))

    

   

print(test_path_array[5])



dtest = {'path': test_path_array}

df_test = pd.DataFrame(data=dtest)

df_test.head()



#test_datagen=ImageDataGenerator(rescale=1./255.)

test_generator=test_datagen.flow_from_dataframe(

                dataframe=df_test,

                #directory="./test/",

                x_col="path",

                y_col=None,

                batch_size=50,

                seed=42,

                shuffle=False,

                class_mode=None,

                target_size=(IMAGE_HT_WID,IMAGE_HT_WID))

STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

test_generator.reset()

pred=model.predict_generator(test_generator,

                steps=STEP_SIZE_TEST,

                verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
len(predicted_class_indices)
predicted_class_indices[:10]
#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb#scrollTo=dn-6c02VmqiN



print(predicted_class_indices[:12])



%matplotlib inline



import matplotlib.image as mpimg

import matplotlib.pyplot as plt



# Parameters for our graph; we'll output images in a 4x4 configuration

nrows = 12

ncols = 4



pic_index = 0 # Index for iterating over images

# Set up matplotlib fig, and size it to fit 4x4 pics

fig = plt.gcf()

fig.set_size_inches(ncols*4, nrows*4)







for i, img_path in enumerate(df_test.path[:48]):

  # Set up subplot; subplot indices start at 1

  sp = plt.subplot(nrows, ncols, i + 1)

  sp.axis('Off') # Don't show axes (or gridlines)



  img = mpimg.imread(img_path)

  plt.imshow(img)

  plt.title(str(predicted_class_indices[i]))
len(predicted_class_indices)
len(df_test)
id = [*range(1,len(df_test)+1)]

dataframe_output=pd.DataFrame({"id":id})

dataframe_output["label"]=predicted_class_indices

dataframe_output.to_csv("submission.csv",index=False)
dataframe_output.head()
#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb#scrollTo=dn-6c02VmqiN



# print(label[:12])



# %matplotlib inline



# import matplotlib.image as mpimg

# import matplotlib.pyplot as plt



# # Parameters for our graph; we'll output images in a 4x4 configuration

# nrows = 4

# ncols = 4



# pic_index = 0 # Index for iterating over images

# # Set up matplotlib fig, and size it to fit 4x4 pics

# fig = plt.gcf()

# fig.set_size_inches(ncols*4, nrows*4)







# for i, img in enumerate(data1[:12]):

#   # Set up subplot; subplot indices start at 1

#   sp = plt.subplot(nrows, ncols, i + 1)

#   sp.axis('Off') # Don't show axes (or gridlines)



#  # img = mpimg.imread(img_path)

#   plt.imshow(img)

#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb#scrollTo=dn-6c02VmqiN



# import tensorflow as tf

# from tensorflow.keras.optimizers import RMSprop

# model = tf.keras.models.Sequential([

#     tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),

#     tf.keras.layers.MaxPooling2D(2, 2),

#     tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),

#     tf.keras.layers.MaxPooling2D(2, 2),

#     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),

#     tf.keras.layers.MaxPooling2D(2, 2),

#     tf.keras.layers.Flatten(),

#     tf.keras.layers.Dense(512, activation='relu'),

#     tf.keras.layers.Dense(1, activation='sigmoid')

# ])



# model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

# model.summary()
#https://www.kaggle.com/abhishekrock/cat-dog-try

# import numpy as np



# data1=np.array(data1)

# print (data1.shape)

# data1=data1.reshape((data1.shape)[0],(data1.shape)[1],(data1.shape)[2],3)

# #data1=data1/255

# labels=np.array(label)

# model.fit(data1,labels,validation_split=0.25,epochs=2,batch_size=10)
#https://www.kaggle.com/abhishekrock/cat-dog-try

# test_data=[]

# id=[]

# counter=0

# for file in os.listdir("../input/test1/test1"):

#     #image_data=cv2.imread(os.path.join("../input/test1/test1",file), cv2.IMREAD_GRAYSCALE)

#     image_data=cv2.imread(os.path.join("../input/test1/test1",file))

#     try:

#         image_data=cv2.resize(image_data,(150,150))

#         test_data.append(image_data/255)

#         id.append((file.split("."))[0])

#     except:

#         print ("exception while processing")

#     counter+=1

#     if counter%1000==0:

#         print (counter," image data retreived")

        

# test_data1=np.array(test_data)

# print (test_data1.shape)

# test_data1=test_data1.reshape((test_data1.shape)[0],(test_data1.shape)[1],(test_data1.shape)[2],3)

# dataframe_output=pd.DataFrame({"id":id})

# predicted_labels=model.predict(test_data1)

# predicted_labels=np.round(predicted_labels,decimals=2)

# labels=[1 if value>0.5 else 0 for value in predicted_labels]

# dataframe_output["label"]=labels

# dataframe_output.to_csv("submission.csv",index=False)

#dataframe_output.head()
#https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/Exercises/Exercise%205%20-%20Real%20World%20Scenarios/Exercise%205%20-%20Answer.ipynb#scrollTo=dn-6c02VmqiN



# print(labels[:12])



# %matplotlib inline



# import matplotlib.image as mpimg

# import matplotlib.pyplot as plt



# # Parameters for our graph; we'll output images in a 4x4 configuration

# nrows = 4

# ncols = 4



# pic_index = 0 # Index for iterating over images

# # Set up matplotlib fig, and size it to fit 4x4 pics

# fig = plt.gcf()

# fig.set_size_inches(ncols*4, nrows*4)







# for i, img in enumerate(test_data[:12]):

#   # Set up subplot; subplot indices start at 1

#   sp = plt.subplot(nrows, ncols, i + 1)

#   sp.axis('Off') # Don't show axes (or gridlines)



#  # img = mpimg.imread(img_path)

#   plt.imshow(img)

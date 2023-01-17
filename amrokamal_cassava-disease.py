import tensorflow as tf

#tf.enable_eager_execution()
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
import os

import numpy as numpy

from PIL import Image

l=[]

for i in os.listdir('../input/cassava-disease/train/train'):

    print (len(os.listdir(os.path.join('../input/cassava-disease/train/train',i))))

    l.append(i)
l
import os

import numpy as numpy

from PIL import Image

from keras.preprocessing import image as IMG

from keras.applications.inception_v3 import preprocess_input



RESHAPE=(64,64)



def load_image(path):

    img = Image.open(path)

    return img





def preprocess_image(cv_img):

    cv_img = cv_img.resize(RESHAPE)

    img = np.array(cv_img)

    return img/255.0



def crop(img, hoffset,woffset):

    # Note: image_data_format is 'channel_last'

    assert img.shape[2] == 3

    height, width = img.shape[0], img.shape[1]

    

    return img[woffset:width-woffset, hoffset:height-hoffset, :]



def load_data(path,n_images):

    images=[]

    labels=[]

    

    i=0

    for cat in os.listdir(path):

        cat_path=os.path.join(path,cat)

        num_imgs_ld=0

        for image in os.listdir(cat_path):

            image_path=os.path.join(cat_path,image)

            #img=load_image(image_path)

            

            img = IMG.load_img(image_path, target_size=(320, 320))

            x = IMG.img_to_array(img)

            #x = np.expand_dims(x, axis=0)

            #x = preprocess_input(x)

            x=crop(x,48,48)

            #images.append(preprocess_image(img))

            images.append(x/255.0)

            labels.append(i)

            num_imgs_ld=num_imgs_ld+1

         #   if num_imgs_ld > n_images[i] - 1: 

           #     break

            

        

        i=i+1    

    return np.array(images),np.array(labels)

    #return (images , labels)

            

    
data,labels=load_data('../input/cassava-disease/train/train',[770,1440,315,465,1400])
import matplotlib.pyplot as plt

plt.imshow(data[0])
for i in range(5):

    print(i)

    idx = np.random.permutation(len(data))

    data,labels = data[idx], labels[idx]



plt.imshow(data[0])
from keras.utils import to_categorical

labels= to_categorical(labels)

labels.shape
train_data=data[:5000]

train_labels=labels[:5000]

val_data=data[5000:]

val_labels=labels[5000:]



print(train_data.shape)

print(val_data.shape)
cache_dir = os.path.expanduser(os.path.join('~', '.keras'))

if not os.path.exists(cache_dir):

    os.makedirs(cache_dir)

models_dir = os.path.join(cache_dir, 'models')

if not os.path.exists(models_dir):

    os.makedirs(models_dir)
!cp ../input/keras-pretrained-models/vgg* ~/.keras/models/

!ls ~/.keras/models
from keras.applications.vgg16 import VGG16

from keras.preprocessing import image

from keras.models import Model

from keras.layers import Dense,Input, GlobalAveragePooling2D,Dropout,Cropping2D

from keras import backend as K



# create the base pre-trained model

base_model = VGG16(weights='imagenet', include_top=False,input_shape=(224,224,3))

base_model.summary()
print(base_model.input)


"""



"""

x = base_model.output

x=GlobalAveragePooling2D()(x)

# let's add a fully-connected layer



#x = Dense(1024, activation='relu')(x)

#x=Dropout(0.5)(x)

x = Dense(1024, activation='relu')(x)

x=Dropout(0.2)(x)



x = Dense(512, activation='relu')(x)

x=Dropout(0.2)(x)





# and a logistic layer -- let's say we have 200 classes

predictions = Dense(5, activation='softmax',name="out")(x)



# this is the model we will train



model = Model(inputs=base_model.input, outputs=predictions)



# first: train only the top layers (which were randomly initialized)

# i.e. freeze all convolutional InceptionV3 layers

for layer in base_model.layers:

    layer.trainable = False



# compile the model (should be done *after* setting layers to non-trainable)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])



model.summary()
from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

        # set input mean to 0 over the dataset

        featurewise_center=False,

        # set each sample mean to 0

        samplewise_center=False,

        # divide inputs by std of dataset

        featurewise_std_normalization=False,

        # divide each input by its std

        samplewise_std_normalization=False,

        # apply ZCA whitening

        zca_whitening=False,

        # epsilon for ZCA whitening

        zca_epsilon=1e-06,

        # randomly rotate images in the range (deg 0 to 180)

        rotation_range=30,

        # randomly shift images horizontally

        width_shift_range=0.1,

        # randomly shift images vertically

        height_shift_range=0.1,

        # set range for random shear

        shear_range=0.,

        # set range for random zoom

        zoom_range=0.1,

        # set range for random channel shifts

        channel_shift_range=0.,

        # set mode for filling points outside the input boundaries

        fill_mode='nearest',

        # value used for fill_mode = "constant"

        cval=0.,

        # randomly flip images

        horizontal_flip=True,

        # randomly flip images

        vertical_flip=False,

        # set rescaling factor (applied before any other transformation)

        rescale=None,

        # set function that will be applied on each input

        preprocessing_function=None,

        # image data format, either "channels_first" or "channels_last"

        data_format='channels_last',

        # fraction of images reserved for validation (strictly between 0 and 1)

        validation_split=0.0)

datagen.fit(train_data)

    
from keras.callbacks import ModelCheckpoint



checkpointer = ModelCheckpoint(filepath='weights-tfl-02.h5',monitor='val_acc',

                              verbose=1, save_best_only=True)









history= model.fit_generator(datagen.flow(train_data, train_labels, batch_size=32),

         epochs=10, verbose=1,validation_data=(val_data, val_labels),

      workers=4,steps_per_epoch=150, callbacks=[checkpointer])



#history= model.fit(train_data,train_labels, batch_size=32,

   #          epochs=1, verbose=1, validation_data=(val_data, val_labels))

    

# Plot training & validation accuracy values

plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()





from keras.models import load_model

model=load_model('weights-tfl-02.h5')
evl=model.evaluate(x=val_data, y=val_labels, batch_size=32, verbose=0)

print(evl)
"""pred = model.predict(val_data)
"""

APC=tf.metrics.mean_per_class_accuracy(

    val_labels,

    tf.convert_to_tensor(pred),

    num_classes=5

    

)

with tf.Session() as sess:

    sess.run(APC)

print(np.array(APC))


for i, layer in enumerate(base_model.layers):

    print(i, layer.name)



# we chose to train the top 2 inception blocks, i.e. we will freeze

# the first 249 layers and unfreeze the rest:



for layer in model.layers[:11]:

    layer.trainable = False

for layer in model.layers[11:]:

    layer.trainable = True



# we need to recompile the model for these modifications to take effect

# we use SGD with a low learning rate

from keras.optimizers import SGD

model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy',metrics=['accuracy'])



# we train our model again (this time fine-tuning the top 2 inception blocks

# alongside the top Dense layers





history2= model.fit_generator(datagen.flow(train_data, train_labels, batch_size=32),

     epochs=5, verbose=1,  validation_data=(val_data, val_labels),

                workers=4,steps_per_epoch=150,callbacks=[checkpointer])

#history2= model.fit(train_data,train_labels, batch_size=32,epochs=20,

                #    verbose=1,validation_data=(val_data, val_labels))

# Plot training & validation accuracy values

plt.plot(history2.history['acc'])

plt.plot(history2.history['val_acc'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()



# Plot training & validation loss values

plt.plot(history2.history['loss'])

plt.plot(history2.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Test'], loc='upper left')

plt.show()
model=load_model('weights-tfl-02.h5')

evl=model.evaluate(x=val_data, y=val_labels, batch_size=32, verbose=0)

print(evl)
testimages=[]

testimagesnames=[]

for imagename in  os.listdir('../input/cassava-disease/test/test/0'):

    imagepath= os.path.join('../input/cassava-disease/test/test/0',imagename)

    img = IMG.load_img(imagepath, target_size=(320, 320))

    x = IMG.img_to_array(img)

            #x = np.expand_dims(x, axis=0)

            #x = preprocess_input(x)

    x=crop(x,48,48)

            #images.append(preprocess_image(img))

    testimages.append(x/255.0)

    testimagesnames.append(imagename)

    

testimages=np.array(testimages)

testimages.shape
testimagesnames[:5]
#make predictions

test_predictions=model.predict(testimages)

predictions=np.argmax(test_predictions,axis=1)
l
index2labs={0:'cgm', 1:'cbsd', 2:'healthy', 3:'cbb', 4:'cmd'}
import pandas as pd

subm = pd.DataFrame({'Category':[index2labs[j] for j in predictions],

         'Id':[testimagesnames[i] for i in range(len(testimagesnames))]})

subm.to_csv('submission_crop_withDA_83.csv', index=False)

subm.head()

# Save tf.keras model in HDF5 format.

# keras_file = "keras_model.h5"

# tf.keras.models.save_model(model, keras_file)



# # Convert to TensorFlow Lite model.

# converter = tf.lite.TFLiteConverter.from_keras_model_file(keras_file)

# tflite_model = converter.convert()

# open("converted_model.tflite", "wb").write(tflite_model)
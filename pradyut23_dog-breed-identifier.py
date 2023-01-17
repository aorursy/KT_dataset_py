import cv2

import numpy as np

import pandas as pd

import shutil

import os

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,GlobalAveragePooling2D,Dense,Flatten,Dropout,AvgPool2D

from keras.layers.normalization import BatchNormalization

from keras.constraints import unit_norm
labels=pd.read_csv('../input/dog-breed-identification/labels.csv')

labels.head()
breeds=list(labels['breed'].unique())

len(breeds)
## dividing the train data into different folders according to their breed names

os.mkdir('/kaggle/working/new_train')



for i in range(len(labels)):

    if labels['breed'][i] not in os.listdir('/kaggle/working/new_train'):

        os.mkdir('/kaggle/working/new_train/'+labels['breed'][i])

    shutil.copy('../input/dog-breed-identification/train/'+labels['id'][i]+'.jpg', '/kaggle/working/new_train/'+labels['breed'][i])

from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



image_size=224  #for the inception model

image_path='/kaggle/working/new_train/'



train_datagen=ImageDataGenerator(

                        rescale=1./255,

                        validation_split=0.1,

                        horizontal_flip=True,

                        width_shift_range=0.2,

                        height_shift_range=0.2,

                        shear_range=0.2,

                        rotation_range=40,

                        fill_mode='nearest'

                        )



train_generator=train_datagen.flow_from_directory(

                        image_path, 

                        target_size=(image_size,image_size),

                        subset='training',

                        shuffle=True,

                        batch_size=128,

                        class_mode='categorical'

                        )



valid_datagen=ImageDataGenerator(

                        validation_split=0.1,

                        rescale=1./255

                        )



valid_generator=valid_datagen.flow_from_directory(

                        image_path, 

                        target_size=(image_size,image_size),

                        subset='validation',

                        shuffle=False,

                        batch_size=128,

                        class_mode='categorical'

                        )
x,y = train_generator.next()

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

for i in range(0,10):

    image = x[i]

    plt.imshow(image)

    c=0

    for i in y[i]:

        if i==0:

            c+=1

        else:break

    label=labels[c]

    label=label.replace('_',' ')

    label=label.lower()

    plt.title(label)

    plt.show()
from tensorflow.keras.applications.inception_v3 import InceptionV3



#using pre-trained weights for the inception model

local_weights_file = '../input/keras-pretrained-models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

Inception = InceptionV3(input_shape = (224,224,3), 

                                include_top = False, 

                                weights = local_weights_file)



#building a sequential model with inception layer base and only an average pooling layer before the output layer



model2=Sequential()

model2.add(Inception)



model2.add(GlobalAveragePooling2D())

model2.add(Dense(512,activation='relu'))

model2.add(Dropout(0.2))

model2.add(Dense(len(breeds),activation='softmax'))



model2.layers[0].trainable=False



model2.compile(optimizer='sgd',

             loss='categorical_crossentropy',

             metrics=['accuracy']

             )



model2.summary()
callback=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=5,min_delta=0,mode='auto',restore_best_weights=False,baseline=None)



history=model2.fit_generator(train_generator,

                   steps_per_epoch=73,

                   epochs=100,

                   validation_data=valid_generator,

                   validation_steps=8,

                   callbacks=[callback])
os.mkdir('/kaggle/working/models/')

model2.save('/kaggle/working/models/my_dog_model.h5')
def plot_model(history):

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))

    fig.suptitle('Model Accuracy and Loss')



    ax1.plot(history.history['accuracy'])

    ax1.plot(history.history['val_accuracy'])

    ax1.title.set_text('Accuracy')

    ax1.set_ylabel('Accuracy')

    ax1.set_xlabel('Epoch')

    ax1.legend(['Train','Valid'],loc=4)



    ax2.plot(history.history['loss'])

    ax2.plot(history.history['val_loss'])

    ax2.title.set_text('Loss')

    ax2.set_ylabel('Loss')

    ax2.set_xlabel('Epoch')

    ax2.legend(['Train','Valid'],loc=1)



    fig.show()



plot_model(history)
os.mkdir('/kaggle/working/new_test')

os.mkdir('/kaggle/working/new_test/test')

test_images=os.listdir('../input/dog-breed-identification/test/')



for i in range(len(test_images)):

    shutil.copy('../input/dog-breed-identification/test/'+test_images[i],'/kaggle/working/new_test/test')
test_generator = valid_datagen.flow_from_directory(

    '/kaggle/working/new_test/',

    target_size=(224,224),

    color_mode="rgb",

    batch_size=32,

    class_mode=None,

    shuffle=False

)
test_generator.reset()

pred=model2.predict_generator(test_generator,verbose=1,steps=10357/32)
n = len(labels)

n_class = len(breeds)  

class_to_num = dict(zip(breeds, range(n_class)))

num_to_class = dict(zip(range(n_class), breeds))



df2 = pd.read_csv('../input/dog-breed-identification/sample_submission.csv')



for b in breeds:

    df2[b] = pred[:,class_to_num[b]]



df2.to_csv('pred.csv', index=None)
predicted_class_indices=np.argmax(pred,axis=1)



labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]
from cv2 import imread

from keras.applications.inception_v3 import preprocess_input



def predict(url, filename):

    # download and save

    os.system("curl -s {} -o {}".format(url, filename))

    img = Image.open(filename)

    img = img.convert('RGB')

    img = img.resize((image_size,image_size))

    img.save(filename)

    # show image

    plt.figure(figsize=(4, 4))

    plt.imshow(img)

    plt.axis('off')

    # predict

    img = imread(filename)

    img = preprocess_input(img)

    probs = model2.predict(np.expand_dims(img, axis=0))

    

    dict1={}

    for i,j in enumerate(probs[0]):

        dict1[i]=j

    

    a=max(dict1.keys(), key=(lambda k: dict1[k]))

    predicted_breed=breeds[a]

    print(predicted_breed)
predict("https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/12224329/Shih-Tzu-On-White-01.jpg",

                     "test_image_1.jpg")
predict("https://vetstreet.brightspotcdn.com/dims4/default/d742db0/2147483647/thumbnail/645x380/quality/90/?url=https%3A%2F%2Fvetstreet-brightspot.s3.amazonaws.com%2F98%2Fd98250a0d311e0a2380050568d634f%2Ffile%2FGolden-Retriever-3-645mk062411.jpg",

                     "test_image_1.jpg")
predict("https://i.pinimg.com/originals/18/59/c2/1859c289470c3fddd8be3d07bf8982b6.jpg",

                     "test_image_1.jpg")
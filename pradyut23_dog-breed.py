import os

import cv2

import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split



import tensorflow as tf

from tensorflow import keras

from keras.models import Sequential

from keras.layers import Conv2D,MaxPooling2D,GlobalAveragePooling2D,Dense,Flatten,Dropout

from keras.layers.normalization import BatchNormalization
#Importing the images



image_path = "/kaggle/input/stanford-dogs-dataset/images/Images/"



categories = os.listdir(image_path)

breeds=[]

for breed in categories:

    a=breed[10:]

    a=a.replace('_',' ')

    a=a.lower()

    breeds.append(a)

print("List of breeds = ",breeds[:20],"\n\nNo. of breeds = ", len(categories))
#Creating DataGenerators to get labels from image directories



from tensorflow.python.keras.preprocessing.image import ImageDataGenerator



image_size=224  #for the inception model



train_datagen=ImageDataGenerator(

                        rescale=1./255,

                        validation_split=0.2,

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

                        batch_size=24,

                        class_mode='categorical'

                        )



valid_datagen=ImageDataGenerator(

                        validation_split=0.2,

                        rescale=1./255

                        )



valid_generator=valid_datagen.flow_from_directory(

                        image_path, 

                        target_size=(image_size,image_size),

                        subset='validation',

                        shuffle=False,

                        batch_size=24,

                        class_mode='categorical'

                        )
#Viewing some augmented images from the dataset with the correct label



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

    label=labels[c][10:]

    label=label.replace('_',' ')

    label=label.lower()

    plt.title(label)

    plt.show()

#Applying transfer learning to build a model



from tensorflow.keras.applications.inception_v3 import InceptionV3



#using pre-trained weights for the inception model

local_weights_file = '../input/inception/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'



Inception = InceptionV3(input_shape = (224,224,3), 

                                include_top = False, 

                                weights = local_weights_file)



#building a sequential model with inception layer base and only an average pooling layer before the output layer



model2=Sequential()

model2.add(Inception)

model2.add(Dropout(0.2))

model2.add(Dense(1024,activation='relu'))

model2.add(Dropout(0.2))

model2.add(Dense(1024,activation='relu'))

model2.add(Dropout(0.2))

model2.add(GlobalAveragePooling2D())

model2.add(Dense(512,activation='relu'))

model2.add(Dense(len(breeds),activation='softmax'))



model2.layers[0].trainable=False



model2.compile(optimizer='sgd',

             loss='categorical_crossentropy',

             metrics=['accuracy']

             )



model2.summary()
#Training the model 

callback=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,min_delta=0,mode='auto',restore_best_weights=False,baseline=None)



history=model2.fit_generator(train_generator,

                   steps_per_epoch=688,

                   epochs=20,

                   validation_data=valid_generator,

                   validation_steps=170,

                   callbacks=[callback])
def plot_model(history):

    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))

    fig.suptitle('Model 2 Accuracy and Loss')



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
#Predicting a random image not present in the dataset



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

    predicted_breed=labels[a][10:]

    predicted_breed=predicted_breed.replace('_',' ')

    predicted_breed=predicted_breed.lower()

    print(predicted_breed)

        

predict("https://s3.amazonaws.com/cdn-origin-etr.akc.org/wp-content/uploads/2017/11/12224329/Shih-Tzu-On-White-01.jpg",

                     "test_image_1.jpg")
from sklearn.metrics import classification_report



valid_generator.reset()

predictions=model2.predict_generator(valid_generator,steps=len(valid_generator))

y=np.argmax(predictions,axis=1)



print('Classification Report')

report=classification_report(y_true=valid_generator.classes,y_pred=y,target_names=valid_generator.class_indices)

print(report)
import pandas as pd

import seaborn as sns

from sklearn.metrics import confusion_matrix



print('Confusion Matrix')



cm=confusion_matrix(valid_generator.classes,y)

df=pd.DataFrame(cm,columns=valid_generator.class_indices)

plt.figure(figsize=(80,80))

sns.heatmap(df,annot=True)
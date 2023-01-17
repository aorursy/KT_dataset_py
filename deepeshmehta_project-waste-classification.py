import numpy as np

import tensorflow as tf 

import matplotlib.pyplot as plt

from tensorflow import keras

import tensorflow_hub as hub



image_generator=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255,validation_split=0.2)



img_data_train=image_generator.flow_from_directory('../input/waste-classification-data/DATASET/TRAIN',

                                             target_size=(224,224),subset='training'

                                             )



img_data_val=image_generator.flow_from_directory('../input/waste-classification-data/DATASET/TRAIN',

                                             target_size=(224,224),#Batch size can be changed ,by default its 32

                                             subset='validation') 
img_data_train.class_indices
for sample_batch,sample_label in img_data_train:

    print(sample_batch.shape)

    print(sample_label.shape)

    break
class_names=['Organic','Recyclable']
def display(img_batch,label_batch):

    plt.figure(figsize=(10,9))

    plt.subplots_adjust(wspace=0.7,hspace=0.7)

    for i in range(30):

        plt.subplot(6,5,i+1)

        plt.imshow(img_batch[i])

        plt.title(class_names[np.argmax(label_batch[i])])

display(sample_batch,sample_label)
feature_extract_url="https://tfhub.dev/tensorflow/resnet_50/feature_vector/1"



    

feature_extraction_layer=hub.KerasLayer(str(feature_extract_url),input_shape=(224,224,3))



feature_extraction_layer.trainable=False


model=tf.keras.Sequential([

feature_extraction_layer,

tf.keras.layers.Dense(img_data_train.num_classes,activation='sigmoid')

])



model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])





steps=np.ceil(img_data_train.samples/img_data_train.batch_size)



history=model.fit(img_data_train,epochs=1,validation_data=img_data_val,steps_per_epoch=steps)
test_data=image_generator.flow_from_directory(str('../input/waste-classification-data/DATASET/TEST'),

                                             target_size=(224,224),#Batch size can be changed ,by default its 32

                                             )



model.evaluate(test_data)
from keras.preprocessing import image

from PIL import Image
# FOR TESTING YOUR OWN IMAGE



img_path='../input/usertest/test.jpg'

img=image.load_img(img_path,target_size=(224,224))

x=image.img_to_array(img)

x=np.expand_dims(x,axis=0)

x=x/255

print('Input img shape:',x.shape)



my_img=Image.open(img_path)

my_img=my_img.resize((224,224))

plt.imshow(my_img)

pred=model.predict(x)

plt.title(f'Material is {class_names[np.argmax(pred)]}')
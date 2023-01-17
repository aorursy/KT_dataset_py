#importing all the libraries needed
import numpy as np
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout,GlobalAveragePooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
print(keras.__version__)
# created a custom callback when accuracy reaches 98%

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if logs.get('accuracy') is not None and logs.get('accuracy') > 0.98:
            print(' 98% accuracy reached')
            self.model.stop_training=True      
#performing image augmentation on the training data , set validation split to a low value to increase no. of training examples

imagegen=ImageDataGenerator(rescale=1./255,rotation_range=10,zoom_range=0.2,height_shift_range=0.2,width_shift_range=0.2,
                                 horizontal_flip=True,fill_mode='nearest',shear_range=0.2,validation_split=0.02,
                            channel_shift_range=10.0,brightness_range=[0.1,0.3])

# set training samples to flow from train directory
train_datagen=imagegen.flow_from_directory(r'C:\Users\admin\Documents\car_classifier\Train',
                                                target_size=(224,224),batch_size=89,shuffle=True,class_mode='categorical',
                                                subset='training')
valid_datagen=imagegen.flow_from_directory(r'C:\Users\admin\Documents\car_classifier\Train',subset='validation',target_size=(224,224))
# used MobileNet architecture as it was lightweight and requires less computing power, provides nearly the same
# accuracy as VGG16 and is less complex in nature, not as accurate as InceptionV3 , but significant reduction in 
# size of the model
mobile = keras.applications.mobilenet.MobileNet()
mobile.summary()
print(len(mobile.layers))
# set output of MobileNet model to layer 87
last_output=mobile.layers[-6].output
predictions=Dense(45,activation='softmax',name='fc3')(last_output)
model=Model(inputs=mobile.input,outputs=predictions)
# freezing the layers that i dont want to train
for layer in model.layers[:-36]:
    layer.trainable=False
    
model.summary()
print(len(model.layers))
#created an object of my_callback class
# used Adam optimizer with learning rate set to 5*10^-5
# loss is set to categorical cross entropy , used mini batch gradient descent , set metrics to 'accuracy'
callbacks=myCallback()
model.compile(Adam(lr=0.00005),loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_datagen,steps_per_epoch=int(train_datagen.samples/train_datagen.batch_size),
                  validation_data=(item for item in valid_datagen),validation_steps=1,
                  epochs=25,verbose=1,callbacks=[callbacks])

model.save('car_classifier_8.h5')
#NOTE: the model is overfitting , i tried using Dropout and l2 kernel regularizer , however with the case of 
# dropout my accuracy decreased , possibly due to information loss as a result of randomly shutting off nodes, with 
#the l2 kernel regularizer my cost increased which increased my training time , it marginally improved my accuracy ,
#wasnt significant.
# Also tried reducing the size of the input image , (224,224)--> (160,160) to reduce overfitting, however again ended 
# up decreasing my accuracy
# plotting the accuracy on train set and validation set
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

# plotting the cost on train set and validation set
plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,2.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
import os 
import cv2
import pandas as pd
from keras.preprocessing import image
import numpy as np

# set test_directory path
test_directory=r'C:\Users\admin\Documents\car_classifier\Test'
path=os.path.join(test_directory)
# used pythons OS library to iterate through the test data 

# since os.listdir(path) randomly iterates through the directory, I sort the all the files in the directory according to the
# image number using " regular expression" "re" library in python
import re
def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(data, key=alphanum_key)

# set dirlist = the sorted directory 
dirlist = sorted_alphanumeric(os.listdir(path))
car_class=[]
images=[]
for img in dirlist:
    img_array=image.load_img(os.path.join(path,img),target_size=(160,160))
    img_array=image.img_to_array(img_array)
    img_array=img_array/255
    img_array=np.expand_dims(img_array,axis=0)
    car=model.predict(img_array).argmax()
    
    # appended the predicted class of car and image no. to an empty list called *car_class* and *images*
    car_class.append(car) 
    images.append(img)

# saved the predictions in an excel file along with the image no. using Pandas Dataframe
predicted_data=pd.DataFrame(np.column_stack([images,car_class]),columns=['image','predictions'])
predicted_data.to_csv('car_predictions14.csv')
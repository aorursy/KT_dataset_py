#Imports

import pandas as pd

import numpy as np

from keras.applications.vgg19 import VGG19
#Importing the dataset into train test and validation directories

train_path='../input/chest-xray-pneumonia/chest_xray/chest_xray/train'

test_path='../input/chest-xray-pneumonia/chest_xray/chest_xray/test'

val_path='../input/chest-xray-pneumonia/chest_xray/chest_xray/val'



IMAGE_SIZE=[224,224]
#Data Preprocessing

from keras.preprocessing.image import ImageDataGenerator

#Train Generator

train_gen=ImageDataGenerator(rescale = 1./255,

                             shear_range = 0.2,

                             zoom_range = 0.2,

                             horizontal_flip = True)

#Test Generator

test_gen=ImageDataGenerator(rescale=1./255)



#Validation Generator

val_gen=ImageDataGenerator(rescale=1./255)
#Generating Training data

train_images=train_gen.flow_from_directory(train_path,

                                           target_size=(224,224),

                                           batch_size=32,

                                           shuffle=True,

                                           class_mode='categorical')
#Generating Testing data

test_images=test_gen.flow_from_directory(test_path,

                                         target_size=(224,224),

                                         batch_size=32,

                                         shuffle=True,

                                         class_mode='categorical')
#Generating Validation data

val_images=val_gen.flow_from_directory(val_path,

                                       target_size=(224,224),

                                       batch_size=32,

                                       

                                       class_mode='categorical')
#Model based imports

from keras.layers import Input, Lambda, Dense, Flatten

from keras.models import Model

from keras.applications.vgg19 import VGG19

from keras.applications.vgg19 import preprocess_input

from keras.preprocessing import image

from keras.models import Sequential

import matplotlib.pyplot as plt

#Add preprocessing layer to the front of VGG

vgg = VGG19(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



#Freeze Current Weights

for layer in vgg.layers:

    layer.trainable = False
#VGG19 model summary

vgg.summary()
#Useful for getting number of classes

from glob import glob

folders = glob(train_path+'/*')

folders
#Adding our layers to the vgg19 model

x = Flatten()(vgg.output)

# x = Dense(1000, activation='relu')(x)

prediction = Dense(len(folders), activation='softmax')(x)



#Create a model object

model = Model(inputs=vgg.input, outputs=prediction)



#Our model Summary

model.summary()
#Compiling the model

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
# Setting CSV Logger

#Store Model Params in log

from keras.callbacks import CSVLogger

csv_logger = CSVLogger('log.csv', separator=',',append=False)
#Fitting the model

hist=model.fit_generator(train_images,

                        steps_per_epoch=150,

                        epochs=20,

                        validation_data=test_images,

                        validation_steps=20,

                        callbacks=[csv_logger])
# Plot loss

plt.plot(hist.history['loss'], label='train loss')

plt.plot(hist.history['val_loss'], label='val loss')

plt.legend()

plt.show()

plt.savefig('Loss-Val Loss')
#Plot accuracies

plt.plot(hist.history['accuracy'], label='train acc')

plt.plot(hist.history['val_accuracy'], label='val acc')

plt.legend()

plt.show()

plt.savefig('Accuracy-Val Accuracy')
# import the modules we'll need to download the log.csv file

from IPython.display import HTML

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv()

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)



# create a random sample dataframe

df = pd.read_csv('log.csv')



# create a link to download the dataframe

create_download_link(df)



# ↓ ↓ ↓  Yay, download link! ↓ ↓ ↓
#Save the model

model.save('vgg19_model.h5')
#Loading Model and predicting for a normal xray

from keras.models import load_model

from keras.preprocessing import image

from keras.applications.vgg16 import preprocess_input

import numpy as np

model = load_model('vgg19_model.h5')

img = image.load_img('../input/chest-xray-pneumonia/chest_xray/chest_xray/val/NORMAL/NORMAL2-IM-1431-0001.jpeg', target_size=(224, 224))

x = image.img_to_array(img)

x = np.expand_dims(x, axis=0)

img_data = preprocess_input(x)

classes = model.predict(img_data)
#Normal Prediction Output

classes
#Predicting for a pneumonia xray

img2 = image.load_img('../input/chest-xray-pneumonia/chest_xray/chest_xray/val/PNEUMONIA/person1954_bacteria_4886.jpeg', target_size=(224, 224))

x = image.img_to_array(img2)

x = np.expand_dims(x, axis=0)

img_data2 = preprocess_input(x)

classes2 = model.predict(img_data2)
#Pneumonia Prediction Output

classes2
#Predicting on Test images 

x,y=test_images.next()

y_true=y.argmax(axis=1)

op=model.predict(x)

preds=op.argmax(axis=1)
#Printing Labels

print('\nTrue Labels:\n',y_true)

print('\nPredicted Labels:\n',preds)

#Different predicted labels:

print('\nDifferent predicted labels:\n')

for i in range(len(preds)):

    if preds[i] != y_true[i]:

        print(i)
#Printing Metrics for model evaluation

from sklearn.metrics import confusion_matrix,classification_report,accuracy_score

cm=confusion_matrix(y_true,preds)

acc=accuracy_score(y_true,preds)

cr=classification_report(y_true,preds)

print('\nAccuracy Score:\n',acc)

print('\nConfusion Matrix:\n',cm)

print('\nClassification Report:\n',cr)
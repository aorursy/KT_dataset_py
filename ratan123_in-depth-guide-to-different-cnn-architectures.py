import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os,cv2

from IPython.display import Image

from keras.preprocessing import image

from keras import optimizers

import numpy as np

from keras import layers,models

from keras.applications.imagenet_utils import preprocess_input

import matplotlib.pyplot as plt

import seaborn as sns

from keras import regularizers

from keras.models import Sequential, Model 

from keras.preprocessing.image import ImageDataGenerator

from keras import applications

print(os.listdir("../input"))



import numpy as np
train_dir="../input/train/train"

test_dir="../input/test/test"

train=pd.read_csv('../input/train.csv')

train.has_cactus=train.has_cactus.astype(str)

df_test=pd.read_csv('../input/sample_submission.csv')
train.head()
train.shape
plt.figure(figsize = (10,6))

sns.countplot(x = 'has_cactus',data = train)

plt.xticks(rotation='vertical')

plt.xlabel('Has cactus or not', fontsize=12)

plt.ylabel('Number of Occurrences', fontsize=12)

plt.show()
print("The number of images in test set is %d"%(len(os.listdir('../input/test/test'))))
Image(os.path.join("../input/train/train",train.iloc[0,0]),width=250,height=250)
datagen=ImageDataGenerator(rescale=1./255)

batch_size=150
train_generator=datagen.flow_from_dataframe(dataframe=train[:15001],directory=train_dir,x_col='id',

                                            y_col='has_cactus',class_mode='binary',batch_size=batch_size,

                                            target_size=(150,150))





validation_generator=datagen.flow_from_dataframe(dataframe=train[15000:],directory=train_dir,x_col='id',

                                                y_col='has_cactus',class_mode='binary',batch_size=50,

                                                target_size=(150,150))

#Reference : https://www.kaggle.com/shahules/getting-started-with-cnn-and-vgg16

model=models.Sequential()

model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(64,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Conv2D(128,(3,3),activation='relu',input_shape=(150,150,3)))

model.add(layers.MaxPool2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512,activation='relu'))

model.add(layers.Dense(1,activation='sigmoid'))

         
model.summary()
model.compile(loss='binary_crossentropy',optimizer=optimizers.adam(),metrics=['acc'])
epochs=5

history=model.fit_generator(train_generator,steps_per_epoch=100,epochs=5,validation_data=validation_generator,validation_steps=50)

acc=history.history['acc']  ##getting  accuracy of each epochs

epochs_=range(0,epochs)    

plt.plot(epochs_,acc,label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val=history.history['val_acc']  ##getting validation accuracy of each epochs

plt.scatter(epochs_,acc_val,label="validation accuracy")

plt.title("no of epochs vs accuracy")

plt.legend()
acc=history.history['loss']    ##getting  loss of each epochs

epochs_=range(0,epochs)

plt.plot(epochs_,acc,label='training loss')

plt.xlabel('No of epochs')

plt.ylabel('loss')



acc_val=history.history['val_loss']  ## getting validation loss of each epochs

plt.scatter(epochs_,acc_val,label="validation loss")

plt.title('no of epochs vs loss')

plt.legend()
model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (150, 150, 3))
model.summary()
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D

for layer in model.layers[:5]:

    layer.trainable = False



#Adding custom Layers 

x = model.output

x = Flatten()(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(1024, activation="relu")(x)

predictions = Dense(1, activation="sigmoid")(x)



# creating the final model 

model_final =  Model(inputs=model.input, outputs=predictions)
model_final.compile(loss='binary_crossentropy',optimizer=optimizers.adam(),metrics=['acc'])
history=model_final.fit_generator(train_generator,steps_per_epoch=100,epochs=5,validation_data=validation_generator,validation_steps=50)
acc=history.history['acc']  ##getting  accuracy of each epochs

epochs_=range(0,epochs)    

plt.plot(epochs_,acc,label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val=history.history['val_acc']  ##getting validation accuracy of each epochs

plt.scatter(epochs_,acc_val,label="validation accuracy")

plt.title("no of epochs vs accuracy")

plt.legend()
acc=history.history['loss']    ##getting  loss of each epochs

epochs_=range(0,epochs)

plt.plot(epochs_,acc,label='training loss')

plt.xlabel('No of epochs')

plt.ylabel('loss')



acc_val=history.history['val_loss']  ## getting validation loss of each epochs

plt.scatter(epochs_,acc_val,label="validation loss")

plt.title('no of epochs vs loss')

plt.legend()
inception_model = applications.inception_v3.InceptionV3(weights = "imagenet", include_top=False, input_shape = (150, 150, 3))
inception_model.summary()
for layer in inception_model.layers[:5]:

    layer.trainable = False



#Adding custom Layers 

x = inception_model.output

x = Flatten()(x)

x = Dense(512, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(1, activation="sigmoid")(x)



# creating the final model 

model_final =  Model(inputs=inception_model.input, outputs=predictions)
model_final.compile(loss='binary_crossentropy',optimizer=optimizers.adam(),metrics=['acc'])
history=model_final.fit_generator(train_generator,steps_per_epoch=100,epochs=5,validation_data=validation_generator,validation_steps=50)
acc=history.history['acc']  ##getting  accuracy of each epochs

epochs_=range(0,epochs)    

plt.plot(epochs_,acc,label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val=history.history['val_acc']  ##getting validation accuracy of each epochs

plt.scatter(epochs_,acc_val,label="validation accuracy")

plt.title("no of epochs vs accuracy")

plt.legend()
acc=history.history['loss']    ##getting  loss of each epochs

epochs_=range(0,epochs)

plt.plot(epochs_,acc,label='training loss')

plt.xlabel('No of epochs')

plt.ylabel('loss')



acc_val=history.history['val_loss']  ## getting validation loss of each epochs

plt.scatter(epochs_,acc_val,label="validation loss")

plt.title('no of epochs vs loss')

plt.legend()
# uncomment this to run

#xception_model = applications.xception.Xception(weights = "imagenet", include_top=False, input_shape = (150, 150, 3))
#uncomment this to run

#xception_model.summary()
#uncomment this to run

'''for layer in xception_model.layers[:5]:

    layer.trainable = False



#Adding custom Layers 

x = xception_model.output

x = Flatten()(x)

x = Dense(1024,activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(1, activation="sigmoid")(x)



# creating the final model 

model_final =  Model(inputs=xception_model.input, outputs=predictions)'''
#model_final.compile(loss='binary_crossentropy',optimizer=optimizers.adam(),metrics=['acc'])
#Uncomment this to run the model

#history=model_final.fit_generator(train_generator,steps_per_epoch=200,epochs=1,validation_data=validation_generator,validation_steps=150)
mobile_model = applications.mobilenet.MobileNet(weights = "imagenet", include_top=False, input_shape = (150, 150, 3))
for layer in mobile_model.layers[:5]:

    layer.trainable = False



#Adding custom Layers 

x = mobile_model.output

x = Flatten()(x)

x = Dense(512, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(1, activation="sigmoid")(x)



# creating the final model 

model_final =  Model(inputs=mobile_model.input, outputs=predictions)

model_final.compile(loss='binary_crossentropy',optimizer=optimizers.adam(),metrics=['acc'])
history=model_final.fit_generator(train_generator,steps_per_epoch=100,epochs=5,validation_data=validation_generator,validation_steps=50)
acc=history.history['acc']  ##getting  accuracy of each epochs

epochs_=range(0,epochs)    

plt.plot(epochs_,acc,label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val=history.history['val_acc']  ##getting validation accuracy of each epochs

plt.scatter(epochs_,acc_val,label="validation accuracy")

plt.title("no of epochs vs accuracy")

plt.legend()
acc=history.history['loss']    ##getting  loss of each epochs

epochs_=range(0,epochs)

plt.plot(epochs_,acc,label='training loss')

plt.xlabel('No of epochs')

plt.ylabel('loss')



acc_val=history.history['val_loss']  ## getting validation loss of each epochs

plt.scatter(epochs_,acc_val,label="validation loss")

plt.title('no of epochs vs loss')

plt.legend()
resnet_model = applications.resnet50.ResNet50(weights = "imagenet", include_top=False, input_shape = (150, 150, 3))
for layer in resnet_model.layers[:5]:

    layer.trainable = False



#Adding custom Layers 

x = resnet_model.output

x = Flatten()(x)

x = Dense(512, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(1024, activation="relu")(x)

x = Dropout(0.5)(x)

predictions = Dense(1, activation="sigmoid")(x)



# creating the final model 

model_final =  Model(inputs=resnet_model.input, outputs=predictions)

model_final.compile(loss='binary_crossentropy',optimizer=optimizers.adam(),metrics=['acc'])
history=model_final.fit_generator(train_generator,steps_per_epoch=100,epochs=5,validation_data=validation_generator,validation_steps=50)
acc=history.history['acc']  ##getting  accuracy of each epochs

epochs_=range(0,epochs)    

plt.plot(epochs_,acc,label='training accuracy')

plt.xlabel('no of epochs')

plt.ylabel('accuracy')



acc_val=history.history['val_acc']  ##getting validation accuracy of each epochs

plt.scatter(epochs_,acc_val,label="validation accuracy")

plt.title("no of epochs vs accuracy")

plt.legend()
acc=history.history['loss']    ##getting  loss of each epochs

epochs_=range(0,epochs)

plt.plot(epochs_,acc,label='training loss')

plt.xlabel('No of epochs')

plt.ylabel('loss')



acc_val=history.history['val_loss']  ## getting validation loss of each epochs

plt.scatter(epochs_,acc_val,label="validation loss")

plt.title('no of epochs vs loss')

plt.legend()
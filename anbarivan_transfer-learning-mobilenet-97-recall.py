import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import keras

import matplotlib.pyplot as plt

from keras.layers import Dense,GlobalAveragePooling2D

from keras.applications import MobileNet

from keras.preprocessing import image

from keras.applications.mobilenet import preprocess_input

from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model

from keras.optimizers import Adam

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

print(os.listdir("../input/chest_xray/chest_xray"))

# Any results you write to the current directory are saved as output.
#preproessing-img aumentation----avoid overfitting

from keras.preprocessing.image import ImageDataGenerator





train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies



train_generator=train_datagen.flow_from_directory('../input/chest_xray/chest_xray/train', # this is where you specify the path to the main data folder

                                                 target_size=(224,224),

                                                 color_mode='rgb',

                                                 batch_size=30,

                                                 class_mode='categorical',

                                                 shuffle=True)

test_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies



test_generator=test_datagen.flow_from_directory('../input/chest_xray/chest_xray/test', # this is where you specify the path to the main data folder

                                                 target_size=(224,224),

                                                 color_mode='rgb',

                                                 batch_size=30,

                                                 class_mode='categorical',

                                                 shuffle=True)
import random

import os

import matplotlib.pyplot as plt

normal_img = os.listdir('../input/chest_xray/chest_xray/train/NORMAL/')

random = random.sample(normal_img,25)

f,ax = plt.subplots(5,5)



for i in range(0,25):

    im = plt.imread('../input/chest_xray/chest_xray/train/NORMAL/'+random[i])

    ax[i//5,i%5].imshow(im)

    ax[i//5,i%5].axis('off')

f.suptitle('Normal Lungs')

plt.show()
from keras.applications.mobilenet import MobileNet

from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization

from keras.models import Sequential



base_model=MobileNet( weights='imagenet',include_top=False,input_shape=(224,224,3)) #imports the mobilenet model and discards the last 1000 neuron layer.


x=base_model.output

x=GlobalAveragePooling2D()(x)



x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.

x=Dense(1024,activation='relu')(x) #dense layer 2

x=Dense(512,activation='relu')(x) #dense layer 3



preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
model=Model(inputs=base_model.input,outputs=preds)

for layer in model.layers[:20]:

    layer.trainable=False

for layer in model.layers[20:]:

    layer.trainable=True
from keras.optimizers import Adam

from keras.callbacks import  EarlyStopping

Adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-5, amsgrad=False)

es = EarlyStopping(patience=5)

Adam
model.compile(optimizer=Adam,loss='categorical_crossentropy',metrics=['accuracy'])

# Adam optimizer

# loss function will be categorical cross entropy

# evaluation metric will be accuracy

model.summary()
step_size_train=train_generator.n//train_generator.batch_size

step_size_test=test_generator.n//test_generator.batch_size

#viszualization



img_path='../input/chest_xray/chest_xray/val/PNEUMONIA/person1946_bacteria_4874.jpeg'

img = image.load_img(img_path,target_size=(224,224))





img_tensor = image.img_to_array(img)

img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.

plt.imshow(img_tensor[0])

plt.show()

print(img_tensor.shape)

history=model.fit_generator(generator=train_generator,steps_per_epoch=step_size_train,

                            validation_data=test_generator,validation_steps=step_size_test,callbacks=[es],epochs=15)
#val accuracy

score  = model.evaluate_generator(test_generator, steps=len(test_generator), verbose=1)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
predgen = model.predict_generator(test_generator, steps=len(test_generator), verbose=1) 

pred = np.argmax(predgen,axis = 1)

y_true=test_generator.classes
#confution matrix



from sklearn.metrics import confusion_matrix

CM = confusion_matrix(y_true, pred)

from mlxtend.plotting import plot_confusion_matrix



fig, ax = plot_confusion_matrix(conf_mat=CM ,  figsize=(5, 5))

plt.show()
# Calculate Precision and Recall

tn, fp, fn, tp = CM.ravel()



precision = tp/(tp+fp)

recall = tp/(tp+fn)



print("Recall of the model is {:.2f}".format(recall))

print("Precision of the model is {:.2f}".format(precision))
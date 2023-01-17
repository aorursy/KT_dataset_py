train_dir = '../input/aicup2020-mango-c1p1-datagen/Mango/train'

valid_dir = '../input/aicup2020-mango-c1p1-datagen/Mango/dev'

test_dir  = '../input/aicup2020-mango-c1p1-datagen/Mango/test'
import os

import numpy as np

import pandas as pd

trainDF = pd.read_csv('/kaggle/input/aicup2020-mango-c1p1-datagen/Mango/train.csv', header=None)
trainFiles = trainDF[0].tolist()

trainClasses = trainDF[1].tolist()
valDF = pd.read_csv('/kaggle/input/aicup2020-mango-c1p1-datagen/Mango/dev.csv', header=None)
valFiles = valDF[0].tolist()

valClasses = valDF[1].tolist()
testFiles = os.listdir(test_dir+'/unknown')
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
plt.imshow(mpimg.imread(train_dir+'/'+trainClasses[0]+'/'+trainFiles[0]))
plt.imshow(mpimg.imread(valid_dir+'/'+valClasses[0]+'/'+valFiles[0]))
labels = ['A', 'B', 'C']
# plot the circle of value counts in dataset

def plot_equilibre(equilibre, labels, title):

    plt.figure(figsize=(5,5))

    my_circle=plt.Circle( (0,0), 0.5, color='white')

    plt.pie(equilibre, labels=labels, colors=['red','green','blue'],autopct='%1.1f%%')

    p=plt.gcf()

    p.gca().add_artist(my_circle)

    plt.title(title)

    plt.show()
equilibreTrain = []

[equilibreTrain.append(trainClasses.count(label)) for label in labels]

print(equilibreTrain)

plot_equilibre(equilibreTrain, labels, 'Train Data')

del equilibreTrain
equilibreDev = []

[equilibreDev.append(valClasses.count(label)) for label in labels]

print(equilibreDev)

plot_equilibre(equilibreDev, labels, 'Development Data')

del equilibreDev
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

from tensorflow.keras.utils import to_categorical
target_size=(224,224)

batch_size = 16
train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=True)



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',    

    shuffle=True,

    seed=42,

    class_mode='categorical')

valid_datagen = ImageDataGenerator(rescale=1./255)



valid_generator = valid_datagen.flow_from_directory(

    valid_dir,

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',

    shuffle=False,    

    class_mode='categorical')
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(

    test_dir,

    target_size=target_size,

    batch_size=batch_size,

    color_mode='rgb',

    shuffle=False,     

    class_mode='categorical')
import tensorflow as tf

import tensorflow.keras as keras

from tensorflow.keras.models import Model, save_model

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

from tensorflow.keras.layers import Input, BatchNormalization, Activation, LeakyReLU, Concatenate

from tensorflow.keras.regularizers import l2

from tensorflow.keras.callbacks import ModelCheckpoint
num_classes = 3

input_shape = (224,224,3)
!pip install -q efficientnet

import efficientnet.tfkeras as efn
# load EfficientNetB7 model with imagenet parameteres

base_model = efn.EfficientNetB7(input_shape=input_shape, weights='imagenet', include_top=False)
# freeze the base model (for transfer learning)

base_model.trainable = False
x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(512)(x)

x = Dense(32)(x)

out = Dense(num_classes, activation="softmax")(x)



model = Model(inputs=base_model.input, outputs=out)

model.summary()
# Compile Model

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
## set Checkpoint : save best only, verbose on

#checkpoint = ModelCheckpoint("mango_classification.hdf5", monitor='accuracy', verbose=0, save_best_only=True, mode='auto', save_freq=1)
STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

STEP_SIZE_TEST =test_generator.n//test_generator.batch_size

num_epochs =50
# Train Model

history = model.fit_generator(train_generator,steps_per_epoch=STEP_SIZE_TRAIN,epochs=num_epochs, validation_data=valid_generator, validation_steps=STEP_SIZE_VALID) #, callbacks=[checkpoint])
## Save Model

save_model(model, 'mango_efficientnetB7.h5')
## load best model weights if using callback (save-best-only)

#model.load_weights("mango_classification.hdf5")
score = model.evaluate_generator(test_generator, steps=STEP_SIZE_TEST)

print(score)
from sklearn.metrics import classification_report, confusion_matrix

predY=model.predict_generator(test_generator)

y_pred = np.argmax(predY,axis=1)

#y_label= [labels[k] for k in y_pred]

y_actual = test_generator.classes

cm = confusion_matrix(y_actual, y_pred)

print(cm)
print(classification_report(y_actual, y_pred, target_names=labels))
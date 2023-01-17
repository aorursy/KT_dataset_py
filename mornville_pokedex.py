import numpy as np 
import pandas as pd
import tensorflow as tf
import tensorflow.keras
import sys
from matplotlib import pyplot
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization, Activation
import tensorflow
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import numpy as np


def define_model():
    # load model
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    # mark loaded layers as not trainable
    for layer in model.layers:
        layer.trainable = False
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(1024, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(166, activation='softmax')(class1)
    # define new model
    model = Model(inputs=model.inputs, outputs=output)
    # compile model
    opt = SGD(lr=0.01)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model
''' For saving .png image for history of plots. '''

def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
	# save plot to file
	filename = sys.argv[0].split('/')[-1]
	pyplot.savefig(filename + '_plot.png')
	pyplot.close()
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
# # specify imagenet mean values for centering
# train_datagen.mean = [123.68, 116.779, 103.939]
# test_datagen.mean = [123.68, 116.779, 103.939]

# prepare iterator
train_it = train_datagen.flow_from_directory('/kaggle/input/pokemonimagedataset/dataset/train/',
    class_mode='categorical', batch_size=64, target_size=(224, 224))
test_it = test_datagen.flow_from_directory('/kaggle/input/pokemonimagedataset/dataset/test/',
    class_mode='categorical', batch_size=32, target_size=(224, 224))
model = define_model()
model.summary()

''' Fitting the model '''
history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
        validation_data=test_it, validation_steps=len(test_it), epochs=5, verbose=1)

# evaluating model
_, acc = model.evaluate_generator(test_it, steps=len(test_it), verbose=0)
print('> %.3f' % (acc * 100.0))
model.save('VGG16_accuracy - %.3f' % (acc * 100.0) + '.h5')

# learning curves
summarize_diagnostics(history)

''' Loading the above model after fitting '''
from tensorflow.keras.models import load_model
import tensorflow as tf

loadedModel = load_model('/kaggle/input/accuracy-55530h5/accuracy - 55.530.h5')

allClasses = train_it.class_indices
print("Total Classes: " + str(len(list(allClasses))))
''' Testing the trained VGG16 model on test dataset '''


import cv2, os

perc = []
for pokemon in os.listdir('/kaggle/input/pokemonimagedataset/dataset/test/'):
    count = 0
    wrongCount = 0
    # Taking Onix as an example Pokemon
    if(pokemon == 'Onix'):    
        for i in os.listdir('/kaggle/input/pokemonimagedataset/dataset/test/'+ pokemon + '/'):
            img = cv2.imread('/kaggle/input/pokemonimagedataset/dataset/test/'+ pokemon + '/' +  i)
            img = cv2.resize(img, (224, 224))
            pred = loadedModel.predict(np.array([img]))
            move_code = np.argmax(pred[0])
            predictedPokemon = list(allClasses.keys())[list(allClasses.values()).index(move_code)]
            if(predictedPokemon == pokemon):
                count+=1
            else:
                wrongCount+=1
#             print(str(move_code) + '  ' +  str(predictedPokemon))
        print("Correct prediction : ", count)
        print("Wrong  prediction : ", wrongCount)
        print('Percentage: ' + str( (count/(wrongCount+count))*100) + ', For ' + pokemon)
        perc.append((count/(wrongCount+count))*100)

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D
from tensorflow.keras.applications.mobilenet import preprocess_input

base_model=MobileNet(weights='imagenet',include_top=False) #imports the mobilenet model and discards the last 1000 neuron layer.

x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dropout(0.2)(x)
# x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
# x=Dense(512,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
# x=Dropout(0.2)(x)
x=Dense(256,activation='tanh')(x) #dense layer 2
# x=Dense(256,activation='tanh')(x)
#dense layer 3
preds=Dense(166,activation='softmax')(x) #final layer with softmax activation

Testmodel=Model(inputs=base_model.input,outputs=preds)
Testmodel.summary()
for layer in Testmodel.layers[:20]:
    layer.trainable=False
for layer in Testmodel.layers[20:]:
    layer.trainable=True
Testmodel.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

test_datagen = ImageDataGenerator(rescale=1.0/255.0)
# # specify imagenet mean values for centering
# train_datagen.mean = [123.68, 116.779, 103.939]
# test_datagen.mean = [123.68, 116.779, 103.939]


train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies

train_generator=train_datagen.flow_from_directory('/kaggle/input/pokemonimagedataset/dataset/train/', # this is where you specify the path to the main data folder
                                                 target_size=(100,100),
                                                 color_mode='rgb',
                                                 batch_size=100,
                                                 class_mode='categorical',
                                                 shuffle=True)
test_it = test_datagen.flow_from_directory('/kaggle/input/pokemonimagedataset/dataset/test/',
    class_mode='categorical', batch_size=50, target_size=(100, 100))
# history = Testmodel.fit_generator(train_generator, steps_per_epoch=len(train_generator),
#         validation_data=test_it, validation_steps=len(test_it), epochs=10, verbose=1)
# # evaluate model
# _, acc = Testmodel.evaluate_generator(test_it, steps=len(test_it), verbose=1)
# print('> %.3f' % (acc * 100.0))
# Testmodel.save('MobileNet_accuracy - %.3f' % (acc * 100.0) + '.h5')
# # learning curves
# summarize_diagnostics(history)
''' Loading the above model after fitting '''
from tensorflow.keras.models import load_model
import tensorflow as tf

Testmodel = load_model('/kaggle/input/mobilenetmodel/MobileNet-63.237.h5')
''' Testing the trained MobileNet model on test dataset '''


import cv2, os

perc = []
for pokemon in os.listdir('/kaggle/input/pokemonimagedataset/dataset/test/'):
    count = 0
    wrongCount = 0
    # Taking Caterpie as an example Pokemon
    if(pokemon == 'Caterpie'):    
        for i in os.listdir('/kaggle/input/pokemonimagedataset/dataset/test/'+ pokemon + '/'):
            img = cv2.imread('/kaggle/input/pokemonimagedataset/dataset/test/'+ pokemon + '/' +  i)
            img = cv2.resize(img, (224, 224))
            pred = Testmodel.predict(np.array([img]))
            move_code = np.argmax(pred[0])
            predictedPokemon = list(allClasses.keys())[list(allClasses.values()).index(move_code)]
            if(predictedPokemon == pokemon):
                count+=1
            else:
                wrongCount+=1
    #             print(str(move_code) + '  ' +  str(predictedPokemon))
        print("Correct prediction : ", count)
        print("Wrong  prediction : ", wrongCount)
        print('Percentage: ' + str( (count/(wrongCount+count))*100) + ', For ' + pokemon)
        perc.append((count/(wrongCount+count))*100)

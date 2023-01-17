# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import tensorflow as tf 



import matplotlib



matplotlib.use("Agg")

from keras.applications import VGG16

from keras.layers import Dropout, Flatten, Dense, Input, Conv2D, Activation, MaxPooling2D, BatchNormalization, concatenate, GlobalAveragePooling2D, GlobalMaxPooling2D

from keras.models import Sequential

from keras.models import Model

from keras.optimizers import SGD

from sklearn.metrics import classification_report

#from configutils import config

#from imutils import paths

import matplotlib.pyplot as plt

import numpy as np

import os
def layer(input) :

	j = BatchNormalization()(input)

	j = Activation('relu')(j)

	j = Conv2D(12, (3, 3), activation = 'relu', padding = 'same')(j)

	j = Dropout(0.2)(j)

	return(j)



def denseBlock(input) :

	a = layer(input)

	b = concatenate([input, a], axis = -1)

	c = layer(b)

	d = concatenate([b, c], axis = -1)

	e = layer(d)

	f = concatenate([d, e], axis = -1)

	g = layer(f)

	h = concatenate([a, c, e, g], axis = -1)

	return(h)
BA = 48



from keras.preprocessing.image import ImageDataGenerator, train_generator, validation_generator



# create a data generator

traingen = ImageDataGenerator(horizontal_flip = True, rescale = 1.0 / 255, fill_mode = "nearest", zoom_range = 0.2, shear_range = 0.2)

testgen = ImageDataGenerator(rescale = 1.0 / 255, fill_mode = "nearest", zoom_range = 0.2, shear_range = 0.2)

    

train_it = traingen.flow_from_directory('../input/AFF11_resized/AFF11_resized_test/', class_mode = 'categorical', batch_size = BA, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)

test_it = testgen.flow_from_directory('../input/AFF11_resized/AFF11_resized_train/', class_mode = 'categorical', batch_size = BA, target_size = (224, 224) ,color_mode = "rgb", shuffle = True)



X_train, y_train = train_generator.next()

X_test, y_test = validation_generator.next()
# Conv

inputs = Input((224, 224, 3))

i = Conv2D(48, (3, 3), activation = 'relu', padding = 'same')(inputs)



#### DB

j = denseBlock(i)



# TD

j = BatchNormalization()(j)

j = Activation('relu')(j)

j = Conv2D(12, (1, 1), activation = 'relu')(j)

j = Dropout(0.2)(j)

j = MaxPooling2D(pool_size = 2 * 2)(j)



#### DB

j = denseBlock(j)



# FC

j = Flatten()(j)

#j = GlobalMaxPooling2D()(j)

outputs = Dense(11, activation = 'softmax')(j) # 11 classes dans le dataset



# Compilation du modèle

model = Model(inputs = inputs, outputs = outputs)

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
# Entraînement du modèle

history = model.fit_generator(train_it, epochs = 10, steps_per_epoch = 593//BA)

loss = model.evaluate_generator(test_it)

print("Evalutaion du modele :", loss)

print("Evalutaion du modele -- loss :", loss[0], "accuracy :", loss[1]*100, "%")
# fine tuning (ou évaluation du model)

# --> impossible à faire avec ImageDataGenerator --> en fait si, apparemment, faut juste ne pas transformer les images dans le imagedatagenerator...

# sauvegarde du model

json = model.to_json()

with open("save.json", "w") as file:

    file.write(json)

    

# sauvegarde de l'historique d'apprentissage du model

hist_df = pd.DataFrame(history.history) 

hist_json_file = 'history.json' 

with open(hist_json_file, mode='w') as f:

    hist_df.to_json(f)

    

# 

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

y_pred1 = model.predict(X_test)

y_pred = np.argmax(y_pred1, axis=1)



# Print f1, precision, and recall scores

print(precision_score(y_test, y_pred , average="macro"))

print(recall_score(y_test, y_pred , average="macro"))

print(f1_score(y_test, y_pred , average="macro"))
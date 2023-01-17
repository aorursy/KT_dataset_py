import numpy as np

import pandas as pd 



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

    #for filename in filenames:

        #print(os.path.join(dirname, filename))

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Convolution2D

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.layers import Flatten

from tensorflow.keras.layers import Dense

from tensorflow.keras.layers import Dropout, Activation , BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from tensorflow.keras import metrics



from sklearn.utils import class_weight

from collections import Counter



import matplotlib.pyplot as plt



import os

from os import listdir

from os.path import isfile, join
train_loc = '../input/Thesis3/train'

print(os.listdir(train_loc))
train_loc = '../input/Thesis3/train'

val_loc = '../input/Thesis3/validation'

test_loc = '../input/Thesis3/test'
droprate=0.1

filter_pixel=3

input_shape = (505, 300, 3)

classifier = Sequential()
#convolution 1st layer

classifier.add(Convolution2D(32, kernel_size=(filter_pixel, filter_pixel), input_shape = input_shape, activation = 'relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(droprate))



#convolution 2nd layer

classifier.add(Convolution2D(64, kernel_size=(filter_pixel, filter_pixel), activation='relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(droprate))



#convolution 3rd layer

classifier.add(Convolution2D(128, kernel_size=(filter_pixel, filter_pixel), activation='relu'))

classifier.add(BatchNormalization())

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Dropout(droprate))



#Fully connected 1st layer

classifier.add(Flatten()) 

classifier.add(Dense(1024)) 

classifier.add(Activation('relu')) 



classifier.add(Dense(1024)) 

classifier.add(Activation('relu')) 



#Fully connected final layer

classifier.add(Dense(1))

classifier.add(Activation('sigmoid'))



# Compiling the CNN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])



classifier.summary()
train_datagen = ImageDataGenerator(rescale = 1./255,

                                   shear_range = 0.0,

                                   zoom_range = 0.2,

                                   horizontal_flip = False)



test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory(train_loc,

                                                 target_size = (300, 505),

                                                 batch_size = 32,

                                                 class_mode = 'binary')



val_set = test_datagen.flow_from_directory(val_loc,

                                            target_size = (300, 505),

                                            batch_size = 32,

                                            class_mode = 'binary')
classifier.fit_generator(training_set,

                         steps_per_epoch=training_set.samples//training_set.batch_size, 

                         validation_data=val_set,

                         validation_steps=val_set.samples//val_set.batch_size,

                         epochs=30)

classifier.save('DR_CNN2.h5') 
test_datagene = ImageDataGenerator()

test_sets = test_datagen.flow_from_directory(test_loc,target_size = (300, 505))

X_test, y_test = test_sets.next()
score = classifier.evaluate(X_test, y_test, verbose=0)

print('Test loss:', score[0])

print('Test accuracy:', score[1])
predictions = classifier.predict(X_test)

print('First prediction:', predictions[0])
from sklearn.metrics import confusion_matrix, classification_report

import numpy as np



prediction = classifier.predict(X_test)

test_result = np.argmax(y_test, axis=1)

prediction_result = np.argmax(prediction, axis=1)

confusion__matrix=confusion_matrix(test_result, prediction_result)

print(classification_report(test_result, prediction_result))

print(confusion__matrix)
# Plot confusion matrix

from sklearn.metrics import plot_confusion_matrix

title='yes no'

class_names=['0','1','2','3']

titles_options = [("Confusion matrix, without normalization", None),

                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:

    disp = plot_confusion_matrix(classifier, X_test, y_test,

                                 display_labels=class_names,

                                 cmap=plt.cm.Blues,

                                 normalize=normalize)

    disp.ax_.set_title(title)

    print(title)

    print(disp.confusion_matrix)

plt.show()
import seaborn as sn

import pandas as pd

import matplotlib.pyplot as plt



df_cm = pd.DataFrame(confusion__matrix, range(5), range(5))

# plt.figure(figsize=(10,7))

sn.set(font_scale=1.4)

sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})



plt.show()
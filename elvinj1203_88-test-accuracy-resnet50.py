import pandas as pd

import numpy as np

from keras.models import Sequential

from keras import optimizers

from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D

from keras.applications import ResNet50
base_model= ResNet50(include_top=False, weights="imagenet", input_shape=(224,224,3))
model= Sequential()

model.add(base_model)

model.add(Conv2D(64, (3, 3), activation = 'relu'))

#model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.40))

model.add(Conv2D(64, (3, 3), activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Dropout(0.40))

model.add(Flatten())

model.add(Dense(512,activation='relu'))

model.add(Dropout(0.4))

model.add(Dense(2, activation='softmax'))
from keras import optimizers

from keras_preprocessing.image import ImageDataGenerator

train_generator = ImageDataGenerator(rescale = 1./255,

                                     rotation_range=10,  

                                     zoom_range = 0.1, 

                                     width_shift_range=0.1,  height_shift_range=0.1) 

test_generator = ImageDataGenerator(rescale = 1./255)
training_set = train_generator.flow_from_directory('../input/data/train',

                                                 target_size = (224,224),

                                                 batch_size = 64,

                                                 class_mode = 'categorical')



test_set = test_generator.flow_from_directory('../input/data/test',

                                            target_size = (224, 224),

                                            batch_size = 64,

                                            class_mode = 'categorical',

                                            shuffle=False)

model.compile(optimizer=optimizers.adam(lr=0.0001),loss="categorical_crossentropy",metrics=["accuracy"])
from keras.callbacks import ReduceLROnPlateau

learn_control = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=.5, min_lr=0.0001)
model.fit_generator(generator=training_set,

                            steps_per_epoch=training_set.samples//training_set.batch_size,

                            validation_data=test_set,

                            verbose=1,

                            validation_steps=test_set.samples//test_set.batch_size,

                            epochs=27,callbacks=[learn_control])
test_set.reset()

predictions = model.predict_generator(test_set, steps=test_set.samples/test_set.batch_size,verbose=1)

y_pred= np.argmax(predictions, axis=1)



print(y_pred)
y_test=np.array([])

for i in range(360):

    y_test=np.append(y_test,0)

for i in range(300):

    y_test=np.append(y_test,1)
from sklearn.metrics import confusion_matrix 

cm= confusion_matrix(y_test,y_pred)
print(cm)
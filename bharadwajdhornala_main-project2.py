train_path="../input/new_dataset2/new_dataset2/new_train"

valid_path="../input/new_dataset2/new_dataset2/new_valid"

test_path="../input/new_dataset2/new_dataset2/new_test"
from keras.preprocessing import image

from keras.layers import Dense,Conv2D,Flatten,Dropout,BatchNormalization,MaxPooling2D,GlobalAveragePooling2D,Activation

from keras.models import Sequential

train_datagen = image.ImageDataGenerator(rotation_range=10,

                                    width_shift_range=0.05,

                                    height_shift_range=0.05,

                                    shear_range=0.05,

                                    zoom_range=0.2,

                                    rescale=1./255,

                                    fill_mode='nearest',

                                    channel_shift_range=0.2*255)

validate_datagen  = image.ImageDataGenerator(rescale=1./255)

test_datagen  = image.ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(train_path,target_size=(300,300),batch_size=32,class_mode='categorical')

valid_generator = validate_datagen.flow_from_directory(valid_path,target_size=(300,300),batch_size=32,class_mode='categorical')

test_generator= test_datagen.flow_from_directory(test_path,target_size=(300,300),batch_size=1,class_mode='categorical')
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)



from keras.applications.vgg16 import VGG16,preprocess_input

from keras.preprocessing import image

from keras.layers import Input, Flatten, Dense ,Dropout,BatchNormalization

from keras.models import Model

from keras import regularizers





my_model = VGG16( weights= 'imagenet',

                 include_top=False )



input = Input(shape=(300,300,3),name = 'image_input')

output = my_model(input)



for layer in my_model.layers:

    layer.trainable=False



x = Flatten(name='flatten')(output)

x = Dense(4096, activation='relu', name='fc1',kernel_regularizer=regularizers.l2(0.0001))(x)

x = Dropout(0.5)(x)

x = BatchNormalization()(x)

x = Dense(256, activation='relu', name='fc2',kernel_regularizer=regularizers.l2(0.0001))(x)

x = Dropout(0.5)(x)

x = Dense(128, activation='relu', name='fc3',kernel_regularizer=regularizers.l2(0.0001))(x)

x = Dropout(0.5)(x)

x = BatchNormalization()(x)

x = Dense(2, activation='softmax', name='predictions')(x)



my_model = Model(input=input, output=x)

image_size=(300,300,3)



my_model.summary()



from keras.callbacks import EarlyStopping,ModelCheckpoint

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

mc = ModelCheckpoint('best_model5.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

callback_list=[es,mc]
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)





from keras import optimizers



my_model.compile(loss="categorical_crossentropy", optimizer='adam',metrics=["accuracy"])



history=my_model.fit_generator(train_generator,steps_per_epoch=train_generator.samples//train_generator.batch_size,epochs=30,

        validation_data = valid_generator,

        validation_steps = valid_generator.samples//valid_generator.batch_size ,

                              callbacks=callback_list )
    

scores=my_model.evaluate_generator(test_generator,steps=test_generator.samples//test_generator.batch_size)

print("\n%s: %.2f%%"  % (my_model.metrics_names[1],scores[1]*100) )
import matplotlib.pyplot as plt

train_acc=history.history['acc']

val_acc=history.history['val_acc']

train_loss=history.history['loss']

val_loss=history.history['val_loss']

epochs=range(len(train_acc))

plt.plot(epochs,train_acc,'b',label='Training Accuracy')

plt.plot(epochs,val_acc,'r',label='Validation Accuracy')

plt.title('Training and Validation Accuracy')

plt.legend()

plt.figure()

plt.show()



plt.plot(epochs,train_loss,'b',label='Training Loss')

plt.plot(epochs,val_loss,'r',label='Validation Loss')

plt.title('Training and Validation Loss ')

plt.legend()

plt.figure()

plt.show()

import numpy as np

pred=my_model.predict_generator(test_generator,verbose=1,steps= test_generator.samples//test_generator.batch_size)

predictedClasses = np.argmax(pred, axis=1)
labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predictedClasses]
import pandas as pd

filenames=test_generator.filenames

results=pd.DataFrame({"Filename":filenames,

                      "Predictions":predictions})

results
from sklearn import metrics

print('Confusion Matrix')

cm = metrics.confusion_matrix(test_generator.classes, predictedClasses)

print(cm)

print('Classification Report')

print(metrics.classification_report(test_generator.classes, predictedClasses)) 
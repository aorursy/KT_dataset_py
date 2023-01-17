import os, cv2, re, random
import numpy as np
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
from keras import layers, models, optimizers

train_dir = r'C:\Users\Prashant\Desktop\seg_train\seg_train'
test_dir = r'C:\Users\Prashant\Desktop\seg_test\seg_test'
pred_dir = r'C:\Users\Prashant\Desktop\seg_pred'
labels = sorted(os.listdir(train_dir))
labels

datagen = ImageDataGenerator(rescale=1./255,
                           shear_range = 0.2,
                           zoom_range = 0.2,
                           horizontal_flip=True,
                           validation_split=0.2)
train_generator = datagen.flow_from_directory(
                        train_dir,
                        shuffle=False,
                        class_mode="categorical",
                        target_size=(150,150),
                        batch_size = 16,
                        subset='training')
valid_generator = datagen.flow_from_directory(
                        train_dir,
                        shuffle=False,
                        class_mode="categorical",
                        target_size=(150,150),
                        batch_size = 16,
                        subset='validation')

tst_aug = ImageDataGenerator(rescale=1./255)
test_gen = tst_aug.flow_from_directory(
                        test_dir,
                        shuffle=False,
                        class_mode="categorical",
                        target_size=(150,150),
                        batch_size = 16)



from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense,Dropout,Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.callbacks import ReduceLROnPlateau
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import tensorflow as tf

inceptionV3 = InceptionV3(include_top= False, input_shape=(150,150,3))

for layer in inceptionV3.layers:
    layer.trainable = False
last_layer = inceptionV3.get_layer('mixed9')

print('last layer output shape: ', last_layer.output_shape)

last_output = last_layer.output
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(units = 1024, activation = tf.nn.relu)(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense  (6, activation = tf.nn.softmax)(x)

model = tf.keras.Model( inceptionV3.input, x)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=1,
                                            verbose=1,
                                            factor=0.25,
                                            min_lr=0.000003)

model.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics=['accuracy'])
model.fit_generator(
    train_generator,
    epochs=30,
    validation_data = valid_generator, 
    callbacks=EarlyStopping(monitor='val_loss', mode='min',verbose=1))
model.evaluate(test_gen)
pred_aug = ImageDataGenerator(rescale=1./255)
pred_gen = pred_aug.flow_from_directory(
                        pred_dir,
                        shuffle=False,
                        class_mode="categorical",
                        target_size=(150,150),
                        batch_size = 16)

results = model.predict(pred_gen)
results
import keras
model.save('imgintelclass.model')
predicted_class_indices=np.argmax(results,axis=1)

predicted_class_indices
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
filenames=pred_gen.filenames
res=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
res








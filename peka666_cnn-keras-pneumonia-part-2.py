train_dir = "../input/chest_xray/chest_xray/train"

val_dir = "../input/chest_xray/chest_xray/val"

test_dir = "../input/chest_xray/chest_xray/test"
from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(

    rescale=1./255,

    rotation_range=40, 

    width_shift_range=0.2, 

    height_shift_range=0.2, 

    shear_range=0.2, 

    zoom_range=0.2,

    horizontal_flip=True,)



test_datagen = ImageDataGenerator(rescale=1./255)



train_generator = train_datagen.flow_from_directory(

    train_dir,

    target_size=(150, 150),

    batch_size=100,

    class_mode='binary')



val_generator = test_datagen.flow_from_directory(

    val_dir,

    target_size=(150, 150),

    batch_size=16,

    class_mode='binary')

    
from keras import layers 

from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation = 'relu', input_shape=(150,150,3)))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(64,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Conv2D(128,(3,3), activation = 'relu'))

model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation = 'relu'))

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation = 'sigmoid'))



model.summary()
from keras import optimizers



model.compile(loss='binary_crossentropy',

              optimizer= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

,metrics=['acc'])


history = model.fit_generator(train_generator,

steps_per_epoch=10,

epochs=20, validation_data=val_generator, validation_steps=20)
from keras.applications import VGG16
conv_basic = VGG16(weights='imagenet', include_top=False)
conv_basic.summary()
import numpy as np



datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20 

def extract_features(directory, sample_count):

    

    features = np.zeros(shape=(sample_count, 4, 4, 512)) 

    labels = np.zeros(shape=(sample_count))



    generator = datagen.flow_from_directory(

        directory, 

        target_size=(150, 150), 

        batch_size=batch_size, 

        class_mode='binary')

    

    i = 0

    

    for inputs_batch, labels_batch in generator:

        features_batch = conv_basic.predict(inputs_batch)

        features[i * batch_size : (i + 1) * batch_size] = features_batch 

        labels[i * batch_size : (i + 1) * batch_size] = labels_batch

        i += 1

        if i * batch_size >= sample_count:

            break

    return features, labels
train_features, train_labels = extract_features(train_dir, 2000) 

val_features, validation_labels = extract_features(val_dir, 16) 

test_features, test_labels = extract_features(test_dir, 624)
train_features = np.reshape(train_features, (2000, 4 * 4 * 512)) 

validation_features = np.reshape(val_features, (16, 4 * 4 * 512)) 

test_features = np.reshape(test_features, (624, 4 * 4 * 512))

from keras import optimizers



model = models.Sequential()

model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512)) 

model.add(layers.Dropout(0.5))

model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer= optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), 

              loss='binary_crossentropy',metrics=['acc'])



history = model.fit(train_features, train_labels, epochs=30, batch_size=20, 

                    validation_data=(validation_features, validation_labels))
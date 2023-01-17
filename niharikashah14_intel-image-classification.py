from tensorflow.keras.layers import Input, Lambda, Conv2D, Dense, Flatten

from tensorflow.keras.models import Sequential, Model

from glob import glob

from tensorflow.keras.layers import MaxPooling2D

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np

from tensorflow.keras.models import load_model

from tensorflow.keras.models import load_model

from tensorflow.keras.preprocessing import image

model = Sequential()

model.add(Conv2D(filters=16, kernel_size=2, padding="same", activation="relu", input_shape=(150,150,3)))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))

model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())

model.add(Dense(100,activation="relu"))

model.add(Dense(6,activation="softmax"))

model.summary()
model.compile(

  loss='categorical_crossentropy',

  optimizer='adam',

  metrics=['accuracy']

)
train_datagen = ImageDataGenerator(rescale = 1./255)



test_datagen = ImageDataGenerator(rescale = 1./255)
training_set = train_datagen.flow_from_directory('../input/intel-image-classification/seg_train/seg_train',

                                                 target_size = (150, 150),

                                                 batch_size = 32,

                                                 class_mode = 'categorical')
test_set = train_datagen.flow_from_directory('../input/intel-image-classification/seg_test/seg_test',

                                                 target_size = (150, 150),

                                                 batch_size = 32,

                                                 class_mode = 'categorical')
model_fit = model.fit_generator(

  training_set,

  validation_data=test_set,

  epochs=10,

  steps_per_epoch=len(training_set),

  validation_steps=len(test_set)

)
model.save('model.h5')
y_pred = model.predict(test_set)
y_pred
y_pred = np.argmax(y_pred,axis=1)

y_pred
model=load_model('model.h5')

img=image.load_img('../input/intel-image-classification/seg_pred/seg_pred/10875.jpg',target_size=(150,150))

x=image.img_to_array(img)

x=np.expand_dims(x,axis=0)

#img_data=preprocess_input(x)

#img_data.shape

model.predict(x)

a=np.argmax(model.predict(x), axis=1)

if a == 0:

    print("buildings")

elif a == 1:

    print("forest")

elif a == 2:

    print("glacier")

elif a == 3:

    print("moutain")

elif a == 4:

    print("sea")

else:

    print("street")


from tensorflow.keras.applications.vgg16 import VGG16



# re-size all the images to this

IMAGE_SIZE = [224, 224]


vgg16 = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg16.layers:

    layer.trainable = False
# useful for getting number of output classes

folders = glob('../input/intel-image-classification/seg_train/seg_train/*')

folders
# our layers - you can add more if you want

x = Flatten()(vgg16.output)



prediction = Dense(len(folders), activation='softmax')(x)



# create a model object

model_vgg16 = Model(inputs=vgg16.input, outputs=prediction)

model_vgg16.summary()
model_vgg16.compile(

  loss='categorical_crossentropy',

  optimizer='adam',

  metrics=['accuracy']

)
train_datagen_vgg16 = ImageDataGenerator(rescale = 1./255)



test_datagen_vgg16 = ImageDataGenerator(rescale = 1./255)



training_set_vgg16 = train_datagen_vgg16.flow_from_directory('../input/intel-image-classification/seg_train/seg_train',

                                                 target_size = (224, 224),

                                                 batch_size = 32,

                                                 class_mode = 'categorical')



test_set_vgg16 = train_datagen_vgg16.flow_from_directory('../input/intel-image-classification/seg_test/seg_test',

                                                 target_size = (244, 244),

                                                 batch_size = 32,

                                                 class_mode = 'categorical')
model_vgg16_fit = model_vgg16.fit_generator(

  training_set_vgg16,

  validation_data=test_set_vgg16,

  epochs=5,

  steps_per_epoch=len(training_set),

  validation_steps=len(test_set)

)
from tensorflow.keras.models import load_model



model_vgg16.save('model_vgg16.h5')
y_pred = model_vgg16.predict(test_set_vgg16)

y_pred = np.argmax(y_pred, axis=1)

y_pred
model=load_model('model_vgg16.h5')

img=image.load_img('../input/intel-image-classification/seg_pred/seg_pred/10875.jpg',target_size=(224,224))

x=image.img_to_array(img)

x=np.expand_dims(x,axis=0)

#img_data=preprocess_input(x)

#img_data.shape

model.predict(x)

a=np.argmax(model.predict(x), axis=1)

if a == 0:

    print("buildings")

elif a == 1:

    print("forest")

elif a == 2:

    print("glacier")

elif a == 3:

    print("moutain")

elif a == 4:

    print("sea")

else:

    print("street")
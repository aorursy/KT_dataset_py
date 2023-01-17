import os 

from keras.models import Sequential

from keras.layers import Conv2D , Flatten , Dense , Dropout , Activation , MaxPooling2D

from keras.preprocessing.image import ImageDataGenerator



print(os.listdir('../input/chest-xray-pneumonia/chest_xray/chest_xray'))



train_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray/train/'

test_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray/test/'

val_dir = '../input/chest-xray-pneumonia/chest_xray/chest_xray/val/'

batch_size = 32

input_shape = (150,150,3)



model = Sequential()

model.add(Conv2D(32,(3,3),input_shape = input_shape))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32,(3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64 , (3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))

model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(64))

model.add(Activation('relu'))

model.add(Dropout(0.25))

model.add(Dense(1))

model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#Training Augmentation

train_datagen = ImageDataGenerator( rescale=1. / 255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)

#Testing Augmentation

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_dir,target_size=(150, 150),batch_size=batch_size,class_mode='binary')

validation_generator  = test_datagen.flow_from_directory(val_dir,target_size=(150, 150),batch_size=batch_size,class_mode='binary')

test_generator = test_datagen.flow_from_directory(test_dir,target_size=(150, 150),batch_size=batch_size,class_mode='binary')

model.fit_generator( train_generator,steps_per_epoch=326,epochs=20, validation_data=validation_generator,validation_steps=16)

#Saving the model

model.save('CNN_model.h5')

#Evaluating the model

scores = model.evaluate_generator(test_generator)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
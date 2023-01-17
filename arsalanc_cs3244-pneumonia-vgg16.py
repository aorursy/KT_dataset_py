from keras import optimizers, metrics, layers, models, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
import keras
import os
train_dir = '../input/cs3244pneumonia/images/train_images/'
val_dir = '../input/cs3244pneumonia/images/val_images/'
train_datagen = ImageDataGenerator(
					zoom_range=0.5,
                    rescale = 1./255
					)
val_datagen = ImageDataGenerator(rescale=1./255) 
train_generator = train_datagen.flow_from_directory(
	train_dir,
	target_size=(150, 150),
	batch_size=32,
	class_mode='binary')
val_generator = val_datagen.flow_from_directory(
	val_dir,
	target_size=(150, 150),
	batch_size=32,
	class_mode='binary')
conv_base = VGG16(weights='../input/vgg16/pretrained_vgg16.h5', include_top=False, input_shape=(150, 150, 3))
conv_base.summary()

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

conv_base.trainable = False

model.compile(loss='binary_crossentropy',
optimizer=optimizers.RMSprop(lr=2e-5),
metrics=['acc'])
callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5
    ),
    keras.callbacks.ModelCheckpoint(
        filepath='./vgg16_imgnet_pneumonia.h5',
        monitor='val_loss',
        save_best_only=True
    )
]
history = model.fit_generator(
	train_generator,
    steps_per_epoch=23366 / 32,
	epochs=50,
	validation_data=val_generator,
    validation_steps=2318 / 32,
    callbacks=callbacks_list
)


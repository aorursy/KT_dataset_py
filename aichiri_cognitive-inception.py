from keras.models import Model

from keras.layers import Dense, GlobalAveragePooling2D

from keras.optimizers import Adadelta

from keras.callbacks import EarlyStopping



from keras.preprocessing.image import ImageDataGenerator



from keras.applications.inception_v3 import InceptionV3, preprocess_input

from keras.preprocessing import image
BATCH_SIZE = 32

OPTIMISER = Adadelta()

NUM_CLASSES = 20



VALIDATION_SPLIT = 0.2



IMG_CHANNELS = 3

IMG_ROWS = 299

IMG_COLUMNS = 299



ICUB_WORLD_DIR = '/kaggle/input/icubworld-cropped/cropped_icub_world'
data_generator = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=VALIDATION_SPLIT)



training_generator = data_generator.flow_from_directory(ICUB_WORLD_DIR,

                                                        target_size=(IMG_ROWS, IMG_COLUMNS),

                                                        color_mode='rgb',

                                                        batch_size=BATCH_SIZE,

                                                        class_mode='categorical',

                                                        subset='training',

                                                        shuffle=True)



validation_generator = data_generator.flow_from_directory(ICUB_WORLD_DIR,

                                                          target_size=(IMG_ROWS, IMG_COLUMNS),

                                                          color_mode='rgb',

                                                          batch_size=BATCH_SIZE,

                                                          class_mode='categorical',

                                                          subset='validation',

                                                          shuffle=True)
base_model = InceptionV3(weights='imagenet', include_top=False)



x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(512, activation='relu')(x)



predictions = Dense(NUM_CLASSES, activation='softmax')(x)



model = Model(inputs=base_model.input, outputs=predictions)
# train only the randomly initiaised top layers for a few epochs

for layer in base_model.layers:

    layer.trainable = False



print('Training the top layers...')

model.compile(optimizer=OPTIMISER, loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit_generator(generator=training_generator, 

                              steps_per_epoch=training_generator.n // BATCH_SIZE,

                              validation_data=validation_generator, 

                              validation_steps=validation_generator.samples // BATCH_SIZE,

                              epochs=2)
for layer in model.layers[:249]:

    layer.trainable = False

for layer in model.layers[249:]:

    layer.trainable = True



print('Retraining the top 2 inception blocks...')

es_callback = EarlyStopping(monitor='val_accuracy', mode='max', patience=2, verbose=1)



model.compile(optimizer=OPTIMISER, loss='categorical_crossentropy', metrics=['accuracy'])



history = model.fit_generator(generator=training_generator, 

                              steps_per_epoch=training_generator.n // BATCH_SIZE,

                              validation_data=validation_generator, 

                              validation_steps=validation_generator.samples // BATCH_SIZE,

                              epochs=4,

                              callbacks=[es_callback])
plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()



plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['train', 'test'], loc='upper left')

plt.show()
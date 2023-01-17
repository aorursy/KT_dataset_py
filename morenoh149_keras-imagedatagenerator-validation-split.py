import keras
from keras import layers
keras.__version__
train_datagen = keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.25,
)
train_generator = train_datagen.flow_from_directory(
    '../input/training_set/training_set',
    target_size=(150, 150),
    class_mode='binary'
)
model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu',
                       input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
             optimizer=keras.optimizers.RMSprop(lr=1e-4),
             metrics=['acc'])
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=2,
    validation_steps=50,
)
history.history
# missing
history.history['val_acc']
import tensorflow.keras as tk
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense, Flatten, GlobalAveragePooling2D, Dropout, Input, Concatenate, BatchNormalization, Conv2D
from tensorflow.keras import Model
import matplotlib.pyplot as plt
train_dir = '../input/cell-images-for-detecting-malaria/cell_images/cell_images/'

train_datagen = ImageDataGenerator(rescale= 1./255, 
                                   horizontal_flip= True, 
                                   vertical_flip= True, 
                                   rotation_range= 45, 
                                   shear_range= 19,
                                   validation_split= 0.25)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size= (96,96), 
                                                    class_mode= 'categorical',
                                                    batch_size= 176,
                                                    shuffle= True,
                                                    subset= 'training')

val_generator = train_datagen.flow_from_directory(train_dir,
                                                  target_size= (96,96),
                                                  class_mode= 'categorical',
                                                  batch_size= 1024,
                                                  shuffle= True,
                                                  subset= 'validation')


inputs = Input((96, 96, 3))
pretrained_model= InceptionV3(include_top= False)
x = pretrained_model(inputs)
output1 = GlobalMaxPooling2D()(x)
output2 = GlobalAveragePooling2D()(x)
output3 = Flatten()(x)

outputs = Concatenate(axis=-1)([output1, output2, output3])
outputs = Dropout(0.5)(outputs)
outputs = BatchNormalization()(outputs)
output = Dense(2, activation= 'softmax')(outputs)

model = Model(inputs, output)


"""
model = tk.models.Sequential()

model.add(Conv2D(16, (3, 3), activation='relu', padding="same", input_shape=(130,130,3)))
model.add(Conv2D(16, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu', padding="same"))
model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(96, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(96, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), dilation_rate=(2, 2), activation='relu', padding="same"))
model.add(Conv2D(128, (3, 3), padding="valid", activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())

model.add(Dense(2 , activation='softmax'))
"""
model.compile(optimizer='RMSProp', loss= 'categorical_crossentropy', metrics= 'accuracy')
history = model.fit_generator(train_generator,
                    epochs = 10,
                    steps_per_epoch = len(train_generator),
                    validation_data = val_generator,
                    validation_steps = len(val_generator),
                    verbose= 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='best')
plt.show()
model.save_weights("weights.h5")
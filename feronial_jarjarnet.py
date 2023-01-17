from PIL import Image
import pandas as pd
from tensorflow.keras.applications import EfficientNetB0, DenseNet121
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input, Conv2D, MaxPool2D, BatchNormalization, Flatten, Dropout, Add
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
BATCH_SIZE = 32
IMG_SHAPE = (256,256,3)
PATH = '/kaggle/input/lego-minifigures-classification/'
NB_CLASS = 29
df = pd.read_csv(PATH + 'index.csv')
df = df.drop(['Unnamed: 0'], axis = 1)

train_df = df[df['train-valid'] == 'train']
validation_df = df[df['train-valid'] == 'valid']

train_df['class_id'] = train_df['class_id'].astype(str)
validation_df['class_id'] = validation_df['class_id'].astype(str)
base_model = DenseNet121(
                            input_shape=IMG_SHAPE, 
                            include_top=False, 
                            weights='imagenet'
    
                            )

base_model.trainable = True

x = base_model.output

global_average = GlobalAveragePooling2D()(x)
output_layer = Dense(NB_CLASS, activation = 'softmax')(global_average)

denseNet_withW= Model(inputs=base_model.input, outputs=output_layer)

#denseNet_withW.summary()
base_model = DenseNet121(
                            input_shape=IMG_SHAPE, 
                            include_top=False, 
                            weights= None
    
                            )

base_model.trainable = True

x = base_model.output

global_average = GlobalAveragePooling2D()(x)
output_layer = Dense(NB_CLASS, activation = 'softmax')(global_average)

denseNet_withoW = Model(inputs=base_model.input, outputs=output_layer)

#denseNet_withoW.summary()
base_model = EfficientNetB0(
                            input_shape=IMG_SHAPE, 
                            include_top=False, 
                            weights= 'imagenet'
    
                            )

base_model.trainable = True

x = base_model.output

global_average = GlobalAveragePooling2D()(x)
output_layer = Dense(NB_CLASS, activation = 'softmax')(global_average)

eNet_withW = Model(inputs=base_model.input, outputs=output_layer)

#eNet_withW.summary()
base_model = EfficientNetB0(
                            input_shape=IMG_SHAPE, 
                            include_top=False, 
                            weights= None
    
                            )

base_model.trainable = True

x = base_model.output

global_average = GlobalAveragePooling2D()(x)
output_layer = Dense(NB_CLASS, activation = 'softmax')(global_average)

eNet_withoW = Model(inputs=base_model.input, outputs=output_layer)

#eNet_withoW.summary()
input_layer = Input(shape = IMG_SHAPE)


x = Conv2D(256, (3,3), activation = 'relu')(input_layer) 
x = MaxPool2D((2,2))(x)


x = Conv2D(256, (3,3), activation = 'relu')(x) 
x = Conv2D(128, (3,3), activation = 'relu')(x)
x = MaxPool2D((3,3))(x)


x = Conv2D(128, (3,3), activation = 'relu')(x)
x = Conv2D(128, (3,3), activation = 'relu')(x)
x = Conv2D(64, (3,3), activation = 'relu')(x) 
x = MaxPool2D((2,2))(x)


x = Conv2D(256, (3,3), activation = 'relu')(x) 
x = Dropout(0.5)(x)
x = Conv2D(256, (3,3), activation = 'relu')(x)
x = Conv2D(32, (3,3), activation = 'relu')(x) 
x = MaxPool2D((2,2))(x)
x = BatchNormalization()(x)

x = Flatten()(x) 
x = Dense(1024, activation = 'relu')(x) 

output_layer = Dense(NB_CLASS, activation = 'softmax')(x)

jarjar_model = Model(inputs = input_layer, outputs = output_layer) 

#jarjar_model.summary()
train_generator = ImageDataGenerator(
                                     rescale=1./255,
                                     rotation_range=45,
                                     width_shift_range=0.25,
                                     height_shift_range=0.25,
                                     shear_range=0.25,
                                     zoom_range=0.25,
                                     horizontal_flip=True,
                                     brightness_range=[0.5, 1.0], 
                                     

                                    )

validation_generator = ImageDataGenerator(rescale=1./255)



train_set = train_generator.flow_from_dataframe(train_df, 
                                                PATH,
                                                x_col = 'path',
                                                y_col = 'class_id',
                                                batch_size = BATCH_SIZE,
                                                class_mode = 'categorical',
                                                target_size = IMG_SHAPE[:2],
                                                suffle=True
                                               )

validation_set = validation_generator.flow_from_dataframe(validation_df, 
                                                          PATH,
                                                          x_col = 'path',
                                                          y_col = 'class_id',
                                                          batch_size = 1,
                                                          class_mode = 'categorical',
                                                          target_size = IMG_SHAPE[:2],
                                                          shuffle = False
                                               )


denseNet_withW.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics='accuracy')
history_denseNet_withW = denseNet_withW.fit_generator(train_set, 
                              steps_per_epoch=train_set.n//train_set.batch_size, 
                              validation_data = validation_set, 
                              epochs = 50)

denseNet_withoW.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics='accuracy')
history_denseNet_withoW = denseNet_withoW.fit_generator(train_set, 
                              steps_per_epoch=train_set.n//train_set.batch_size, 
                              validation_data = validation_set, 
                              epochs = 100)
eNet_withoW.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics='accuracy')
history_eNet_withoW = eNet_withoW.fit_generator(train_set, 
                              steps_per_epoch=train_set.n//train_set.batch_size, 
                              validation_data = validation_set, 
                              epochs = 100)
eNet_withW.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics='accuracy')
history_eNet_withW = eNet_withW.fit_generator(train_set, 
                              steps_per_epoch=train_set.n//train_set.batch_size, 
                              validation_data = validation_set, 
                              epochs = 50)
jarjar_model.compile(optimizer='RMSProp', loss='categorical_crossentropy', metrics='accuracy')
history_jarjar = jarjar_model.fit_generator(train_set, 
                              steps_per_epoch=train_set.n//train_set.batch_size, 
                              validation_data = validation_set, 
                              epochs = 300)
import matplotlib.pyplot as plt

plt.plot(history_jarjar.history['accuracy'])
plt.plot(history_jarjar.history['val_accuracy'])
plt.title('JARJAR-NET ACCURACY PLOT')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history_denseNet_withW.history['accuracy'])
plt.plot(history_denseNet_withW.history['val_accuracy'])
plt.title('DENSE-NET WITH IMAGENET ACCURACY PLOT')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history_denseNet_withoW.history['accuracy'])
plt.plot(history_denseNet_withoW.history['val_accuracy'])
plt.title('DENSE-NET WITHOUT ACCURACY PLOT')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history_eNet_withW.history['accuracy'])
plt.plot(history_eNet_withW.history['val_accuracy'])
plt.title('EFFICIENT-NET WITH IMAGENET ACCURACY')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history_eNet_withoW.history['accuracy'])
plt.plot(history_eNet_withoW.history['val_accuracy'])
plt.title('EFFICIENT-NET WITHOUT IMAGENET ACCURACY PLOT')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
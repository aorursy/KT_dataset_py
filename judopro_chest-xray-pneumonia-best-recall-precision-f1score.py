import pandas as pd

from keras.preprocessing.image import ImageDataGenerator

from keras.applications.vgg16 import preprocess_input

from keras.applications.inception_v3 import preprocess_input as incep_preprocess_input



batch_size = 128

target_size = (299, 299)

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,

                                   shear_range=0.2, zoom_range=0.2,

                                   horizontal_flip=True, fill_mode='nearest')



train_generator = train_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/train/',

                                                    target_size=target_size, color_mode='rgb',

                                                    batch_size=batch_size, class_mode='binary',

                                                    shuffle=True, seed=42)



val_datagen = ImageDataGenerator(preprocessing_function=incep_preprocess_input)



val_generator = val_datagen.flow_from_directory('/kaggle/input/chest-xray-pneumonia/chest_xray/chest_xray/test/',

                                                target_size=target_size, color_mode="rgb",

                                                batch_size=batch_size, shuffle=False, class_mode="binary")





step_size_train = train_generator.n // train_generator.batch_size

step_size_valid = val_generator.n // val_generator.batch_size



df = pd.DataFrame({'data':train_generator.classes})

no_pne = int(df[df.data==train_generator.class_indices['NORMAL']].count())

yes_pne = int(df[df.data==train_generator.class_indices['PNEUMONIA']].count())

print("Normal:{}  Pneumonia:{}".format(no_pne, yes_pne));

from keras.applications import InceptionV3,VGG16

from keras.models import Model

from keras.layers import Flatten, Dense, BatchNormalization, Dropout

from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint



print("Using InceptionV3")

base_model = InceptionV3(weights=None, input_shape=(299,299, 3), include_top=False)

base_model.load_weights('/kaggle/input/weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')



x = base_model.output

x = Flatten()(x)

x = Dense(64, activation='relu')(x)

x = Dropout(0.33)(x)

x = BatchNormalization()(x)

output = Dense(1, activation='sigmoid')(x)



for layer in base_model.layers:

    layer.trainable = False

    

model = Model(inputs=base_model.input, outputs=output)



model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])



model.summary()

chkpt1 = ModelCheckpoint(filepath="best_model_acc.hd5", monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False)

chkpt2 = ModelCheckpoint(filepath="best_model_val_acc.hd5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False)

chkpt3 = ModelCheckpoint(filepath="best_model_val_loss.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)



#history = model.fit_generator(generator=train_generator,

#                    steps_per_epoch=step_size_train,

#                    validation_data=val_generator,

#                    validation_steps=step_size_valid,

#                    callbacks=[chkpt1,chkpt2,chkpt3],

#                    epochs=10, verbose=1)
base_model = VGG16(weights=None, input_shape=(150, 150, 3), include_top=False)

base_model.load_weights('/kaggle/input/weights/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')



x = base_model.output

x = Flatten()(x)

x = Dense(64, activation='relu')(x)

x = Dropout(0.33)(x)

x = BatchNormalization()(x)

output = Dense(1, activation='sigmoid')(x)



for layer in base_model.layers:

    layer.trainable = False

    

model = Model(inputs=base_model.input, outputs=output)



model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])



model.summary()

for layer in base_model.layers:

    if layer.name != 'block5_conv3':

        layer.trainable = False

    else:

        layer.trainable = True

        print("Setting 'block5_conv3' trainable")



for layer in model.layers:

    print("{} {}".format(layer.name, layer.trainable))



model.summary()
model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=0.0001), metrics=['accuracy'])

#Load first stage weights to fine tune off off

#model.load_weights("vgg_best_model.hd5")
import pandas as pd

import math



df = pd.DataFrame({'data':train_generator.classes})

no_pne = int(df[df.data==train_generator.class_indices['NORMAL']].count())

yes_pne = int(df[df.data==train_generator.class_indices['PNEUMONIA']].count())



imb_rat = round(yes_pne / no_pne, 2)



no_weight = imb_rat

yes_weight = 1.0



cweights = {

    train_generator.class_indices['NORMAL']:no_weight,

    train_generator.class_indices['PNEUMONIA']:yes_weight

}



text = "Normal:{:.0f}\nPneumonia:{:.0f}\nImbalance Ratio: {:.2f}\n".format(no_pne, yes_pne, imb_rat)

print(text)

text = "Using class_weights as:\nNormal:{:.2f}\nPneumonia:{:.2f}\n".format(no_weight, yes_weight)

print(text)



#history = model.fit_generator(generator=train_generator,

#                    steps_per_epoch=step_size_train,

#                    validation_data=val_generator,

#                    validation_steps=step_size_valid,

#                    callbacks=[chkpt1,chkpt2,chkpt3],

#                    class_weight=cweights,

#                    epochs=20, verbose=1)

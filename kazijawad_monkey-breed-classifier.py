from keras.applications import MobileNet



# MobileNet was designed to work on 224x224 images

img_rows, img_cols = 224, 224



# Load MobileNet without the top layers

CustomMobileNet = MobileNet(weights="imagenet", include_top=False, input_shape=(img_rows, img_cols, 3))



# Freeze last four layers

for layer in CustomMobileNet.layers:

    layer.trainable = False

    

# Print Layers

for (i, layer) in enumerate(CustomMobileNet.layers):

    print(str(i) + " " + layer.__class__.__name__, layer.trainable)
def add_top_layer(bottom_model, num_classes):

    top_model = bottom_model.output

    top_model = GlobalAveragePooling2D()(top_model)

    top_model = Dense(1024, activation="relu")(top_model)

    top_model = Dense(1024, activation="relu")(top_model)

    top_model = Dense(512, activation="relu")(top_model)

    top_model = Dense(num_classes, activation="softmax")(top_model)

    return top_model
from keras.models import Sequential, Model

from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D

from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D

from keras.layers.normalization import BatchNormalization



num_classes = 10



fc_head = add_top_layer(CustomMobileNet, num_classes)



model = Model(inputs=CustomMobileNet.input, output=fc_head)

print(model.summary())
from keras.preprocessing.image import ImageDataGenerator



train_data_dir = "../input/10-monkey-species/training/training"

validation_data_dir = "../input/10-monkey-species/validation/validation"



# Data Augmentation

train_datagen = ImageDataGenerator(rescale=1./255,

                                   rotation_range=45,

                                   width_shift_range=0.3,

                                   height_shift_range=0.3,

                                   horizontal_flip=True,

                                   fill_mode="nearest")



validation_datagen = ImageDataGenerator(rescale=1./255)



batch_size = 32



train_generator = train_datagen.flow_from_directory(train_data_dir,

                                                    target_size=(img_rows, img_cols),

                                                    batch_size=batch_size,

                                                    class_mode="categorical")



validation_generator = validation_datagen.flow_from_directory(validation_data_dir,

                                                              target_size=(img_rows, img_cols),

                                                              batch_size=batch_size,

                                                              class_mode="categorical")
from keras.optimizers import RMSprop

from keras.callbacks import ModelCheckpoint, EarlyStopping



checkpoint = ModelCheckpoint("./monkey_breed_MobileNet.h5",

                             monitor="val_loss",

                             mode="min",

                             save_best_only=True,

                             verbose=1)



earlystop = EarlyStopping(monitor="val_loss",

                          min_delta=0,

                          patience=3,

                          verbose=1,

                          restore_best_weights=True)



callbacks = [earlystop, checkpoint]



model.compile(loss="categorical_crossentropy",

              optimizer=RMSprop(lr=0.001),

              metrics=["accuracy"])



nb_train_samples = 1098

nb_validation_samples = 272



epochs = 5

batch_size = 16



history = model.fit_generator(train_generator,

                              steps_per_epoch=nb_train_samples//batch_size,

                              epochs=epochs,

                              callbacks=callbacks,

                              validation_data=validation_generator,

                              validation_steps=nb_validation_samples//batch_size)
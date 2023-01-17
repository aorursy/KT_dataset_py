import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline



from os import listdir

from PIL import Image



from keras.models import Sequential, Model, load_model

from keras.layers import Conv2D, Dense, Flatten, Input, BatchNormalization, Activation, Dropout

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from keras.initializers import glorot_uniform

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array

from keras.optimizers import Adam



from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array

from os import listdir

import pandas as pd

import numpy as np
def loadImages(path):

    count = 1

    images = []

    imageList = sorted(listdir(path))

    for i in imageList:

        image = load_img(path + i)

        image = img_to_array(image)

        images.append(image)

        if (count % 1000 == 0):

            print("Processing image", count)

        count += 1

    print("Done.")

    return np.asarray(images, dtype='float')

    
path = "../input/aerial-cactus-identification/train/train/"

X_train = loadImages(path)
X_train.shape
path = "../input/aerial-cactus-identification/test/test/"

X_test = loadImages(path)
X_test.shape
Y_train = pd.read_csv("../input/aerial-cactus-identification/train.csv")
Y_train.head(5)
Y_train = Y_train.sort_values("id", ascending=True).has_cactus
Y_train.value_counts().plot.bar()
epochs = 3

batch_size = 32
plt.figure(figsize=(8, 8))

for i in range(0, 15):

    plt.subplot(5, 3, i+1)

    j = np.random.randint(0, Y_train.shape[0])

    plt.imshow(X_train[j][:, :, 0])

plt.tight_layout()

plt.show()
# Note that for errors occur when trying Inception models due to dimension errors.



from keras.applications.xception import Xception

from keras.applications.vgg16 import VGG16

from keras.applications.vgg19 import VGG19

from keras.applications.mobilenet import MobileNet

from keras.applications.densenet import DenseNet201

from keras.applications.nasnet import NASNetLarge
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, \

                                            factor=0.7, min_lr=0.00001)
datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.2, 

                             height_shift_range=0.2, zoom_range=0.2, 

                             horizontal_flip=True, vertical_flip=True, 

                             validation_split=0.1)
train_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, subset='training')

val_generator = datagen.flow(X_train, Y_train, batch_size=batch_size, subset='validation')
def buildModel(base_model, freeze=0.8):

    

    # Freeze 80% layers, if freeze not specified

    threshold = int(len(base_model.layers) * freeze)

    for i in base_model.layers[:threshold]:

        i.trainable = False

    for i in base_model.layers[threshold:]:

        i.trainable = True



    X = base_model.output

    X = Flatten()(X)

    X = Dense(512, activation='relu', kernel_regularizer='l2')(X)

    X = Dense(1, activation='sigmoid')(X)

    

    model = Model(inputs=base_model.input, outputs=X)

    

    return model
def fitModel(model, lr=0.0005, cp=False):



    # Compile model

    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy', metrics=['accuracy'])



    # Train model

    cb = [learning_rate_reduction, checkpoint] if cp else [learning_rate_reduction]

    history.append(model.fit_generator(generator=train_generator, epochs=epochs,

                                       steps_per_epoch=int(X_train.shape[0] // batch_size * 1.5),

                                       validation_data=val_generator, 

                                       validation_steps=int(X_train.shape[0] // batch_size * 0.4),

                                       callbacks=cb, verbose=2))

    return model
# To store history of training

history = []
# Build model

X_input = Input((32, 32, 3))

base_model = Xception(weights="../input/pretrained-models/xception_weights_tf_dim_ordering_tf_kernels_notop.h5",

                      include_top=False, input_tensor=X_input)

model = buildModel(base_model)



# Train model

fitModel(model)
# Build model

X_input = Input((32, 32, 3))

base_model = VGG16(weights="../input/pretrained-models/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5",

                   include_top=False, input_tensor=X_input)

model = buildModel(base_model)



# Train model

fitModel(model)
# Build model

X_input = Input((32, 32, 3))

base_model = VGG19(weights="../input/pretrained-models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",

                   include_top=False, input_tensor=X_input)

model = buildModel(base_model)



# Train model

fitModel(model)
# # Build model

# X_input = Input((32, 32, 3))

# base_model = ResNet50(weights="../input/pretrained-models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5",

#                       include_top=False, input_tensor=X_input)

# model = buildModel(base_model)



# # Train model

# fitModel(model)
# Build model

X_input = Input((32, 32, 3))

base_model = DenseNet201(weights="../input/pretrained-models/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5",

                         include_top=False, input_tensor=X_input)

model = buildModel(base_model)



# Train model

fitModel(model)
# Build model

X_input = Input((32, 32, 3))

base_model = NASNetLarge(weights="../input/pretrained-models/NASNet-large-no-top.h5",

                         include_top=False, input_tensor=X_input)

model = buildModel(base_model)



# Train model

fitModel(model)
plt.figure(figsize=(10, 7))

for i, j in enumerate(history):

    plt.plot(j.history['val_acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(["Xception val", "VGG16 val", 

            "VGG19 val", "DenseNet201 val",

            "NASNetLarge val"], loc='lower right')

plt.show()
plt.figure(figsize=(10, 7))

for i, j in enumerate(history):

    plt.plot(j.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(["Xception val", "VGG16 val", 

            "VGG19 val", "DenseNet201 val",

            "NASNetLarge val"], loc='lower right')

plt.show()
epochs = 5

history = []
# Turn off learning rate decay

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=100, verbose=1, \

                                            factor=0.7, min_lr=0.00001)
# Build model

X_input = Input((32, 32, 3))

lrs = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01]

for lr in lrs:

    print("=============================")

    print("Fitting model with learning_rate =", lr)

    base_model = VGG19(weights="../input/pretrained-models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",

                       include_top=False, input_tensor=X_input)

    model = buildModel(base_model)



    # Train model

    fitModel(model, lr=lr)
plt.figure(figsize=(10, 7))

val_loss = []

for i, j in enumerate(history):

    val_loss.append(sum(j.history['val_loss']) / 5)

plt.plot(lrs, val_loss)

plt.title('model loss')

plt.ylabel('val loss')

plt.xlabel('learning rate')

plt.show()
epochs = 5

history = []
# Turn on learning rate decay

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, \

                                            factor=0.5, min_lr=1e-5)
# Build model

X_input = Input((32, 32, 3))

frs = [0, 0.25, 0.5, 0.75, 1]

for fr in frs:

    print("=============================")

    print("Fitting model with freeze fraction =", fr, "and learning_rate = 1e-4")

    base_model = VGG19(weights="../input/pretrained-models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",

                       include_top=False, input_tensor=X_input)

    model = buildModel(base_model, freeze=fr)



    # Train model

    fitModel(model, lr=1e-4)
plt.figure(figsize=(10, 7))

val_loss = []

for i, j in enumerate(history):

    val_loss.append(sum(j.history['val_loss']) / 5)

plt.plot(frs, val_loss)

plt.title('model loss')

plt.ylabel('average val loss')

plt.xlabel('freeze fraction')

plt.show()
plt.figure(figsize=(10, 7))

val_loss = []

for i, j in enumerate(history):

    val_loss.append(min(j.history['val_loss']))

plt.plot(frs, val_loss)

plt.title('model loss')

plt.ylabel('min val loss')

plt.xlabel('freeze fraction')

plt.show()
def buildModel(base_model, freeze=0.8):

    

    # Freeze 80% layers, if freeze not specified

    threshold = int(len(base_model.layers) * freeze)

    for i in base_model.layers[:threshold]:

        i.trainable = False

    for i in base_model.layers[threshold:]:

        i.trainable = True

    

    X = base_model.output

    X = Flatten()(X)

    

    X = Dense(512, use_bias=True)(X)

    X = BatchNormalization()(X)

    X = Activation('relu')(X)

    X = Dropout(0.3)(X)

    

    X = Dense(256, use_bias=True)(X)

    X = BatchNormalization()(X)

    X = Activation('relu')(X)

    X = Dropout(0.3)(X)

    

    X = Dense(1, activation='sigmoid')(X)

    

    model = Model(inputs=base_model.input, outputs=X)

    

    return model
checkpoint = ModelCheckpoint("vgg19.h5",monitor="val_acc", 

                             verbose=1, save_best_only=True,

                             mode='max')
epochs = 30
# Build model

X_input = Input((32, 32, 3))

base_model = VGG19(weights="../input/pretrained-models/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5",

                   include_top=False, input_tensor=X_input)

model = buildModel(base_model, freeze=0.25)

# Train model

model = fitModel(model, lr=1e-4, cp=True)
submission = test_images = pd.read_csv("../input/aerial-cactus-identification/sample_submission.csv")

submission = pd.DataFrame(test_images.iloc[:, 0], columns=["id"])
pred = model.predict(X_test).reshape(-1)
submission["has_cactus"] = pd.Series(pred, index=None)
submission.to_csv("submission.csv", index=False)
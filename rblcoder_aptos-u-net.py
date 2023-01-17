import tensorflow as tf
tf.__version__
import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))

import matplotlib.pyplot as plt

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_sample_submission = pd.read_csv('../input/sample_submission.csv')
import imageio
imageio.imread('../input/train_images/' + os.listdir('../input/train_images/')[0]).shape
imageio.imread('../input/test_images/' + os.listdir('../input/test_images/')[0]).shape
_ = plt.imshow(imageio.imread('../input/train_images/' + os.listdir('../input/train_images/')[0]))
_ = plt.imshow(imageio.imread('../input/test_images/' + os.listdir('../input/test_images/')[0]))
#https://stackoverflow.com/questions/44114463/stratified-sampling-in-pandas

df_train_sample = df_train.groupby('diagnosis', group_keys=False).apply(lambda x: x.sample(min(len(x), 2)))
df_train_sample
_ = plt.imshow(imageio.imread('../input/train_images/' + df_train_sample.iloc[4,0] + '.png'))
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(20,20))

i = 0

for ar in axes:

    for ac in ar:

        ac.imshow(imageio.imread('../input/train_images/' + df_train_sample.iloc[i,0] + '.png'))

        ac.set_title(str(df_train_sample.iloc[i,1]))

        i += 1



from keras_preprocessing.image import ImageDataGenerator

#From the book, Keras Deep Learning Cookbook, published by Packt

#https://github.com/PacktPublishing/Keras-Deep-Learning-Cookbook/blob/master/Chapter03/feature_standardization_image.ipynb



#https://github.com/aleju/imgaug/issues/66

# from imgaug import augmenters as iaa

# aug1 = iaa.GaussianBlur(sigma=(0, 2.0))

# aug2 = iaa.AdditiveGaussianNoise(scale=0.01 * 255)



# def additional_augmenation(image):

#     image = aug1.augment_image(image)

#     image = aug2.augment_image(image)

#     return image
#https://medium.com/@arindambaidya168/https-medium-com-arindambaidya168-using-keras-imagedatagenerator-b94a87cdefad

train_datagen = ImageDataGenerator(

                              # rotation_range=8,

                              # horizontal_flip=True,

                              # brightness_range=[0.5, 1.5],

                               validation_split=0.25,

                              # samplewise_center=True,

                              # samplewise_std_normalization=True,

                           #    preprocessing_function=additional_augmenation

                               #featurewise_center=True, 

                               #featurewise_std_normalization=True

                               #rescale=1./255

                               )
# images = []

# for img in os.listdir('../input/train_images/'):

#     image = imageio.imread('../input/train_images/' + img)

#     images.append(image)

    
#train_datagen.fit(images)
test_datagen = ImageDataGenerator(

  #  samplewise_center=True,

   # samplewise_std_normalization=True,

  #  preprocessing_function=additional_augmenation

    #    rescale=1./255

)
IMAGE_HT_WID = 96

BATCH_SIZE = 100
df_train['id_code'] = df_train['id_code'] + '.png'

df_train['diagnosis'] = df_train['diagnosis'].astype(str)
train_generator=train_datagen.flow_from_dataframe(

                    dataframe = df_train,

                    x_col = 'id_code',

                    y_col = 'diagnosis',

                    directory="../input/train_images/",

                    subset="training",

                    batch_size=BATCH_SIZE,

                    seed=42,

                    shuffle=True,

                    class_mode="categorical",

                    color_mode = "grayscale", 

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))



valid_generator=train_datagen.flow_from_dataframe(

                    dataframe = df_train,

                    x_col = 'id_code',

                    y_col = 'diagnosis',

                    directory="../input/train_images/",

                    subset="validation",

                    batch_size=BATCH_SIZE,

                    seed=42,

                    shuffle=True,

                    class_mode="categorical",

                    color_mode = "grayscale",

                    target_size=(IMAGE_HT_WID,IMAGE_HT_WID))
train_generator.class_indices
import tensorflow as tf

from tensorflow import keras

#https://towardsdatascience.com/easy-image-classification-with-tensorflow-2-0-f734fee52d13

print(tf.__version__)

IMG_SHAPE = (IMAGE_HT_WID, IMAGE_HT_WID, 1)

# model = tf.keras.Sequential([

#     tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=IMG_SHAPE,padding='same'),

#     #tf.keras.layers.MaxPooling2D(2, 2),

#     tf.keras.layers.Conv2D(32, (3, 3), activation=tf.nn.relu),

#     tf.keras.layers.MaxPooling2D(2, 2),

#     tf.keras.layers.Dropout(.25),

    

# #     tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu,padding='same'),

# #     #tf.keras.layers.MaxPooling2D(2, 2),

# #     tf.keras.layers.Conv2D(64, (3, 3), activation=tf.nn.relu),

# #     tf.keras.layers.MaxPooling2D(2, 2),

# #     tf.keras.layers.Dropout(.25),

    

#     tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu,padding='same'),

#     #tf.keras.layers.MaxPooling2D(2, 2),

#     tf.keras.layers.Conv2D(128, (3, 3), activation=tf.nn.relu),

#     tf.keras.layers.MaxPooling2D(2, 2),

#     tf.keras.layers.Dropout(.25),



    

#     tf.keras.layers.Flatten(),

#     #tf.keras.layers.GlobalMaxPooling2D(),

#     tf.keras.layers.Dense(512, activation=tf.nn.relu),

#     tf.keras.layers.Dropout(.5), 

#     tf.keras.layers.Dense(5, activation='softmax')

# ])





# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), 

#               loss='categorical_crossentropy', 

#               metrics=['accuracy'])



# #https://keras.io/getting-started/sequential-model-guide/

# # sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])





# model.summary()
# def create_layers(prev_layer, filters, kernel_size, batch_norm=False):

#     if not batch_norm:

#         conv2d_1 = tf.keras.layers.Conv2D(filters=filters,  kernel_size = kernel_size, strides=(1, 1), padding='same', activation='relu')(prev_layer)

#         #batch_1 = tf.keras.layers.BatchNormalization()(conv2d_1)

#         conv2d_2 = tf.keras.layers.Conv2D(filters=filters,  kernel_size = kernel_size, strides=(1, 1), activation='relu')(conv2d_1)

#         #batch_2 = tf.keras.layers.BatchNormalization()(conv2d_2)

#         max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2d_2)

#         drop_1 = tf.keras.layers.Dropout(0.25)(max_pool_1)

#         return drop_1, conv2d_2

#     else:

#         conv2d_1 = tf.keras.layers.Conv2D(filters=filters,  kernel_size = kernel_size, strides=(1, 1), padding='same', activation='relu')(prev_layer)

#         batch_1 = tf.keras.layers.BatchNormalization()(conv2d_1)

#         conv2d_2 = tf.keras.layers.Conv2D(filters=filters,  kernel_size = kernel_size, strides=(1, 1), activation='relu')(batch_1)

#         batch_2 = tf.keras.layers.BatchNormalization()(conv2d_2)

#         max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(batch_2)

#         drop_1 = tf.keras.layers.Dropout(0.25)(max_pool_1)

#         return drop_1, conv2d_2
# def create_only_conv_layers(prev_layer, filters, kernel_size):

#     conv2d_1 = tf.keras.layers.Conv2D(filters=filters,  kernel_size = kernel_size, strides=(1, 1), padding='same', activation='relu')(prev_layer)

#     conv2d_2 = tf.keras.layers.Conv2D(filters=filters,  kernel_size = kernel_size, strides=(1, 1), activation='relu')(conv2d_1)

#     return conv2d_2
# def unet(conv_1, conv_2, conv2d_filters, conv2d_kernel_size, conv2dtransp_filters, 

#          conv2dtransp_kernel_size=2):

#     tp = tf.keras.layers.Conv2DTranspose(conv2dtransp_filters, conv2dtransp_kernel_size, 

#          strides=(2, 2), padding='same')(conv_2)

#     conv2dtransp_1 = tf.keras.layers.concatenate(

#         [tp,

#          conv_1])

#     conv2d_1 = Conv2D(filters=conv2d_filters,  kernel_size = conv2d_kernel_size, activation='relu',

#                       padding='same')(conv2dtransp_1)

#     conv2d_2 = Conv2D(filters=conv2d_filters,  kernel_size = conv2d_kernel_size, activation='relu', 

#                       padding='same')(conv2d_1)

#     return conv2d_2

    
#input_layer = tf.keras.layers.Input(IMG_SHAPE)


# conv2d_1_1 = tf.keras.layers.Conv2D(filters=8,  kernel_size = 2, strides=(1, 1), padding='same', activation='relu')(input_layer)

# conv2d_1_2 = tf.keras.layers.Conv2D(filters=8,  kernel_size = 2, strides=(1, 1), activation='relu')(conv2d_1_1)

# max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2d_1_2)

# drop_1 = tf.keras.layers.Dropout(0.25)(max_pool_1)
# input_layer = tf.keras.layers.Input(IMG_SHAPE)

# conv2d_1_1 = tf.keras.layers.Conv2D(filters=8,  kernel_size = 2, strides=(1, 1), padding='same', activation='relu')(input_layer)

# conv2d_1_2 = tf.keras.layers.Conv2D(filters=8,  kernel_size = 2, strides=(1, 1), activation='relu')(conv2d_1_1)

# max_pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2d_1_2)

# drop_1 = tf.keras.layers.Dropout(0.25)(max_pool_1)
# conv2d_2_1 = tf.keras.layers.Conv2D(filters=16,  kernel_size = 3, strides=(1, 1), padding='same', activation='relu')(drop_1)

# conv2d_2_2 = tf.keras.layers.Conv2D(filters=16,  kernel_size = 3, strides=(1, 1), activation='relu')(conv2d_2_1)

# max_pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2d_2_2)

# drop_2 = tf.keras.layers.Dropout(0.25)(max_pool_2)
# conv2d_3_1 = tf.keras.layers.Conv2D(filters=32,  kernel_size = 3, strides=(1, 1), padding='same', activation='relu')(drop_2)

# conv2d_3_2 = tf.keras.layers.Conv2D(filters=32,  kernel_size = 3, strides=(1, 1), activation='relu')(conv2d_3_1)

# max_pool_3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2d_3_2)

# drop_3 = tf.keras.layers.Dropout(0.25)(max_pool_3)
# conv2d_4_1 = tf.keras.layers.Conv2D(filters=128,  kernel_size = 3, strides=(1, 1), padding='same', activation='relu')(drop_3)

# conv2d_4_2 = tf.keras.layers.Conv2D(filters=128,  kernel_size = 3, strides=(1, 1), activation='relu')(conv2d_4_1)

# max_pool_4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2d_4_2)

# drop_4 = tf.keras.layers.Dropout(0.25)(max_pool_3)
#concat = tf.keras.layers.concatenate([drop_1, drop_2])
# input_layer = tf.keras.layers.Input(IMG_SHAPE)

# conv_1, block_1 = create_layers(input_layer, filters=16, kernel_size=3)

# conv_2, block_2 = create_layers(block_1, filters=32, kernel_size=3)

# conv_3, block_3 = create_layers(block_2, filters=64, kernel_size=3)

# conv_4, block_4 = create_layers(block_3, filters=128, kernel_size=3)

#prev_layer = create_layers(prev_layer, filters=256, kernel_size=3)
# conv_5 = create_only_conv_layers(block_4, filters=256, kernel_size=3)
# conv_6 = unet(conv_4, conv_5, conv2d_filters=128, conv2d_kernel_size=3, conv2dtransp_filters=128)

# conv_7 = unet(conv_3, conv_6, conv2d_filters=64, conv2d_kernel_size=3, conv2dtransp_filters=64)

# conv_8 = unet(conv_2, conv_7, conv2d_filters=32, conv2d_kernel_size=3, conv2dtransp_filters=32)

# conv_9 = unet(conv_1, conv_8, conv2d_filters=16, conv2d_kernel_size=3, conv2dtransp_filters=16)
# flat = tf.keras.layers.Flatten()(prev_layer)

# dense = tf.keras.layers.Dense(512, activation='relu')(flat)

# drop = tf.keras.layers.Dropout(0.25)(dense)

# output_dense = tf.keras.layers.Dense(5, activation='softmax')(drop)
# model_1 = tf.keras.models.Model(input_layer,output_dense)

# model_1.compile(

#               #optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), 

#               optimizer=tf.keras.optimizers.RMSprop(lr=0.01),   

#               loss='categorical_crossentropy', 

#               metrics=['accuracy'])
#https://www.kaggle.com/cjansen/u-net-in-keras

# from keras.models import Sequential, Model

# from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Input, merge, UpSampling2D, Cropping2D, ZeroPadding2D, Reshape, core, Convolution2D

# from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler

# from keras import optimizers

# from keras import backend as K

# from keras.optimizers import SGD, RMSprop

# from keras.layers.merge import concatenate



def get_crop_shape(target, refer):

        # width, the 3rd dimension

        cw = (target.get_shape()[2] - refer.get_shape()[2]).value

        assert (cw >= 0)

        if cw % 2 != 0:

            cw1, cw2 = int(cw/2), int(cw/2) + 1

        else:

            cw1, cw2 = int(cw/2), int(cw/2)

        # height, the 2nd dimension

        ch = (target.get_shape()[1] - refer.get_shape()[1]).value

        assert (ch >= 0)

        if ch % 2 != 0:

            ch1, ch2 = int(ch/2), int(ch/2) + 1

        else:

            ch1, ch2 = int(ch/2), int(ch/2)



        return (ch1, ch2), (cw1, cw2)

    

def get_unet(n_ch,patch_height,patch_width):

    concat_axis = 3



    inputs = tf.keras.layers.Input((patch_height, patch_width, n_ch))

    

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", name="conv1_1", activation="relu", data_format="channels_last")(inputs)

    conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv1)

    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool1)

    conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv2)

    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv2)



    conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool2)

    conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv3)

    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv3)



    conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool3)

    conv4 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv4)

    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), data_format="channels_last")(conv4)



    conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(pool4)

    conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv5)



    up_conv5 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(conv5)

    ch, cw = get_crop_shape(conv4, up_conv5)

    crop_conv4 = tf.keras.layers.Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv4)

    up6   = tf.keras.layers.concatenate([up_conv5, crop_conv4], axis=concat_axis)

    conv6 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(up6)

    conv6 = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv6)



    up_conv6 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(conv6)

    ch, cw = get_crop_shape(conv3, up_conv6)

    crop_conv3 = tf.keras.layers.Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv3)

    up7   = tf.keras.layers.concatenate([up_conv6, crop_conv3], axis=concat_axis)

    conv7 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(up7)

    conv7 = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv7)



    up_conv7 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(conv7)

    ch, cw = get_crop_shape(conv2, up_conv7)

    crop_conv2 = tf.keras.layers.Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv2)

    up8   = tf.keras.layers.concatenate([up_conv7, crop_conv2], axis=concat_axis)

    conv8 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(up8)

    conv8 = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv8)



    up_conv8 = tf.keras.layers.UpSampling2D(size=(2, 2), data_format="channels_last")(conv8)

    ch, cw = get_crop_shape(conv1, up_conv8)

    crop_conv1 = tf.keras.layers.Cropping2D(cropping=(ch,cw), data_format="channels_last")(conv1)

    up9   = tf.keras.layers.concatenate([up_conv8, crop_conv1], axis=concat_axis)

    conv9 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(up9)

    conv9 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu", data_format="channels_last")(conv9)



    #ch, cw = get_crop_shape(inputs, conv9)

    #conv9  = ZeroPadding2D(padding=(ch[0],cw[0]), data_format="channels_last")(conv9)

    #conv10 = Conv2D(1, (1, 1), data_format="channels_last", activation="sigmoid")(conv9)

    

    flatten =  tf.keras.layers.Flatten()(conv9)

    Dense1 = tf.keras.layers.Dense(512, activation='relu')(flatten)

    BN = tf.keras.layers.BatchNormalization() (Dense1)

    Dense2 = tf.keras.layers.Dense(5, activation='softmax')(BN)

    

    model = tf.keras.models.Model(inputs=inputs, outputs=Dense2)

    

    return model





model_u = get_unet(1, 96, 96)

model_u.summary()
#len(model.trainable_variables)
#https://www.kaggle.com/c/imaterialist-fashion-2019-FGVC6/discussion/95247#latest-567841



model_u.compile(

             #  optimizer=tf.keras.optimizers.SGD(lr=0.003, momentum=0.9, decay=0.0001), 

              # optimizer=tf.keras.optimizers.RMSprop(lr=0.0001), 

     optimizer=tf.keras.optimizers.RMSprop(lr=0.1), 

              #optimizer=tf.keras.optimizers.Adam(lr=0.0001),   

               loss='categorical_crossentropy', 

               metrics=['accuracy'] 

               )
#len(model_u.trainable_variables)
tf.keras.utils.plot_model(model_u, to_file='model_u.png')
EPOCHS=20

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size + 1

STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size + 1

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.7, 

                                                                    patience=3, verbose=2, mode='auto',

                                                                    min_lr=1e-6)
#https://keras.io/callbacks/#modelcheckpoint

callbacks_ls = [tf.keras.callbacks.ModelCheckpoint('weights.hdf5', 

                                                   monitor='val_loss', verbose=1,

                                                   save_best_only=True)

               #,reduce_lr

               ]
history = model_u.fit_generator(generator=train_generator,

                    steps_per_epoch=STEP_SIZE_TRAIN,

                    validation_data=valid_generator,

                    validation_steps=STEP_SIZE_VALID,

                    epochs=EPOCHS,

                    callbacks=callbacks_ls,

                    workers=4,

                    verbose=2

)
import matplotlib.pyplot as plt

acc = history.history['acc']

val_acc = history.history['val_acc']



epochs = range(len(acc))



plt.plot(epochs, acc, 'r', label='Training accuracy')

plt.plot(epochs, val_acc, 'b', label='Validation accuracy')

plt.title('Training and validation accuracy')

plt.legend(loc=0)

plt.figure()





_=plt.show()
model_u.load_weights('weights.hdf5')
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size + 1

valid_generator.reset()

pred_v=model_u.predict_generator(valid_generator,

                steps=STEP_SIZE_VALID,

                verbose=1)
predicted_v_class_indices=np.argmax(pred_v,axis=1)
from sklearn.metrics import classification_report

print(classification_report(valid_generator.classes, predicted_v_class_indices))
from sklearn.metrics import cohen_kappa_score

cohen_kappa_score(valid_generator.classes, predicted_v_class_indices)
df_test['id_code'] = df_test['id_code'] + '.png'
test_generator=test_datagen.flow_from_dataframe(

                dataframe = df_test,

                x_col = 'id_code',

                directory="../input/test_images/",

                batch_size=BATCH_SIZE,

                seed=42,

                shuffle=False,

                class_mode=None,

                color_mode = "grayscale",

                target_size=(IMAGE_HT_WID,IMAGE_HT_WID))
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size + 1

test_generator.reset()

pred=model_u.predict_generator(test_generator,

                steps=STEP_SIZE_TEST,

                verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
#https://www.kaggle.com/hsinwenchang/keras-mobilenet-data-augmentation-visualize

labels = (train_generator.class_indices)

labels = dict((v,k) for k,v in labels.items())

predictions = [labels[k] for k in predicted_class_indices]

filenames=test_generator.filenames

results=pd.DataFrame({"id_code":filenames,

                      "diagnosis":predictions})

#subm = pd.merge(df_test, results, on='file_name')[['id','Category']]

results.loc[:,'id_code'] = results['id_code'].str.replace('test_images/','')

results.loc[:,'id_code'] = results['id_code'].str.replace('.png','')

results['diagnosis'].value_counts()
results.to_csv("submission.csv",index=False)
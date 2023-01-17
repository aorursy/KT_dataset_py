import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras

import h5py

import os

import seaborn

from keras.layers import *

from keras.models import Model

from keras import optimizers

from keras import regularizers

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

test_df = pd.read_csv('../input/petfinder-adoption-prediction/test/test.csv')

train_dir = '../input/petfinder-adoption-prediction/train_images'

test_dir = '../input/petfinder-adoption-prediction/test_images'

rf_result = pd.read_csv('../input/random-forest-result/submission.csv')
name_target_dict = train_df.set_index('PetID')['AdoptionSpeed'].to_dict()

rf_result_dict = rf_result.set_index('PetID')['AdoptionSpeed'].to_dict()

train_image_names = os.listdir('../input/petfinder-adoption-prediction/train_images')

test_image_names = os.listdir('../input/petfinder-adoption-prediction/test_images')
generator_dict = {'filename': [], 'class': []}



for name in train_image_names:

    short_name = name.split('-')[0]

    label = name_target_dict[short_name]

    

    generator_dict['filename'].append(name)

    generator_dict['class'].append(str(label))



generator_df = pd.DataFrame(generator_dict)

generator_df.head(8)
datagen = ImageDataGenerator(

    featurewise_center=False,  # set input mean to 0 over the dataset

    samplewise_center=False,  # set each sample mean to 0

    featurewise_std_normalization=False,  # divide inputs by std of the dataset

    samplewise_std_normalization=False,  # divide each input by its std

    zca_whitening=False,  # apply ZCA whitening

    zca_epsilon=1e-06,  # epsilon for ZCA whitening

    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)

    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)

    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)

    shear_range=0.1,  # set range for random shear

    zoom_range=0.1,  # set range for random zoom

    channel_shift_range=0.,  # set range for random channel shifts

    # set mode for filling points outside the input boundaries

    fill_mode='nearest',

    cval=0.,  # value used for fill_mode = "constant"

    horizontal_flip=True,  # randomly flip images

    vertical_flip=False,  # randomly flip images

    # set rescaling factor (applied before any other transformation)

    rescale=1/255.,

    # set function that will be applied on each input

    preprocessing_function=None,

    # image data format, either "channels_first" or "channels_last"

    data_format=None,

    # fraction of images reserved for validation (strictly between 0 and 1)

    validation_split=0.1)



def create_generator(subset):

    return datagen.flow_from_dataframe(

        generator_df, 

        train_dir, 

        x_col='filename',

        y_col='class', 

        has_ext=True,  # If image extension is given in x_col

        target_size=(100, 100), 

        color_mode='rgb',

        class_mode='categorical', 

        batch_size=128, 

        shuffle=True, 

        seed=2018,

        subset=subset

    )
train_generator = create_generator('training')

validation_generator = create_generator('validation')
num_train = 52480

num_val = 5831
vgg16_model = keras.applications.VGG16(weights= None, include_top=False)



vgg16_dir = '../input/VGG-16'

model_path_pattern = '../input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

vgg16_model.load_weights(model_path_pattern)



vgg16_model.summary()
# set the original layers (up to the last conv block)

# to non-trainable (weights will not be updated)

for layer in vgg16_model.layers:

    layer.trainable = False
# print('Adding Average Pooling Layer and Softmax Output Layer ...')

x = vgg16_model.get_layer(index = -1).output  # Shape: (8, 8, 2048)



# output = GlobalAveragePooling2D()(output)

# # # let's add a fully-connected layer

# output = Dense(1024, activation='relu')(output)

# # # and a logistic layer -- let's say we have 200 classes

# output = Dense(5, activation='softmax')(output)



x = GlobalAveragePooling2D()(x)

#x = Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)

#x = Dropout(0.5)(x)

x = Dense(5, activation='softmax')(x)





vgg16 = Model(vgg16_model.input, x)
vgg16.summary()
vgg16.compile(loss='categorical_crossentropy',

              optimizer=optimizers.Adam(lr=1e-4),

              metrics=['accuracy'])
fine_weights_path = 'tune_weights.h5'
reduce_lr = ReduceLROnPlateau(monitor='val_loss', 

                              mode='min', 

                              factor=0.5, 

                              patience=5,

                              min_lr=0.0001,

                              verbose=1)
callbacks_list = [

    ModelCheckpoint(fine_weights_path, monitor='val_acc', verbose=1, save_best_only=True),

    EarlyStopping(monitor='val_acc', patience=5, verbose=1),reduce_lr]
history = vgg16.fit_generator(

        train_generator,

        steps_per_epoch=num_train//128, 

        epochs=30,

        validation_data=validation_generator,

        validation_steps=num_val//128,

        callbacks=callbacks_list)
print ('Training Accuracy = ' + str(history.history['acc']))

print ('Validation Accuracy = ' + str(history.history['val_acc']))
import matplotlib.pyplot as plt



# list all data in history

print(history.history.keys())



plt.plot(history.history['val_acc'])

plt.plot(history.history['acc'])

plt.title('model accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()

# summarize history for loss

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('model loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train', 'validation'], loc='upper left')

plt.show()
submission_dict = {'PetID': [], 'filename': []}



for name in test_image_names:

    pet_id = name.split('-')[0]

    submission_dict['PetID'].append(pet_id)

    submission_dict['filename'].append(name)

    

submission_df = pd.DataFrame(submission_dict)

#submission_df.head()



test_generator = ImageDataGenerator(rescale=1/255.).flow_from_dataframe(

    submission_df,

    test_dir,

    has_ext=True,

    target_size=(100, 100),

    color_mode='rgb',

    batch_size=256,

    shuffle=False,

    class_mode=None

)



test_predictions = vgg16.predict_generator(

    test_generator,

    #workers=2, 

    #use_multiprocessing=True, 

    #verbose=1

    steps = 1000

)
submission_df = submission_df.join(pd.DataFrame(test_predictions))

submission_df.drop(columns=['filename'], inplace=True)

#print(submission_df.shape)

#submission_df.head()

submission_df = submission_df.groupby('PetID', as_index=False).mean()

#print(submission_df.shape)

#submission_df.head()

submission_df['AdoptionSpeed'] = submission_df.iloc[:,1:6].values.argmax(axis=1)

submission_df.drop(columns=[0,1,2,3,4], inplace=True)

#print(submission_df.shape)

#submission_df.head()
#If pets do not have any images, then give it the random forest predicted adoption speed

for id in test_df['PetID']:

    if (id not in submission_df['PetID'].values):

        submission_df = submission_df.append(pd.DataFrame({'PetID': [id], 'AdoptionSpeed': [rf_result_dict[str(id)]]}))
submission_df.to_csv('submission.csv',index=False)
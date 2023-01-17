# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



#import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.model_selection import train_test_split



train_df_whole = pd.read_csv("/kaggle/input/severstal-steel-defect-detection/train.csv")

train_df_whole["image_id"] = train_df_whole["ImageId_ClassId"].str[:-2]

train_df_whole["class_id"] = train_df_whole["ImageId_ClassId"].str[-1].astype("float")

train_df_whole["defected"] = ~train_df_whole["EncodedPixels"].isna()

train_df_whole["class_id"] = train_df_whole["class_id"]*train_df_whole["defected"].astype("float")

train_df_whole.drop("ImageId_ClassId", axis=1, inplace=True)

train_df_whole.drop_duplicates(inplace=True)

freq_df = pd.DataFrame(train_df_whole["image_id"].value_counts().reset_index())

freq_df.columns = ["image_id", "frequency"]

train_df_whole = train_df_whole.merge(freq_df, how="left", on="image_id")

the_filter = ~((train_df_whole["frequency"] >= 2) & (train_df_whole["EncodedPixels"].isna()))

train_df_whole = train_df_whole[the_filter]

train_df_whole = train_df_whole.drop(["EncodedPixels", "class_id", "frequency"], axis=1).drop_duplicates()

train_df_whole.reset_index(drop=True, inplace=True)



train_df, validate_df = train_test_split(train_df_whole, test_size=0.1, stratify=train_df_whole[["defected"]].copy())

train_df.reset_index(drop=True, inplace=True)

validate_df.reset_index(drop=True, inplace=True)



print(train_df["defected"].value_counts())
import os

from keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile



data_parent_dir = "/kaggle/classification"

if not os.path.exists(data_parent_dir):

    os.mkdir(data_parent_dir)



data_dir = "/kaggle/classification/data"

if not os.path.exists(data_dir):

    os.mkdir(data_dir)



defected_images_dir = "/kaggle/classification/data/defected"

if not os.path.exists(defected_images_dir):

    os.mkdir(defected_images_dir)

    

for the_file in os.listdir(defected_images_dir):

    file_path = os.path.join(defected_images_dir, the_file)

    try:

        if os.path.isfile(file_path):

            os.unlink(file_path)

        #elif os.path.isdir(file_path): shutil.rmtree(file_path)

    except Exception as e:

        print(e)

    

not_defected_images_dir = "/kaggle/classification/data/not_defected"

if not os.path.exists(not_defected_images_dir):

    os.mkdir(not_defected_images_dir)

    

for the_file in os.listdir(not_defected_images_dir):

    file_path = os.path.join(not_defected_images_dir, the_file)

    try:

        if os.path.isfile(file_path):

            os.unlink(file_path)

        #elif os.path.isdir(file_path): shutil.rmtree(file_path)

    except Exception as e:

        print(e)

    

validation_data_dir = "/kaggle/classification/validation_data"

if not os.path.exists(validation_data_dir):

    os.mkdir(validation_data_dir)

    

defected_validation_images_dir = "/kaggle/classification/validation_data/defected"

if not os.path.exists(defected_validation_images_dir):

    os.mkdir(defected_validation_images_dir)

    

for the_file in os.listdir(defected_validation_images_dir):

    file_path = os.path.join(defected_validation_images_dir, the_file)

    try:

        if os.path.isfile(file_path):

            os.unlink(file_path)

        #elif os.path.isdir(file_path): shutil.rmtree(file_path)

    except Exception as e:

        print(e)

    

not_defected_validation_images_dir = "/kaggle/classification/validation_data/not_defected"

if not os.path.exists(not_defected_validation_images_dir):

    os.mkdir(not_defected_validation_images_dir)

    

for the_file in os.listdir(not_defected_validation_images_dir):

    file_path = os.path.join(not_defected_validation_images_dir, the_file)

    try:

        if os.path.isfile(file_path):

            os.unlink(file_path)

        #elif os.path.isdir(file_path): shutil.rmtree(file_path)

    except Exception as e:

        print(e)



orig_train_dir = "/kaggle/input/severstal-steel-defect-detection/train_images/"

for file_name, the_label in zip(train_df["image_id"], train_df["defected"]):

    src = os.path.join(orig_train_dir, file_name)

    if the_label:

        dst = os.path.join(defected_images_dir, file_name)

    else:

        dst = os.path.join(not_defected_images_dir, file_name)

    copyfile(src, dst)



for file_name, the_label in zip(validate_df["image_id"], validate_df["defected"]):

    src = os.path.join(orig_train_dir, file_name)

    if the_label:

        dst = os.path.join(defected_validation_images_dir, file_name)

    else:

        dst = os.path.join(not_defected_validation_images_dir, file_name)

    copyfile(src, dst)

        

train_batch_size = 8

validate_batch_size = 8



data_gen_args_image = dict(

    rotation_range=40.0,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.2,

    zoom_range=0.1,

    horizontal_flip=True,

    vertical_flip=True,

    rescale=1.0/255.0

)



# we create two instances with the same arguments

image_datagen = ImageDataGenerator(**data_gen_args_image)



# Provide the same seed and keyword arguments to the fit and flow methods

image_generator = image_datagen.flow_from_directory(

    '/kaggle/classification/data',

    target_size=(256, 1600),

    batch_size=train_batch_size,

    color_mode="rgb",

    class_mode="binary",

    classes=["not_defected", "defected"]

)

train_generator = image_generator



validation_image_datagen = ImageDataGenerator(rescale=1.0/255.0)

validation_image_generator = validation_image_datagen.flow_from_directory(

    '/kaggle/classification/validation_data',

    target_size=(256, 1600),

    batch_size=validate_batch_size,

    color_mode="rgb",

    class_mode="binary",

    classes=["not_defected", "defected"]

)



validation_generator = validation_image_generator
from keras.losses import binary_crossentropy

from keras.applications import VGG16

from keras.layers import Flatten, Dense, UpSampling2D, Conv2D, Activation, MaxPooling2D  # , Conv2DTranspose

from keras.models import Model

from keras import optimizers

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint



conv_base = VGG16(weights="/kaggle/input/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5", include_top=False, input_shape=(256, 1600, 3))



x = conv_base.output

# x = UpSampling2D(32, interpolation='bilinear')(x)

x = Conv2D(1, (2, 2))(x)

x = Flatten()(x)

outputs = Dense(units=1, activation="sigmoid")(x)

model = Model(inputs=conv_base.input, outputs=outputs)

for layer in conv_base.layers:

    layer.trainable = False

model.summary()



callbacks_list = [

    # EarlyStopping(monitor="acc", patience=2),

    ModelCheckpoint(filepath="best_model_binary.h5", monitor="val_loss", save_best_only=True, verbose=1),

    ReduceLROnPlateau(monitor="val_loss", factor=0.8, patience=1, cooldown=2, verbose=1),

    #TensorBoard(log_dir="./log_dir", histogram_freq=1)

]



model.compile(loss=binary_crossentropy, optimizer=optimizers.Adam(), metrics=["acc"])



train_set_size = train_df.shape[0]

if train_batch_size == train_set_size:

    steps_per_epoch = 1

else:

    steps_per_epoch = (train_set_size // train_batch_size) + 1



validate_set_size = validate_df.shape[0]

if validate_batch_size == validate_set_size:

    validate_steps_per_epoch = 1

else:

    validate_steps_per_epoch = (validate_set_size // validate_batch_size) + 1



history = model.fit_generator(

    generator=train_generator,

    steps_per_epoch=steps_per_epoch,

    epochs=22,

    validation_data=validation_generator,

    validation_steps = validate_steps_per_epoch,

    callbacks=callbacks_list

)
import matplotlib.pyplot as plt



acc = history.history['acc']

val_acc = history.history['val_acc']

loss = history.history['loss']

val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.figure(1)

_ = plt.plot(epochs, acc, 'bo', label='Training acc')

_ = plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')

_ = plt.legend()

plt.figure(2)

_ = plt.plot(epochs, loss, 'bo', label='Training loss')

_ = plt.plot(epochs, val_loss, 'b', label='Validation loss')

plt.title('Training and validation loss')

_ = plt.legend()
from keras.preprocessing.image import load_img, img_to_array

from numpy import squeeze



the_image = load_img("/kaggle/input/severstal-steel-defect-detection/test_images/1804f41eb.jpg")

image_array = img_to_array(the_image)

image_array = np.expand_dims(image_array, 0)

predicted_type = model.predict(image_array, batch_size=1)

print("predicted type:")

print(predicted_type)
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os



from PIL import Image

from matplotlib import pyplot as plt
data_dir = "../input"

model_dir = "ppe-training"
data_parts = os.listdir(data_dir)

data_parts = [p for p in data_parts if p != model_dir]

data_parts
!pip install iterative-stratification

!pip install googledrivedownloader
from google_drive_downloader import GoogleDriveDownloader as gdd

gdd.download_file_from_google_drive(file_id='1dS0qMIWiZaxaFE250eT8KdHUtsPt-D8i', dest_path='../working/ppe_data.zip', unzip=True)

!rm -rf ppe_data.zip
!ls
gdd.download_file_from_google_drive(file_id='1BrYKVyGaKFk7wh5IJ3vqrairNrfeV2f9', dest_path='../working/label.xlsx')
data_df_old = pd.read_excel("label.xlsx")

data_df_old.head()
data_df_old["Name"] = data_df_old["Name"].apply(lambda x: os.path.join("../working/data", x))

data_df_old.head()
class_list = data_df_old.columns.tolist()[1:]

class_dict = dict((i, class_list[i]) for i in  range(len(class_list)))

class_dict
data = []

num_wrong_format_images = 0

for dirname, _, filenames in os.walk(data_dir):

    if model_dir in dirname:

        continue

        

    for filename in filenames:

        if ".jpg" not in filename and ".jpeg" not in filename and ".png" not in filename:

            num_wrong_format_images += 1

            continue

            

        dirname_splits = dirname.split("/")

        label = dirname_splits[3]

        

        sample = {

            "Name": os.path.join(dirname, filename),

            "ClassName": label

        }

        

        data.append(sample)





data_df_new = pd.DataFrame(data)

data_df_new.head()
print("Num of wrong format images: {}".format(num_wrong_format_images))
for class_name in class_list: 

    data_df_new[class_name] = data_df_new.apply(lambda row: 1 if row.ClassName == class_name else 0, axis=1)



data_df_new.head()
data_df_new.drop(columns=['ClassName'], inplace=True)

data_df_new.head()
print("Old data: {}".format(len(data_df_old)))

print("New data: {}".format(len(data_df_new)))
data_df = data_df_old.append(data_df_new)

print("Total data: {}".format(len(data_df)))

data_df.head()
data_df =  data_df.sort_values(by=['Name'])

data_df.head()
class Config:

    

    img_width = 512

    img_height = 512

    

    num_classes = 3



    batch_size = 16

    epochs = 25

    warmup_epochs = 2

    

    seed = 2019

    

    path_to_model = 'best_model.h5'

    path_to_latest_model = 'latest_model.h5'

    log_file = 'result.csv'

    verbose = 1

    

config = Config()
import imgaug as ia

from imgaug import augmenters as iaa

sometimes = lambda aug: iaa.Sometimes(0.5, aug)

seq = iaa.Sequential(

    [

        # apply the following augmenters to most images

        iaa.Fliplr(0.5), # horizontally flip 50% of all images

        iaa.Flipud(0.2), # vertically flip 20% of all images

        # crop images by -5% to 10% of their height/width

        sometimes(iaa.CropAndPad(

            percent=(-0.05, 0.1),

            pad_mode=ia.ALL,

            pad_cval=(0, 255)

        )),

        sometimes(iaa.Affine(

            scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis

            translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)

            rotate=(-45, 45), # rotate by -45 to +45 degrees

            shear=(-16, 16), # shear by -16 to +16 degrees

            order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)

            cval=(0, 255), # if mode is constant, use a cval between 0 and 255

            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)

        )),

        # execute 0 to 5 of the following (less important) augmenters per image

        # don't execute all of them, as that would often be way too strong

        iaa.SomeOf((0, 5),

            [

                sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation

                iaa.OneOf([

                    iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0

                    iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7

                    iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7

                ]),

                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images

                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images

                # search either for all edges or for directed edges,

                # blend the result with the original image using a blobby mask

                iaa.SimplexNoiseAlpha(iaa.OneOf([

                    iaa.EdgeDetect(alpha=(0.5, 1.0)),

                    iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),

                ])),

                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images

                iaa.OneOf([

                    iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels

                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),

                ]),

                iaa.Invert(0.05, per_channel=True), # invert color channels

                iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

                iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation

                # either change the brightness of the whole image (sometimes

                # per channel) or change the brightness of subareas

                iaa.OneOf([

                    iaa.Multiply((0.5, 1.5), per_channel=0.5),

                    iaa.FrequencyNoiseAlpha(

                        exponent=(-4, 0),

                        first=iaa.Multiply((0.5, 1.5), per_channel=True),

                        second=iaa.ContrastNormalization((0.5, 2.0))

                    )

                ]),

                iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast

                iaa.Grayscale(alpha=(0.0, 1.0)),

                sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)

                sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around

                sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))

            ],

            random_order=True

        )

    ],

    random_order=True

)
# data_df = data_df[data_df["Name"].map(lambda x: cv2.imread(x) != None)]
len(data_df)
from sklearn.utils import shuffle



import cv2



class ImageGenerator:

    

    def create(image_df, augument=True):

        while True:

            image_df = shuffle(image_df, random_state=config.seed)

            for start in range(0, len(image_df), config.batch_size):

                end = min(start + config.batch_size, len(image_df))

                batch_images = []

                X_train_batch = image_df.iloc[start:end]

                batch_labels = np.zeros((len(X_train_batch), config.num_classes))

                

                deleted_idxs = []

                for i in range(len(X_train_batch)):

                    image_path = os.path.join(data_dir, X_train_batch.iloc[i].Name)

                    image = ImageGenerator.load_image(image_path)

                    if image is None:

                        deleted_idxs.append(i)

                        continue

                    if augument:

                        image = ImageGenerator.augment(image)



                    batch_images.append(image/255.)

                    batch_labels[i] = X_train_batch.iloc[i].values[1:]

                    

                batch_labels = np.delete(batch_labels, deleted_idxs, axis=0)

                yield np.array(batch_images, np.float32), batch_labels



    def load_image(path):

#         image = np.array(Image.open(path)) 

        image = cv2.imread(path)

        if image is None:

            return image

        image = image.astype(np.uint8)

        image = cv2.resize(image,(config.img_width,config.img_height))

        

        if len(image.shape) == 2:

            image = np.expand_dims(image, axis=-1).repeat(3, axis=-1)



        return image



    def augment(image):

#         augment_img = iaa.Sequential([

#             iaa.OneOf([

#                 iaa.Affine(rotate=0),

#                 iaa.Affine(rotate=90),

#                 iaa.Affine(rotate=180),

#                 iaa.Affine(rotate=270),

#                 iaa.Fliplr(0.5),

#                 iaa.Flipud(0.5),

#             ])], random_order=True)

        augment_img = seq



        image_aug = augment_img.augment_image(image)

        return image_aug
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Sequential, Model

from keras.layers import Dropout, Flatten, Dense, BatchNormalization, Input, Conv2D

from keras import applications

from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, CSVLogger

from keras.optimizers import Adam





from keras.applications.resnet50 import ResNet50

from keras.applications.inception_v3 import InceptionV3
def create_model(input_shape, n_out):

    input_tensor = Input(shape=input_shape)

    base_model = InceptionV3(include_top=False,

                   weights='imagenet',

                   input_shape=input_shape)

    bn = BatchNormalization()(input_tensor)

    x = base_model(bn)

    x = Conv2D(32, kernel_size=(1,1), activation='relu')(x)

    x = Flatten()(x)

    x = Dropout(0.5)(x)

    x = Dense(256, activation='relu')(x) # 1024

    x = Dropout(0.5)(x)

    output = Dense(n_out, activation='sigmoid')(x)

    model = Model(input_tensor, output)

    

    return model
best_checkpoint = ModelCheckpoint(config.path_to_model,

                             monitor='val_loss',

                             verbose=1, 

                             save_best_only=True,

                             mode='min',

                             save_weights_only = False)



latest_checkpoint = ModelCheckpoint(config.path_to_latest_model,

                                    monitor='val_loss',

                                    verbose=1, 

                                    save_best_only=False,

                                    mode='min',

                                    save_weights_only=False)



reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss',

                                   factor=0.1,

                                   patience=3, 

                                   verbose=1,

                                   mode='auto',

                                   epsilon=0.0001)



early = EarlyStopping(monitor="val_loss", 

                      mode="min", 

                      patience=6)



csv_logger = CSVLogger(config.log_file, append=True)



callbacks_list = [

    best_checkpoint,

    latest_checkpoint,

    early,

    reduceLROnPlat,

    csv_logger

    

]



warmup_callbacks_list = [

    best_checkpoint,

    latest_checkpoint,

    csv_logger

]
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

mskf = MultilabelStratifiedKFold(n_splits=5, random_state=0)



folds = mskf.split(data_df, data_df[class_list])

for train_idxs, val_idxs in folds:

    break

    

print("Num train: {}".format(len(train_idxs)))

print("Num val  : {}".format(len(val_idxs)))

print("First train idx: {}".format(train_idxs[0])) # 0

print("First val idx: {}".format(val_idxs[0])) # 3
train_generator = ImageGenerator.create(

    data_df.iloc[train_idxs],

    augument=True

)



val_generator = ImageGenerator.create(

    data_df.iloc[val_idxs],

    augument=False

)
latest_model_path = os.path.join(data_dir, model_dir, config.path_to_latest_model)

log_file_path = os.path.join(data_dir, model_dir, config.log_file)





if os.path.isfile(latest_model_path):

    print("Load from previously trained model...")

    model = load_model(latest_model_path)

    result = pd.read_csv(log_file_path)

    initial_epoch = int(result.epoch.iloc[-1]) + 1  # len(result)

    print("Trained {} epochs.".format(initial_epoch))

    

else:

    print("Init model and warm it up...")



    model = create_model(

        input_shape=(config.img_width, config.img_height, 3),

        n_out=config.num_classes

    )



    # warm up model

    for layer in model.layers:

        layer.trainable = False

    model.layers[-1].trainable = True

    model.layers[-2].trainable = True

    model.layers[-3].trainable = True

    model.layers[-4].trainable = True

    model.layers[-5].trainable = True

    model.layers[-6].trainable = True



    model.compile(

        loss='binary_crossentropy', 

        optimizer=Adam(lr=1e-03),

        metrics=['acc'])



    # model.summary()

    model.fit_generator(

        train_generator,

        steps_per_epoch=np.ceil(float(len(train_idxs)) / config.batch_size),

        validation_data=val_generator,

        validation_steps=np.ceil(float(len(val_idxs)) / config.batch_size),

        epochs=config.warmup_epochs,

        verbose=config.verbose,

        callbacks=warmup_callbacks_list)

    

    initial_epoch = config.warmup_epochs

    config.epochs -= config.warmup_epochs
print("Train full model...")

# train all layers

for layer in model.layers:

    layer.trainable = True



model.compile(loss='binary_crossentropy',

            optimizer=Adam(lr=1e-4),

            metrics=['accuracy'])



model.fit_generator(

    train_generator,

    steps_per_epoch=np.ceil(float(len(train_idxs)) / float(config.batch_size)),

    validation_data=val_generator,

    validation_steps=np.ceil(float(len(val_idxs)) / float(config.batch_size)),

    epochs=initial_epoch+config.epochs, 

    verbose=config.verbose,

    callbacks=callbacks_list,

    initial_epoch=initial_epoch)
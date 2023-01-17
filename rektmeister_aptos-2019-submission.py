import os

import math



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt





# set to false in submission notebook

TRAINING = False
# !pip install kaggle
# download data from Kaggle



# from googleapiclient.discovery import build

# import io, os

# from googleapiclient.http import MediaIoBaseDownload

# from google.colab import auth



# auth.authenticate_user()

# drive_service = build('drive', 'v3')

# results = drive_service.files().list(

#         q="name = 'kaggle.json'", fields="files(id)").execute()

# kaggle_api_key = results.get('files', [])

# filename = "/content/.kaggle/kaggle.json"

# os.makedirs(os.path.dirname(filename), exist_ok=True)

# request = drive_service.files().get_media(fileId=kaggle_api_key[0]['id'])

# fh = io.FileIO(filename, 'wb')

# downloader = MediaIoBaseDownload(fh, request)

# done = False

# while done is False:

#     status, done = downloader.next_chunk()

#     print("Download %d%%." % int(status.progress() * 100))

# os.chmod(filename, 600)
# !mkdir ~/.kaggle

# !cp /content/.kaggle/kaggle.json ~/.kaggle/kaggle.json
# !kaggle competitions download -c aptos2019-blindness-detection
# !unzip -q train_images.zip -d train_images

# !unzip -q test_images.zip -d test_images
!ls ../input/aptos-2019-densenet121-weights
# load CSV files



train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')

test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')



print('Number of train samples: ', train.shape[0])

print('Number of test samples: ', test.shape[0])
# set id codes to full image name (add .png)

train['id_code'] = train['id_code'].apply(lambda x: str(x) + ".png")

test['id_code'] = test['id_code'].apply(lambda x: str(x) + ".png")

train['diagnosis'] = train['diagnosis'].astype(str)



# add label columns

label_cols = ['lbl_0', 'lbl_1', 'lbl_2', 'lbl_3', 'lbl_4']

label_mat = np.zeros((train.shape[0], len(label_cols)), dtype=np.int32)



for i in range(train.shape[0]):

    for j in range(int(train['diagnosis'][i]) + 1):

        label_mat[i, j] = 1



train = pd.concat([train, pd.DataFrame(label_mat, columns=label_cols)], axis=1)



print(train.head(10))
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator



IMG_SIZE = 224

NB_CHANNELS = 3

NB_CLASSES = 5     # 0, 1, 2, 3, 4

BATCH_SIZE = 32

TEST_BATCH_SIZE = 1
"""

crops black parts around the image (intensity is <= tol)

"""

def crop_image(img, tol=10):

    

    # for one channel

    def crop_image_1(img):

        mask = img > tol

        return img[np.ix_(mask.any(1), mask.any(0))]

    

    if img.ndim == 2:

        return crop_image_1(img)

    

    elif img.ndim == 3:

        try:

            img_cpy = img.copy()

            h, w, _ = img.shape

            img1 = cv2.resize(crop_image_1(img[:, :, 0]), (w, h))

            img2 = cv2.resize(crop_image_1(img[:, :, 1]), (w, h))

            img3 = cv2.resize(crop_image_1(img[:, :, 2]), (w, h))



            img[:,:,0] = img1

            img[:,:,1] = img2

            img[:,:,2] = img3

            

        except:

            return img_cpy

        

        

        """

        # add edges to img

        

        sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=5)

        sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=5)

        sobelx = cv2.cvtColor(sobelx, cv2.COLOR_RGB2GRAY)

        sobely = cv2.cvtColor(sobely, cv2.COLOR_RGB2GRAY)

        

        img4 = sobelx ** 2 + sobely ** 2

        

        img4 -= np.min(img4)

        img4 = img4 / np.max(img4)

        

        img4 *= 255.

        img4.astype(np.uint8)

        """

        

        return img





"""

crops black parts and enhances image (Ben Graham's method)

"""

def preprocess_image(img):

    

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = crop_image(img)

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), IMG_SIZE/10), -4, 128)

    

    return img
# looking at raw/processed image



# GRID_ROWS = 6

# GRID_COLS = 6



# rand_idx = np.random.randint(0, len(train['id_code']), size=(GRID_ROWS*GRID_COLS//2))



# fig, ax = plt.subplots(GRID_ROWS, GRID_COLS, figsize=(GRID_COLS * 2.5, GRID_ROWS * 2.5))



# for i in range(GRID_ROWS):

#     for j in range(GRID_COLS // 2):

#         test_img = cv2.imread(f"train_images/{train['id_code'][rand_idx[i * GRID_COLS // 2 + j]]}")

        

#         ax[i, j * 2].imshow(test_img)

#         ax[i, j * 2].axis("off")

        

#         test_img = preprocess_image(test_img)

        

#         ax[i, j * 2 + 1].set_title(f"diagnosis: {train['diagnosis'][rand_idx[i * GRID_COLS // 2 + j]]}")

#         ax[i, j * 2 + 1].imshow(test_img)

#         ax[i, j * 2 + 1].axis("off")
train_datagen = ImageDataGenerator(rescale=1./255,

                                   validation_split=0.2,

                                   horizontal_flip=True,

                                   preprocessing_function=preprocess_image)



train_gen = train_datagen.flow_from_dataframe(

    dataframe=train,

    directory="../input/aptos2019-blindness-detection/train_images/",

    x_col='id_code',

#     y_col='diagnosis',

    y_col=label_cols,

    batch_size=BATCH_SIZE,

#     class_mode="categorical",

    class_mode="other",

    target_size=(IMG_SIZE, IMG_SIZE),

    subset="training",

    shuffle=True

)



val_gen = train_datagen.flow_from_dataframe(

    dataframe=train,

    directory="../input/aptos2019-blindness-detection/train_images/",

    x_col='id_code',

#     y_col='diagnosis',

    y_col=label_cols,

    batch_size=BATCH_SIZE,

#     class_mode="categorical",

    class_mode="other",

    target_size=(IMG_SIZE, IMG_SIZE),

    subset="validation",

    shuffle=True

)





test_datagen = ImageDataGenerator(rescale=1./255,

                                  preprocessing_function=preprocess_image)



test_gen = test_datagen.flow_from_dataframe(  

    dataframe=test,

    directory="../input/aptos2019-blindness-detection/test_images/",

    x_col='id_code',

    batch_size=TEST_BATCH_SIZE,

    class_mode=None,

    target_size=(IMG_SIZE, IMG_SIZE),

    shuffle=False,

)
import tensorflow as tf

from tensorflow.keras.models import Model

from tensorflow.keras.layers import (Input,

                                            GlobalAveragePooling2D,

                                            Dense,

                                            Dropout,

                                            BatchNormalization,

                                            Conv2D,

                                            MaxPooling2D)

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.callbacks import (CSVLogger,

                                        ModelCheckpoint,

                                        EarlyStopping)

# from tensorflow.keras.metrics import Metric

from tensorflow.keras.applications.resnet50 import ResNet50

from tensorflow.keras.applications.densenet import DenseNet121



from sklearn.metrics import cohen_kappa_score





MODEL_NAME = 'densenet121'



NB_WARMUP_EPOCHS = 2

NB_EPOCHS = 30

INITIAL_LR = 1e-3





weights_path_template = os.path.join("../input/aptos-2019-densenet121-weights/", "{}_weights.hdf5")

log_path_template     = os.path.join("logs/", "{}_training_log.csv")
# for creating weights dir

!mkdir -p weights

!mkdir -p logs
# # check tf version

# print("tf version:", tf.VERSION)



# # check GPU

# device_name = tf.test.gpu_device_name()

# if "GPU" not in device_name:

#     print("GPU device not found")

# else:

#     print('Found GPU at: {}'.format(device_name))
"""

ResNet50 based model

"""

def get_resnet50(input_shape, nb_out):

    inputs = Input(shape=input_shape)



    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)



    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)



    x = Dense(2048, activation='relu')(x)

    x = Dropout(0.5)(x)



    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.5)(x)



    output = Dense(nb_out, activation='softmax', name='final_output')(x)



    model = Model(inputs, output)



    return model
def get_densenet121(input_shape, nb_out):

    inputs = Input(shape=input_shape)



    base_model = DenseNet121(weights=None, include_top=False, input_tensor=inputs)



    x = GlobalAveragePooling2D()(base_model.output)

    x = Dropout(0.5)(x)



    x = Dense(1024, activation='relu')(x)



    output = Dense(nb_out, activation='sigmoid')(x)



    model = Model(inputs, output)



    return model
"""

simple CNN (conv1)

"""

def get_conv1(input_shape, nb_out):

    inputs = Input(shape=input_shape)



    x = Conv2D(64, (7, 7), activation='relu')(inputs)

    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)

    

    x = Conv2D(64, (7, 7), activation='relu')(x)

    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)

    

    x = Conv2D(128, (5, 5), activation='relu')(x)

    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)

    

    x = Conv2D(256, (3, 3), activation='relu')(x)

    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)

    

    x = Conv2D(512, (3, 3), activation='relu')(x)

    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)

    

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)

    

    x = Dense(2048, activation='relu')(x)

    x = Dropout(0.5)(x)

    

    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.5)(x)



    output = Dense(nb_out, activation='softmax', name='final_output')(x)



    model = Model(inputs, output)



    return model
"""

simple CNN v2 (conv2)

"""

def get_conv2(input_shape, nb_out):

    inputs = Input(shape=input_shape)



    x = Conv2D(32, (7, 7), activation='relu')(inputs)

    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)

    

    x = Conv2D(64, (5, 5), activation='relu')(x)

    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)

    

    x = Conv2D(128, (3, 3), activation='relu')(x)

    x = MaxPooling2D((2, 2))(x)

    x = BatchNormalization()(x)

    

    x = GlobalAveragePooling2D()(x)

    x = Dropout(0.5)(x)



    x = Dense(1024, activation='relu')(x)

    x = Dropout(0.5)(x)



    output = Dense(nb_out, activation='softmax')(x)



    model = Model(inputs, output)



    return model
"""

returns model

"""

def get_model(name, input_shape, nb_out):

    

    models = {

        'resnet50':    get_resnet50,

        'conv1':       get_conv1,

        'conv2':       get_conv2,

        'densenet121': get_densenet121

    }

    

    if name not in models:

        print(f"No model named '{name}'")

        return

    

    model = models[name](input_shape, nb_out)

    

    weights_path = weights_path_template.format(name)

    if os.path.isfile(weights_path):

        model.load_weights(weights_path)

        print(f"loaded model weights from {weights_path}")

        

    return model
# class qwk_metric(Metric):



#     def __init__(self, name='qwk', **kwargs):

#         super(qwk_metric, self).__init__(name=name, **kwargs)

#         self.kappa = None





#     def update_state(self, y_true, y_pred, sample_weight=None):

#         labels = tf.reduce_sum(tf.cast(y_true, tf.int32), axis=1) - 1

        

#         self.kappa = tf.contrib.metrics.cohen_kappa(

            

#         )





#     def result(self):

#         return self.true_positives





#     def reset_states(self):

#         # The state of the metric will be reset at the start of each epoch.

#         self.true_positives.assign(0.)
"""

trains a ResNet50-based model

"""

def train_resnet50(model, train_generator, val_generator, weights_path, log_path):

    

    # ==== train top layers only (warm up) ====

    

    for i in range(len(model.layers)):

        model.layers[i].trainable = False



    for i in range(-5, 0):

        model.layers[i].trainable = True



    metrics_list = ["accuracy"]

    optimizer = Adam(lr=INITIAL_LR)



    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics_list)

    

    mc = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1)

    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)

    cl = CSVLogger(log_path)



    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size



    model.fit_generator(

        generator=train_generator,

        steps_per_epoch=STEP_SIZE_TRAIN,

        validation_data=val_generator,

        validation_steps=STEP_SIZE_VAL,

        epochs=NB_WARMUP_EPOCHS,

        callbacks=[mc, cl],

        verbose=1

    )

    

    

    # ==== fine tune all layers ====

    

    train_gen.reset()

    val_gen.reset()

    

    for i in range(len(model.layers)):

        model.layers[i].trainable = True

    

    optimizer = Adam(lr=INITIAL_LR)



    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics_list)

    

    mc = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1)

    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)

    cl = CSVLogger(log_path)



    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size



    model.fit_generator(

        generator=train_generator,

        steps_per_epoch=STEP_SIZE_TRAIN,

        validation_data=val_generator,

        validation_steps=STEP_SIZE_VAL,

        epochs=NB_WARMUP_EPOCHS,

        callbacks=[mc, cl],

        verbose=1

    )
def train_densenet121(model, train_generator, val_generator, weights_path, log_path):

    

    metrics_list = ["accuracy"]

    

    optimizer = Adam(lr=INITIAL_LR)



    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics_list)

    

    mc = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1)

    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)

    cl = CSVLogger(log_path)



    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size



    model.fit_generator(

        generator=train_generator,

        steps_per_epoch=STEP_SIZE_TRAIN,

        validation_data=val_generator,

        validation_steps=STEP_SIZE_VAL,

        epochs=NB_EPOCHS,

        callbacks=[mc, cl],

        verbose=1

    )
"""

trains the simple CNN

"""

def train_conv1(model, train_generator, val_generator, weights_path, log_path):

    

    metrics_list = ["accuracy"]

    optimizer = Adam(lr=INITIAL_LR)



    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics_list)

    

    mc = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1)

    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)

    cl = CSVLogger(log_path)



    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size



    model.fit_generator(

        generator=train_generator,

        steps_per_epoch=STEP_SIZE_TRAIN,

        validation_data=val_generator,

        validation_steps=STEP_SIZE_VAL,

        epochs=NB_EPOCHS,

        callbacks=[mc, cl],

        verbose=1

    )
def train_conv2(model, train_generator, val_generator, weights_path, log_path):

        

    metrics_list = ["accuracy"]

    optimizer = Adam(lr=INITIAL_LR)



    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics_list)

    

    mc = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1)

    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)

    cl = CSVLogger(log_path)



    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size



    model.fit_generator(

        generator=train_generator,

        steps_per_epoch=STEP_SIZE_TRAIN,

        validation_data=val_generator,

        validation_steps=STEP_SIZE_VAL,

        epochs=NB_EPOCHS,

        callbacks=[mc, cl],

        verbose=1

    )
def train_model(name, input_shape, nb_out, train_generator, val_generator):

    model = get_model(name, input_shape, nb_out)



    trainers = {

        'resnet50':    train_resnet50,

        'conv1':       train_conv1,

        'conv2':       train_conv2,

        'densenet121': train_densenet121

    }



    if name not in trainers:

        print(f"No model named '{name}'")

        return



    trainers[name](model, train_generator, val_generator,

                   weights_path_template.format(name),

                   log_path_template.format(name))
if TRAINING:

    train_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES, train_gen, val_gen)
model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)

test_gen.reset()

# STEP_SIZE_TEST = math.ceil(test_gen.n / test_gen.batch_size)



preds = model.predict_generator(test_gen, verbose=1)

# predictions = np.argmax(preds, axis=1)

preds = preds > 0.5

predictions = preds.astype(int).sum(axis=1) - 1



filenames = test_gen.filenames

results = pd.DataFrame({'id_code': filenames, 'diagnosis': predictions})

results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])   # remove .png

results.to_csv('submission.csv', index=False)

# results.head(10)
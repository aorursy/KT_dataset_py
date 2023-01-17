!pip install efficientnet==0.0.4

!pip install git+https://github.com/aleju/imgaug.git
import os

from keras.applications.xception import Xception  # (299 x 299)

from keras.applications.inception_v3 import InceptionV3  # (299 x 299)

from keras.applications.inception_resnet_v2 import InceptionResNetV2  # (299 x 299)

from efficientnet import EfficientNetB4, EfficientNetB5  # (299 x 299)





MODEL_N = 0  # 0 ~ 19 (!important)

MODEL_N_STR = '0' + str(MODEL_N) if len(str(MODEL_N)) == 1 else str(MODEL_N)

MODEL_LIST = (

    Xception, InceptionV3, InceptionResNetV2, EfficientNetB4, EfficientNetB5,

    Xception, InceptionV3, InceptionResNetV2, EfficientNetB4, EfficientNetB5,

    Xception, InceptionV3, InceptionResNetV2, EfficientNetB4, EfficientNetB5,

    Xception, InceptionV3, InceptionResNetV2, EfficientNetB4, EfficientNetB5,

)

PRETAINED_MODEL = MODEL_LIST[MODEL_N]

if MODEL_N in (0, 5, 10, 15): PRETAINED_MODEL_STR = 'Xception'

elif MODEL_N in (1, 6, 11, 16): PRETAINED_MODEL_STR = 'InceptionV3'

elif MODEL_N in (2, 7, 12, 17): PRETAINED_MODEL_STR = 'InceptionResNetV2'

elif MODEL_N in (3, 8, 13, 18): PRETAINED_MODEL_STR = 'EfficientNetB4'

elif MODEL_N in (4, 9, 14, 19): PRETAINED_MODEL_STR = 'EfficientNetB5'



# Ensemble Mode

INPUT_PATH_ORIGIN = os.path.join('..', 'input')

UPLOADED_MODEL_PATH = os.path.join(

    os.path.join(INPUT_PATH_ORIGIN, 'kakr-2019-3rd-cutmix-saved-models'),

    'kakr_2019_3rd_cutmix_saved_models'

)

UPLOADED_MODEL_PATH = os.path.join(UPLOADED_MODEL_PATH, 'kakr_2019_3rd_cutmix_saved_models')

ENSEMBLE_MODE = True  #  (!importrant)

if ENSEMBLE_MODE and not os.path.isdir(UPLOADED_MODEL_PATH):

    raise  # ENSEMBLE_MODE = True, but no uploaded model files



# Full Data Mode

# If False, Lightweight Data Mode For Test (Get 20% Of Data)

FULL_DATA = True

if FULL_DATA: LIGHT_DATA_TRAIN, LIGHT_DATA_TEST = False, False

else: LIGHT_DATA_TRAIN, LIGHT_DATA_TEST = True, True

    

# Size Of Pretained Model (IMG_SIZE x IMG_SIZE)

IMG_SIZE = 299
import gc

import random

import warnings

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import cv2

import PIL

from PIL import ImageDraw

from keras import regularizers

from keras import models, layers, optimizers

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.utils.class_weight import compute_class_weight

import imgaug as ia

from imgaug import augmenters as iaa





# Ignore Warnings

warnings.filterwarnings('ignore')



# Class Weights

GET_CLASS_WEIGHTS = False



# Undersampling class 119 for class balance

UNDERSAMPLING_119 = False



# If EfficientNet

is_EfficientNet = PRETAINED_MODEL_STR in ('EfficientNetB4', 'EfficientNetB5')



# Constant Variables

LEARNING_RATE = 1e-4

OPTIMIZER = optimizers.Adam(lr=LEARNING_RATE)

EPOCHS = 80 if FULL_DATA else 4

EPOCHS = 64 if (FULL_DATA and is_EfficientNet) else EPOCHS

BATCH_SIZE = 16 # if set high value, it would occur ResourceExhaustedError

INITIALIZER = 'he_normal'

REGULARIZER = regularizers.l2(1e-3)

DROPOUT_RATE = 0.5

PATIENCE = 12 if is_EfficientNet else 16  # Patience Value For Early Stop

VERBOSE = 2 if FULL_DATA else 1  # Verbosity mode (2: No progress bar)

RANDOM_SEED = 2019

ia.seed(RANDOM_SEED)



# Set Basic Path

INPUT_PATH = os.path.join('..', 'input')

if os.path.isdir(os.path.join(INPUT_PATH, '2019-3rd-ml-month-with-kakr')):

    INPUT_PATH = os.path.join(INPUT_PATH, '2019-3rd-ml-month-with-kakr')

TRAIN_IMG_PATH = os.path.join(INPUT_PATH, 'train')

TEST_IMG_PATH = os.path.join(INPUT_PATH, 'test')



# Set Path of Cropped Train Images

TRAIN_IMG_CROP_PATH = os.path.join('..', 'train_crop')

if not os.path.exists(TRAIN_IMG_CROP_PATH):

    os.makedirs(TRAIN_IMG_CROP_PATH, exist_ok=True)



# Set Path of Cropped Test Images

TEST_IMG_CROP_PATH = os.path.join('..', 'test_crop')

if not os.path.exists(TEST_IMG_CROP_PATH):

    os.makedirs(TEST_IMG_CROP_PATH, exist_ok=True)

    

# Set Path of Histogram Equalized Train Images

TRAIN_IMG_PREP_PATH = os.path.join('..', 'train_prep')

if not os.path.exists(TRAIN_IMG_PREP_PATH):

    os.makedirs(TRAIN_IMG_PREP_PATH, exist_ok=True)



# Set Path of Histogram Equalized Test Images

TEST_IMG_PREP_PATH = os.path.join('..', 'test_prep')

if not os.path.exists(TEST_IMG_PREP_PATH):

    os.makedirs(TEST_IMG_PREP_PATH, exist_ok=True)



# Set Path of Model

MODEL_FILE_SAVE_DIR_PATH = '.'

if not os.path.exists(MODEL_FILE_SAVE_DIR_PATH):

    os.makedirs(MODEL_FILE_SAVE_PATH, exist_ok=True)

MODEL_FILE_PATH = os.path.join(

    MODEL_FILE_SAVE_DIR_PATH,

    MODEL_N_STR + '_' + PRETAINED_MODEL_STR + '.hdf5'

)



# Load DataFrame (Read CSV)

df_class = pd.read_csv(os.path.join(INPUT_PATH, 'class.csv'))  # Label of class column (df_train)

df_train = pd.read_csv(os.path.join(INPUT_PATH, 'train.csv'))

df_test = pd.read_csv(os.path.join(INPUT_PATH, 'test.csv'))

# df_class: |    id    |  name   |

# df_train: | img_file | bbox_x1 | bbox_y1 | bbox_x2 | bbox_y2 | class |

# df_test:  | img_file | bbox_x1 | bbox_y1 | bbox_x2 | bbox_y2 |       |



# Number Of Class

CLASS_NB = len(df_class)  # == 196
from datetime import datetime

from pytz import timezone, utc





KST = timezone('Asia/Seoul')



def system_print(string):  

    os.system(f'echo \"{string}\"')

    print(string)



# To Add CallBack List

class PrintSystemLogPerEpoch(Callback):

    def on_epoch_begin(self, epoch, logs={}):

        t = utc.localize(datetime.utcnow()).astimezone(KST).time()

        system_print(f'* [Epoch {epoch+1}] begins at {t}')

    def on_epoch_end(self, epoch, logs={}):

        t = utc.localize(datetime.utcnow()).astimezone(KST).time()

        system_print(f'* [Epoch {epoch+1}] ends at {t} | acc={logs["acc"]:0.4f}, val_acc={logs["val_acc"]:0.4f}')
# Get 20% Of Original Train Data Set

if LIGHT_DATA_TRAIN:

    _, df_train = train_test_split(

        df_train,

        train_size=0.8,

        random_state=RANDOM_SEED,

        stratify=df_train['class'],

    )



# Get 10% Of Original Test Data Set

if LIGHT_DATA_TEST:

    _, df_test = train_test_split(

        df_test,

        train_size=0.8,

        random_state=RANDOM_SEED,

    )
print('------ df_class ---' + '-' * 15)

df_class.info()

print('\n------ df_train ---' + '-' * 15)

df_train.info()

print('\n------ df_test ---' + '-' * 15)

df_test.info()
# Check train image files missing

if set(df_train.img_file) == set(os.listdir(TRAIN_IMG_PATH)):  # Ignore ordering with set

    print("Train image files are no Problem")

else: print("There is some missing train image files")



# Check test image files missing

if set(df_test.img_file) == set(os.listdir(TEST_IMG_PATH)):

    print("Test image files are no Problem")

else: print("There is some missing test image files")
car_class_nb = df_class.shape[0]

train_class_nb = df_train['class'].nunique()

print(f'Number of car class: {car_class_nb}')

print(f'Number of train data class: {train_class_nb}')



if car_class_nb == train_class_nb:

    print("No problem")
train_nb_per_class = df_train['class'].value_counts(ascending=True)

train_nb_per_class.describe()
print(f'Min Count Class: {train_nb_per_class.index[0]}')

print(f'Max Count Class: {train_nb_per_class.index[-1]}')
plt.figure(figsize=(16,8))

plt.title('Number of train data per class')

sns.countplot(df_train['class'])

plt.show()
random.seed(RANDOM_SEED)



if UNDERSAMPLING_119:

    class_119_idx_bool = (df_train['class'] == 119)

    class_119_idx_nb = sum(class_119_idx_bool)

    class_119_idx_list = list(df_train[class_119_idx_bool].index)

    class_119_undersampling_idx = random.sample(class_119_idx_list, int(class_119_idx_nb*0.25))



    # Delete 22 rows(119 Class) For Undersampling

    for idx in class_119_undersampling_idx:

        df_train = df_train.drop(idx)



    df_train = df_train.reset_index(drop=True)

    

    # Show Undersampling Result

    plt.figure(figsize=(16,8))

    plt.title('Number of train data per class After Undersampling')

    sns.countplot(df_train['class'])

    plt.show()
if GET_CLASS_WEIGHTS:

    class_weights = compute_class_weight('balanced', np.unique(df_train['class']), df_train['class'])

    class_weights_dict = dict(zip(np.unique(df_train['class']), class_weights))
tmp_imgs = df_train['img_file'][:4]  # Load 4 images (0, 1, 2, 3)

plt.figure(figsize=(14,10))



for idx, img_file_name in enumerate(tmp_imgs, 1):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, img_file_name))

    plt.subplot(2, 2, idx)

    plt.title(img_file_name)

    plt.imshow(img)

    plt.axis('off')
def img_crop_resize(img_file_name, resize_val, margin=6):

    if img_file_name.split('_')[0] == 'train':

        IMG_PATH = TRAIN_IMG_PATH

        data = df_train

    elif img_file_name.split('_')[0] == 'test':

        IMG_PATH = TEST_IMG_PATH

        data = df_test

        

    img = PIL.Image.open(os.path.join(IMG_PATH, img_file_name))

    pos = data.loc[data['img_file'] == img_file_name, ['bbox_x1','bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)

    width, height = img.size

    x1 = max(0, pos[0] - margin)

    y1 = max(0, pos[1] - margin)

    x2 = min(pos[2] + margin, width)

    y2 = min(pos[3] + margin, height)

    return img.crop((x1, y1, x2, y2)).resize(resize_val)





# Define Function For Test

def test_crop(img_file_name):

    # Show Original Image

    img_origin = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, img_file_name))

    plt.figure(figsize=(16,12))

    plt.subplot(1, 2, 1)

    plt.title(f'Original Image - {img_file_name}')

    plt.imshow(img_origin)

    plt.axis('off')

    

    # Show Crop Image

    img_crop = img_crop_resize(img_file_name, resize_val=(IMG_SIZE, IMG_SIZE))

    plt.subplot(1, 2, 2)

    plt.title(f'Cropped & Resized Image - {img_file_name}')

    plt.imshow(img_crop)

    plt.axis('off')

    

    # Show Result

    plt.show()
# Test bbox and crop

test_crop(img_file_name=df_train['img_file'].iloc[100])
# Save Cropped Train Images (Path: /kaggle/working/train_crop)

if not os.listdir(TRAIN_IMG_CROP_PATH):  # If PATH_IMG_TRAIN_CROP is empty

    for idx, row in df_train.iterrows():

        img_file_name = row['img_file']

        img_crop = img_crop_resize(img_file_name, resize_val=(IMG_SIZE, IMG_SIZE))

        img_crop.save(os.path.join(TRAIN_IMG_CROP_PATH, img_file_name))



# Save Cropped Test Images (Path: /kaggle/working/test_crop)

if not os.listdir(TEST_IMG_CROP_PATH):

    for idx, row in df_test.iterrows():

        img_file_name = row['img_file']

        img_crop = img_crop_resize(img_file_name, resize_val=(IMG_SIZE, IMG_SIZE))

        img_crop.save(os.path.join(TEST_IMG_CROP_PATH, img_file_name))
def img_clahe(img_file_name):

    if img_file_name.split('_')[0] == 'train':

        IMG_CROP_PATH = TRAIN_IMG_CROP_PATH

        data = df_train

    elif img_file_name.split('_')[0] == 'test':

        IMG_CROP_PATH = TEST_IMG_CROP_PATH

        data = df_test

        

    # --- Histogram Equalization --- #

    img = cv2.imread(os.path.join(IMG_CROP_PATH, img_file_name))

    img_y_cr_cb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    y, cr, cb = cv2.split(img_y_cr_cb)



    # Equalize y (CLAHE (Contrast Limited Adaptive Histogram Equalization))

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(6,6))

    y_eq = clahe.apply(y)

    img_y_cr_cb_eq = cv2.merge((y_eq, cr, cb))

    img_bgr_eq = cv2.cvtColor(img_y_cr_cb_eq, cv2.COLOR_YCR_CB2BGR)

    img_prep = img_bgr_eq



    # --- Convert BGR To RGB (Just On cv2) ---

    # b, g, r = cv2.split(img_bgr_eq)

    # img_prep = cv2.merge((r,g,b))

    return img_prep
# Save CLAHE Train Images (Path: ../train_prep)

if not os.listdir(TRAIN_IMG_PREP_PATH):  # If PATH_IMG_TRAIN_PREP is empty

    for idx, row in df_train.iterrows():

        img_file_name = row['img_file']

        img_prep = img_clahe(img_file_name)

        cv2.imwrite(os.path.join(TRAIN_IMG_PREP_PATH, img_file_name), img_prep)



# Save CLAHE Test Images (Path: ../test_prep)

if not os.listdir(TEST_IMG_PREP_PATH):

    for idx, row in df_test.iterrows():

        img_file_name = row['img_file']

        img_prep = img_clahe(img_file_name)

        cv2.imwrite(os.path.join(TEST_IMG_PREP_PATH, img_file_name), img_prep)
def test_clahe(img_file_name):

    # Show Cropped Image

    img_crop = PIL.Image.open(os.path.join(TRAIN_IMG_CROP_PATH, img_file_name))

    plt.figure(figsize=(12, 9))

    plt.subplot(1, 2, 1)

    plt.title(f'Cropped & Resized Image - {img_file_name}')

    plt.imshow(img_crop)

    plt.axis('off')



    # Show CLAHE Image

    img_clahe = PIL.Image.open(os.path.join(TRAIN_IMG_PREP_PATH, img_file_name))

    plt.subplot(1, 2, 2)

    plt.title(f'CLAHE - {img_file_name}')

    plt.imshow(img_clahe)

    plt.axis('off')

    

    # Show Result

    plt.show()

    

# Test CLAHE Image

test_clahe(img_file_name=df_train['img_file'].iloc[114])
# class_mode="categorical", y_col="class" column values must be type string (flow method options)

df_train['class'] = df_train['class'].astype('str')



# Take Necessary Columns

df_train = df_train[['img_file', 'class']]



# K-Fold Cross Validation

idx_split = list()

skf = StratifiedKFold(n_splits=20, random_state=RANDOM_SEED)

for train_idx, val_idx in skf.split(X=df_train['img_file'], y=df_train['class']):

    idx_split.append((train_idx, val_idx))

    

X_train = df_train.iloc[idx_split[MODEL_N][0]].reset_index()

X_val = df_train.iloc[idx_split[MODEL_N][1]].reset_index()

X_test = df_test[['img_file']]



# Delete Unncessary Data For Memory

del df_train

del df_test

gc.collect()
# Define Get Model Function

def get_model(pretained_model, img_size, initializer, regularizer, optimizer, dropout_rate, class_nb):

    model = models.Sequential()

    model.add(

        pretained_model(

            include_top=False,

            weights='imagenet',

            input_shape=(img_size, img_size, 3)

        )

    )

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(

        1024,

        activation='relu',

        kernel_initializer=initializer,

        kernel_regularizer=regularizer,

    ))

    model.add(layers.Dropout(dropout_rate))

    model.add(layers.Dense(class_nb, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    return model





# Define Get Step Function

def get_steps(num_data, batch_size):

    quotient, remainder = divmod(num_data, batch_size)

    return (quotient + 1) if remainder else quotient
# Define Get Callback Function

def get_callback_list(model_file_path, patience, verbose, system_print=False):

    early_stop = EarlyStopping(

        monitor='val_loss',

        patience=patience,

        verbose=verbose,

        mode='min',

    )

    reduce_lr = ReduceLROnPlateau(

        monitor='val_loss',

        factor=0.5,

        patience=patience//2,

        verbose=verbose,

        mode='min',

        min_lr=1e-6,

    )

    model_chk = ModelCheckpoint(

        filepath=model_file_path,

        monitor='val_loss',

        verbose=verbose,

        save_best_only=True,

        mode='min',

    )

    callback_list = [early_stop, reduce_lr, model_chk]

    if system_print:

        callback_list.append(PrintSystemLogPerEpoch())

    return callback_list
# Set imgaug Sequential

seq = iaa.Sequential(

    [

        iaa.Affine(rotate=(-15, 15)),

        iaa.Fliplr(0.5),

        iaa.GaussianBlur((0, 2.0)),

        iaa.ElasticTransformation(alpha=(0, 70), sigma=9),

        iaa.AdditiveGaussianNoise(scale=(0, 0.05), per_channel=True),

        iaa.ChannelShuffle(p=0.5),

    ],

    random_order=False,

)



class CutMixImageDataGenerator():

    def __init__(self, generator1, generator2, img_size, batch_size):

        self.batch_index = 0

        self.samples = generator1.samples

        self.class_indices = generator1.class_indices

        self.generator1 = generator1

        self.generator2 = generator2

        self.img_size = img_size

        self.batch_size = batch_size



    def reset_index(self):  # Ordering Reset (If Shuffle is True, Shuffle Again)

        self.generator1._set_index_array()

        self.generator2._set_index_array()



    def reset(self):

        self.batch_index = 0

        self.generator1.reset()

        self.generator2.reset()

        self.reset_index()



    def get_steps_per_epoch(self):

        quotient, remainder = divmod(self.samples, self.batch_size)

        return (quotient + 1) if remainder else quotient

    

    def __len__(self):

        self.get_steps_per_epoch()



    def __next__(self):

        if self.batch_index == 0: self.reset()



        crt_idx = self.batch_index * self.batch_size

        if self.samples > crt_idx + self.batch_size:

            self.batch_index += 1

        else:  # If current index over number of samples

            self.batch_index = 0



        reshape_size = self.batch_size

        last_step_start_idx = (self.get_steps_per_epoch()-1) * self.batch_size

        if crt_idx == last_step_start_idx:

            reshape_size = self.samples - last_step_start_idx

            

        X_1, y_1 = self.generator1.next()

        X_2, y_2 = self.generator2.next()

        

        cut_ratio = np.random.beta(a=1, b=1, size=reshape_size)

        cut_ratio = np.clip(cut_ratio, 0.2, 0.8)

        label_ratio = cut_ratio.reshape(reshape_size, 1)

        cut_img = X_2



        X = X_1

        for i in range(reshape_size):

            cut_size = int((self.img_size-1) * cut_ratio[i])

            y1 = random.randint(0, (self.img_size-1) - cut_size)

            x1 = random.randint(0, (self.img_size-1) - cut_size)

            y2 = y1 + cut_size

            x2 = x1 + cut_size

            cut_arr = cut_img[i][y1:y2, x1:x2]

            cutmix_img = X_1[i]

            cutmix_img[y1:y2, x1:x2] = cut_arr

            X[i] = cutmix_img

            

        X = seq.augment_images(X)  # Sequential of imgaug

        y = y_1 * (1 - (label_ratio ** 2)) + y_2 * (label_ratio ** 2)

        return X, y



    def __iter__(self):

        while True:

            yield next(self)
# Settings For CutMix

train_datagen = ImageDataGenerator(

    brightness_range=(0.6, 1.4),

    horizontal_flip=True,

    vertical_flip=False,

    rescale=1./255,

)

train_generator1 = train_datagen.flow_from_dataframe(

    dataframe=X_train,

    directory=TRAIN_IMG_PREP_PATH,

    x_col='img_file',

    y_col='class',

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode='rgb',

    class_mode='categorical',

    batch_size=BATCH_SIZE,

    shuffle=True,

)

train_generator2 = train_datagen.flow_from_dataframe(

    dataframe=X_train,

    directory=TRAIN_IMG_PREP_PATH,

    x_col='img_file',

    y_col='class',

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode='rgb',

    class_mode='categorical',

    batch_size=BATCH_SIZE,

    shuffle=True,

)



# Train Generator (CutMix)

train_generator = CutMixImageDataGenerator(

    generator1=train_generator1,

    generator2=train_generator2,

    img_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

)



# Validation Generator

valid_datagen = ImageDataGenerator(rescale=1./255,)

valid_generator = valid_datagen.flow_from_dataframe(

    dataframe=X_val,

    directory=TRAIN_IMG_PREP_PATH,

    x_col='img_file',

    y_col='class',

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode='rgb',

    class_mode='categorical',

    batch_size=BATCH_SIZE,

    shuffle=False,

)



# Test Generator

test_datagen = ImageDataGenerator(rescale=1./255,)

test_generator = test_datagen.flow_from_dataframe(

    dataframe=X_test,

    directory=TEST_IMG_PREP_PATH,

    x_col='img_file',

    y_col=None,

    target_size=(IMG_SIZE, IMG_SIZE),

    color_mode='rgb',

    class_mode=None,

    batch_size=BATCH_SIZE,

    shuffle=False,

)
if not ENSEMBLE_MODE:

    # Get Model

    model = get_model(

        pretained_model=PRETAINED_MODEL,

        img_size=IMG_SIZE,

        initializer=INITIALIZER,

        regularizer=REGULARIZER,

        optimizer=OPTIMIZER,

        dropout_rate=DROPOUT_RATE,

        class_nb=CLASS_NB,

    )



    # Train With Generators

    history = model.fit_generator(

        generator=train_generator,

        steps_per_epoch=train_generator.get_steps_per_epoch(),

        epochs=EPOCHS,

        verbose=VERBOSE,

        callbacks=get_callback_list(

            model_file_path=MODEL_FILE_PATH,

            patience=PATIENCE,

            verbose=VERBOSE,

            system_print=True,

        ),

        validation_data=valid_generator,

        validation_steps=get_steps(valid_generator.samples, BATCH_SIZE),

        class_weight=class_weights_dict if GET_CLASS_WEIGHTS else None,

        shuffle=False,

    )

    

    # Visualization Loss & Accuracy

    history_info = history.history

    loss = history_info['loss']

    val_loss = history_info['val_loss']

    acc = history_info['acc']

    val_acc = history_info['val_acc']

    epochs = range(1, len(loss)+1)



    # Plot

    fig = plt.figure(figsize=(20, 4))



    fig.add_subplot(1, 2, 1)

    plt.plot(epochs, loss, 'b', label=f'Training Loss')

    plt.plot(epochs, val_loss,'r', label=f'Validation Loss')

    plt.title(f'Loss Per Epoch ({MODEL_N}.{PRETAINED_MODEL_STR})')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()



    fig.add_subplot(1, 2, 2)

    plt.plot(epochs, acc, 'b', label='Training Accuracy')

    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')

    plt.title(f'Accuracy Per Epoch ({MODEL_N}.{PRETAINED_MODEL_STR})')

    plt.xlabel('Epochs')

    plt.ylabel('Accuracy')

    plt.legend()



    # Show Visualization Result

    plt.show()

    

    # Prediction

    model.load_weights(MODEL_FILE_PATH)

    pred = model.predict_generator(

        generator=test_generator,

        steps=get_steps(len(X_test), BATCH_SIZE),

        verbose=VERBOSE,

    )

    pred_class_indices = np.argmax(pred, axis=1)
if ENSEMBLE_MODE:

    UPLOADED_MODEL_LIST = sorted(os.listdir(UPLOADED_MODEL_PATH))[:10]



    model_pred_list = list()

    for model_n, model_name in enumerate(UPLOADED_MODEL_LIST):

        

        # Load Model

        model_file_path = os.path.join(UPLOADED_MODEL_PATH, model_name)

        model = get_model(

            pretained_model=MODEL_LIST[model_n],

            img_size=IMG_SIZE,

            initializer=INITIALIZER,

            regularizer=REGULARIZER,

            optimizer=OPTIMIZER,

            dropout_rate=DROPOUT_RATE,

            class_nb=CLASS_NB,

        )

        model.load_weights(model_file_path)

        

        # Prediction

        test_generator.reset()

        pred = model.predict_generator(

            generator=test_generator,

            steps=get_steps(len(X_test), BATCH_SIZE),

            verbose=VERBOSE,

        )

        model_pred_list.append(pred)



    # Model (0 ~ 9) Ensemble

    pred_mean = np.mean(model_pred_list, axis=0)

    pred_class_indices = np.argmax(pred_mean, axis=1)
get_class_indices = train_generator.class_indices

labels = {val:key for key, val in get_class_indices.items()}

pred_result = [labels[idx] for idx in pred_class_indices]



submission = pd.read_csv(os.path.join(INPUT_PATH, 'sample_submission.csv'))

submission['class'] = pred_result

submission.to_csv('submission.csv', index=False)
submission.head()
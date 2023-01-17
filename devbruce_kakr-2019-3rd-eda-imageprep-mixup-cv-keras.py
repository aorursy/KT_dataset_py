!pip install -U efficientnet
import os

from efficientnet import EfficientNetB5





CV_N = 4  # 0 ~ 4 (n_splits=5)

PRETAINED_MODEL = EfficientNetB5

PRETAINED_MODEL_STR = 'EfficientNetB5'



# Required Size Of Pretained Model (299 x 299)

IMG_SIZE = 299



# Ensemble Mode

ENSEMBLE_MODE = True



# Uploaded Model Path

INPUT_PATH_ORIGIN = os.path.join('..', 'input')

UPLOADED_MODEL_PATH = os.path.join(

    os.path.join(INPUT_PATH_ORIGIN, 'efficientnetb5-kfold-n5'),

    'efficientnetb5_kfold_n5'

)

UPLOADED_MODEL_PATH = os.path.join(UPLOADED_MODEL_PATH, 'EfficientNetB5_kfold_n5')
import random

import warnings

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns

import cv2

import PIL

from PIL import ImageDraw

from keras import models, layers, optimizers

from keras import regularizers

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split, StratifiedKFold





# Ignore Warnings

warnings.filterwarnings('ignore')



# Full Data Set Mode

# If False, Lightweight Data Mode For Test (Get 20% Of Data)

FULL_DATA = True

if FULL_DATA: LIGHT_DATA_TRAIN, LIGHT_DATA_TEST = False, False

else: LIGHT_DATA_TRAIN, LIGHT_DATA_TEST = True, True



# Undersampling class 119 for class balance

UNDERSAMPLING_119 = True



# Constant Variables

LEARNING_RATE = 1e-4

OPTIMIZER = optimizers.Adam(lr=LEARNING_RATE)

EPOCHS = 50 if FULL_DATA else 4

BATCH_SIZE = 16  # if set high value, it would occur ResourceExhaustedError

INITIALIZER = 'he_normal'

REGULARIZER = regularizers.l2(1e-3)

DROPOUT_RATE = 0.5

PATIENCE = 8  # Patience Value For Early Stop

VERBOSE = 2 if FULL_DATA else 1  # Verbosity mode (2: No progress bar)

RANDOM_SEED = 2019

    

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



# Set Path of Preprocessed Train Images

TRAIN_IMG_PREP_PATH = os.path.join('..', 'train_prep')

if not os.path.exists(TRAIN_IMG_PREP_PATH):

    os.makedirs(TRAIN_IMG_PREP_PATH, exist_ok=True)



# Set Path of Preprocessed Test Images

TEST_IMG_PREP_PATH = os.path.join('..', 'test_prep')

if not os.path.exists(TEST_IMG_PREP_PATH):

    os.makedirs(TEST_IMG_PREP_PATH, exist_ok=True)



# Set Save Path of Model

MODEL_FILE_SAVE_DIR_PATH = '.'

if not os.path.exists(MODEL_FILE_SAVE_DIR_PATH):

    os.makedirs(MODEL_FILE_SAVE_PATH, exist_ok=True)

MODEL_FILE_PATH = os.path.join(

    MODEL_FILE_SAVE_DIR_PATH,

    str(CV_N) + '_' + PRETAINED_MODEL_STR + '.hdf5'

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
tmp_imgs = df_train['img_file'][:4]  # Load 4 images (0, 1, 2, 3)

plt.figure(figsize=(14,10))



for idx, img_file_name in enumerate(tmp_imgs, 1):

    img = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, img_file_name))

    plt.subplot(2, 2, idx)

    plt.title(img_file_name)

    plt.imshow(img)

    plt.axis('off')
def img_crop_or_draw_bbox(img_file_name, mode='crop', margin=6):

    '''

    mode = 'crop' or 'draw_bbox' (Default: 'crop')

    '''

    if img_file_name.split('_')[0] == 'train':

        IMG_PATH = TRAIN_IMG_PATH

        data = df_train

    elif img_file_name.split('_')[0] == 'test':

        IMG_PATH = TEST_IMG_PATH

        data = df_test

        

    if mode == 'crop':

        img = PIL.Image.open(os.path.join(IMG_PATH, img_file_name))

        pos = data.loc[data['img_file'] == img_file_name, ['bbox_x1','bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)

        width, height = img.size

        x1 = max(0, pos[0] - margin)

        y1 = max(0, pos[1] - margin)

        x2 = min(pos[2] + margin, width)

        y2 = min(pos[3] + margin, height)

        return img.crop((x1, y1, x2, y2))

    

    elif mode == 'draw_bbox':

        def draw_bbox(img_file_name, pos, outline_color='lightgreen', width=8):

            x1, y1 = pos[0], pos[1]  # Coordinate: Upper Left

            x2, y2 = pos[2], pos[3]  # Coordinate: Bottom Right

            rect_coordinates = (x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)

            img_file_name.line(rect_coordinates, fill=outline_color, width=width)

        

        img = PIL.Image.open(os.path.join(IMG_PATH, img_file_name))

        pos = data.loc[data['img_file'] == img_file_name, ['bbox_x1','bbox_y1', 'bbox_x2', 'bbox_y2']].values.reshape(-1)

        draw = ImageDraw.Draw(img)  # ImageDraw (from PIL)

        draw_bbox(draw, pos)

        return img





# Define Function For Test

def img_test_bbox_crop(img_file_name):

    # Show Original Image

    img_origin = PIL.Image.open(os.path.join(TRAIN_IMG_PATH, img_file_name))

    plt.figure(figsize=(22,14))

    plt.subplot(1, 3, 1)

    plt.title(f'Original Image - {img_file_name}')

    plt.imshow(img_origin)

    plt.axis('off')



    # Show Image Including bbox

    img_bbox = img_crop_or_draw_bbox(img_file_name, mode='draw_bbox')

    plt.subplot(1, 3, 2)

    plt.title(f'Boxing Image - {img_file_name}')

    plt.imshow(img_bbox)

    plt.axis('off')

    

    # Show Crop Image

    img_crop = img_crop_or_draw_bbox(img_file_name, mode="crop")

    plt.subplot(1, 3, 3)

    plt.title(f'Cropped Image - {img_file_name}')

    plt.imshow(img_crop)

    plt.axis('off')

    

    # Show Result

    plt.show()
# Test bbox and crop

img_test_bbox_crop(img_file_name=df_train['img_file'].iloc[100])
# Save Cropped Train Images (Path: /kaggle/working/train_crop)

if not os.listdir(TRAIN_IMG_CROP_PATH):  # If PATH_IMG_TRAIN_CROP is empty

    for idx, row in df_train.iterrows():

        img_file_name = row['img_file']

        img_crop = img_crop_or_draw_bbox(img_file_name, mode='crop')

        img_crop.save(os.path.join(TRAIN_IMG_CROP_PATH, img_file_name))



# Save Cropped Test Images (Path: /kaggle/working/test_crop)

if not os.listdir(TEST_IMG_CROP_PATH):

    for idx, row in df_test.iterrows():

        img_file_name = row['img_file']

        img_crop = img_crop_or_draw_bbox(img_file_name, mode='crop')

        img_crop.save(os.path.join(TEST_IMG_CROP_PATH, img_file_name))
# Histogram Equalization And Add Padding After Cropping

def img_he_pad(img_file_name, add_padding=True):

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

    

    # -------- Add Padding --------- #

    if add_padding:

        img_prep_h, img_prep_w = img_prep.shape[0], img_prep.shape[1]  # (height, width)

        ratio = float(IMG_SIZE) / max(img_prep_h, img_prep_w)

        shape_no_padding = (int(img_prep_h * ratio), int(img_prep_w * ratio))



        img_prep_no_padding = cv2.resize(img_prep, shape_no_padding[::-1])

        

        size_h = IMG_SIZE - shape_no_padding[0]

        size_w = IMG_SIZE - shape_no_padding[1]

        

        top, bottom = size_h // 2, size_h - (size_h // 2)

        left, right = size_w // 2, size_w - (size_w // 2)



        PADDING_COLOR = (0, 0, 0)  # black

        img_prep = cv2.copyMakeBorder(

            img_prep_no_padding,

            top,

            bottom,

            left,

            right,

            cv2.BORDER_CONSTANT,

            value=PADDING_COLOR

        )

    

    return img_prep
# Save Preprocessed Train Images (Path: ../train_prep)

if not os.listdir(TRAIN_IMG_PREP_PATH):  # If PATH_IMG_TRAIN_PREP is empty

    for idx, row in df_train.iterrows():

        img_file_name = row['img_file']

        img_prep = img_he_pad(img_file_name, add_padding=True)

        cv2.imwrite(os.path.join(TRAIN_IMG_PREP_PATH, img_file_name), img_prep)



# Save Preprocessed Test Images (Path: ../test_prep)

if not os.listdir(TEST_IMG_PREP_PATH):

    for idx, row in df_test.iterrows():

        img_file_name = row['img_file']

        img_prep = img_he_pad(img_file_name, add_padding=True)

        cv2.imwrite(os.path.join(TEST_IMG_PREP_PATH, img_file_name), img_prep)
# Define Function For Test

def test_he_padding(img_file_name):

    # Show Cropped Image

    img_crop = PIL.Image.open(os.path.join(TRAIN_IMG_CROP_PATH, img_file_name))

    plt.figure(figsize=(12, 9))

    plt.subplot(1, 2, 1)

    plt.title(f'Cropped Image - {img_file_name}')

    plt.imshow(img_crop)

    plt.axis('off')



    # Show Preprocessed Image

    img_he_pad = PIL.Image.open(os.path.join(TRAIN_IMG_PREP_PATH, img_file_name))

    plt.subplot(1, 2, 2)

    plt.title(f'Historgram Equalized Cropped Image(Add Padding) - {img_file_name}')

    plt.imshow(img_he_pad)

    plt.axis('off')

    

    # Show Result

    plt.show()

    

# Test Histogram Equalization & Add Padding

test_he_padding(img_file_name=df_train['img_file'].iloc[114])
# class_mode="categorical", y_col="class" column values must be type string (flow method options)

df_train['class'] = df_train['class'].astype('str')



# Take Necessary Columns

df_train = df_train[['img_file', 'class']]

df_test = df_test[['img_file']]
# Define Get Model Function

def get_model(pretained_model, img_size, optimizer, class_nb):

    model_options = dict(

        include_top=False,

        weights='imagenet',

        input_shape=(img_size, img_size, 3),

    )

    model = models.Sequential()

    model.add(pretained_model(**model_options))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(

        1024,

        activation='relu',

        kernel_initializer=INITIALIZER,

        kernel_regularizer=REGULARIZER,

    ))

    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Dense(class_nb, activation='softmax'))

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['acc'])

    return model





# Define Get Step Function

def get_steps(num_data, batch_size):

    quotient, remainder = divmod(num_data, batch_size)

    return (quotient + 1) if remainder else quotient
# Define Get Callback Function

def get_callback_list(model_file_path, patience, verbose):

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

    return [early_stop, reduce_lr, model_chk]
idx_split = list()

skf = StratifiedKFold(n_splits=5, random_state=RANDOM_SEED)

for train_idx, val_idx in skf.split(X=df_train['img_file'], y=df_train['class']):

    idx_split.append((train_idx, val_idx))

    

X_train = df_train.iloc[idx_split[CV_N][0]].reset_index()

X_val = df_train.iloc[idx_split[CV_N][1]].reset_index()
class MixupImageDataGenerator():

    def __init__(self, generator, dataframe, directory, img_size, alpha, subset=None):

        self.batch_index = 0

        self.batch_size = BATCH_SIZE

        self.alpha = alpha



        # First iterator yielding tuples of (x, y)

        self.generator1 = generator.flow_from_dataframe(

            dataframe=dataframe,

            directory=directory,

            target_size=(img_size, img_size),

            x_col='img_file',

            y_col='class',

            color_mode='rgb',

            class_mode='categorical',

            batch_size=BATCH_SIZE,

        )



        # Second iterator yielding tuples of (x, y)

        self.generator2 = generator.flow_from_dataframe(

            dataframe=dataframe,

            directory=directory,

            target_size=(img_size, img_size),

            x_col='img_file',

            y_col='class',

            color_mode='rgb',

            class_mode='categorical',

            batch_size=BATCH_SIZE,

        )

        

        self.n = self.generator1.samples

    

    @property

    def class_indices(self):

        return self.generator1.class_indices

    

    @property

    def samples(self):

        return self.generator1.samples

    

    def reset_index(self):

        self.generator1._set_index_array()

        self.generator2._set_index_array()



    def on_epoch_end(self):

        self.reset_index()



    def reset(self):

        self.batch_index = 0



    def __len__(self):

        return (self.n + self.batch_size - 1) // self.batch_size



    def get_steps_per_epoch(self):

        quotient, remainder = divmod(self.n, self.batch_size)

        return (quotient + 1) if remainder else quotient



    def __next__(self):

        if self.batch_index == 0:

            self.reset_index()



        current_index = (self.batch_index * self.batch_size) % self.n

        if self.n > current_index + self.batch_size:

            self.batch_index += 1

        else:

            self.batch_index = 0



        reshape_size = self.batch_size

        if current_index == (self.get_steps_per_epoch()-1) * self.batch_size:

            reshape_size = self.n - ((self.get_steps_per_epoch()-1) * self.batch_size)

            

        # random sample the lambda value from beta distribution.

        l = np.random.beta(a=self.alpha, b=self.alpha, size=reshape_size)

            

        X_l = l.reshape(reshape_size, 1, 1, 1)

        y_l = l.reshape(reshape_size, 1)

        

        # Get a pair of inputs and outputs from two iterators.

        X1, y1 = self.generator1.next()

        X2, y2 = self.generator2.next()



        # Perform the mixup.

        X = X1 * X_l + X2 * (1 - X_l)

        y = y1 * y_l + y2 * (1 - y_l)

        return X, y



    def __iter__(self):

        while True:

            yield next(self)
# Train Generator (Mixup)

train_datagen = ImageDataGenerator(

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    vertical_flip=False,

    rescale=1./255,

)

train_generator = MixupImageDataGenerator(

    generator=train_datagen,

    dataframe=X_train,

    directory=TRAIN_IMG_PREP_PATH,

    img_size=IMG_SIZE,

    alpha=0.2,

)



# Validation Generator

valid_datagen = ImageDataGenerator(rescale=1./255,)

valid_generator = valid_datagen.flow_from_dataframe(

    dataframe=X_val,

    directory=TRAIN_IMG_PREP_PATH,

    target_size=(IMG_SIZE, IMG_SIZE),

    x_col='img_file',

    y_col='class',

    color_mode='rgb',

    class_mode='categorical',

    batch_size=BATCH_SIZE,

)



# Test Generator

test_datagen = ImageDataGenerator(rescale=1./255,)

test_generator = test_datagen.flow_from_dataframe(

    dataframe=df_test,

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

        img_size=IMG_SIZE,

        pretained_model=PRETAINED_MODEL,

        optimizer=OPTIMIZER,

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

            verbose=VERBOSE

        ),

        validation_data=valid_generator,

        validation_steps=get_steps(valid_generator.samples, BATCH_SIZE),

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

    plt.title(f'Loss Per Epoch ({CV_N}.{PRETAINED_MODEL_STR})')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()



    fig.add_subplot(1, 2, 2)

    plt.plot(epochs, acc, 'b', label='Training Accuracy')

    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')

    plt.title(f'Accuracy Per Epoch ({CV_N}.{PRETAINED_MODEL_STR})')

    plt.xlabel('Epochs')

    plt.ylabel('Loss')

    plt.legend()

    plt.show()

    

    # Prediction

    pred = model.predict_generator(

        generator=test_generator,

        steps=get_steps(len(df_test), BATCH_SIZE),

        verbose=VERBOSE,

    )

    

    # Get Pred Indices

    pred_class_indices = np.argmax(pred, axis=1)
if ENSEMBLE_MODE:

    UPLOADED_MODEL_LIST = sorted(os.listdir(UPLOADED_MODEL_PATH))



    model_pred_list = list()

    for model_n, model_name in enumerate(UPLOADED_MODEL_LIST):

        model_file_path = os.path.join(UPLOADED_MODEL_PATH, model_name)

        model = get_model(

            img_size=IMG_SIZE,

            pretained_model=PRETAINED_MODEL,

            optimizer=OPTIMIZER,

            class_nb=CLASS_NB,

        )

        model.load_weights(model_file_path)

        test_generator.reset()

        pred = model.predict_generator(

            generator=test_generator,

            steps=get_steps(len(df_test), BATCH_SIZE),

            verbose=VERBOSE,

        )

        model_pred_list.append(pred)



    # Model Ensemble

    pred_mean = np.mean(model_pred_list, axis=0)

    

    # Get Pred Indices

    pred_class_indices = np.argmax(pred_mean, axis=1)
get_class_indices = train_generator.class_indices

labels = {val:key for key, val in get_class_indices.items()}

pred_result = [labels[idx] for idx in pred_class_indices]



submission = pd.read_csv(os.path.join(INPUT_PATH, 'sample_submission.csv'))

submission['class'] = pred_result

submission.to_csv('submission.csv', index=False)
submission.head()
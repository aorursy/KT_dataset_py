import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time
import os

import tensorflow as tf

from PIL import Image

# from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm

import tensorflow as tf
import tensorflow.keras.backend as be
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.metrics import AUC
from tensorflow.keras.losses import BinaryCrossentropy
# from tensorflow_addons.losses import *
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

import memory_profiler
import gc
from pprint import pprint
from imblearn.over_sampling import *
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)
img_path = '/kaggle/input/jpeg-melanoma-256x256/'
base_path = '/kaggle/input/siim-isic-melanoma-classification/'

df_train = pd.read_csv(os.path.join(base_path, 'train.csv'))
df_test = pd.read_csv(os.path.join(base_path, 'test.csv'))
df_submission = pd.read_csv(os.path.join(base_path, 'sample_submission.csv'))

def timg(name, test=False):
    if test:
        return img_path + 'test/' + name + '.jpg'
    else:
        return img_path + 'train/' + name + '.jpg'
def show_grid(df, cols=9, rows=4):
    if df.shape[0] == 0:
        return
    plt.figure(figsize=(18,9))
    for i in range(min(df.shape[0], cols * rows)):
        plt.subplot(rows, cols, i+1, xticks=[], yticks=[])
        idx = np.random.randint(0, df.shape[0], 1)[0]
        im = Image.open(timg(df.iloc[idx]['image_name']))
        plt.imshow(im)
        plt.xlabel(df.iloc[idx]['benign_malignant'])
        plt.ylabel(df.iloc[idx]['anatom_site_general_challenge'])
    plt.show()

# Check young people
print('Young folks')
show_grid(df_train[(df_train['age_approx'] < 40.0) & (df_train['target'] == 1)])
# Check diagnosis
print('Target 1 with Melanoma')
show_grid(df_train[(df_train['diagnosis'] == 'melanoma') & (df_train['target'] == 1)])
# Check a single patient
# print("Patient IP_0962375")
# pat = 'IP_0962375'
# show_grid(df_train[(df_train['patient_id'] == pat) & (df_train['target'] == 1)])
# show_grid(df_train[(df_train['patient_id'] == pat) & (df_train['target'] == 0)])
# Mark male female as 1/0
# There are only two values, there are some missing values, which should be filled with mode
df_train['sex'] = df_train['sex'].replace({ 'female': 0, 'male': 1 })
df_test['sex'] = df_test['sex'].replace({ 'female': 0, 'male': 1 })
df_train['sex'].fillna(df_train['sex'].mode()[0], inplace=True)

# Remove benign malignant, it's the same as target
df_train.drop(['benign_malignant'], inplace=True, axis=1)

# Add dummies for anatom_site_general_challenge
# Fill the nan's with a new dummy
def add_dummies(dataset, column, short_name):
    dummy = pd.get_dummies(
        dataset[column], 
        drop_first=True, 
        prefix=short_name, 
        prefix_sep='_',
        dummy_na=True
    )
    merged = pd.concat([dataset, dummy], axis=1)
    return merged.drop([column], axis=1)

df_train = add_dummies(df_train, 'anatom_site_general_challenge', 'anatom')
df_test = add_dummies(df_test, 'anatom_site_general_challenge', 'anatom')

# Age has some missing values, fill with median
df_train['age_approx'].fillna(df_train['age_approx'].median(), inplace=True)
df_test['age_approx'].fillna(df_test['age_approx'].median(), inplace=True)

# %% [code]
# Check how many times are their images taken
df_train['image_count'] = df_train['patient_id'].map(df_train.groupby(['patient_id'])['image_name'].count())
df_test['image_count'] = df_test['patient_id'].map(df_test.groupby(['patient_id'])['image_name'].count())

# Diagnosis is only in train, removing it
df_train.drop(['diagnosis', 'patient_id'], inplace=True, axis=1)
df_test.drop(['patient_id'], inplace=True, axis=1)
df_train['image_name'] = df_train['image_name'].apply(lambda x: timg(x))
df_test['image_name'] = df_test['image_name'].apply(lambda x: timg(x, test=True))
sc = StandardScaler()

def scale(df, cols_to_remove, fit=False):
    removed = df[cols_to_remove]
    df = df.drop(cols_to_remove, axis=1)
    cols = df.columns
    if fit:
        df = sc.fit_transform(df)
    else:
        df = sc.transform(df)
    df = pd.DataFrame(df, columns=cols)
    df[cols_to_remove] = removed
    return df

df_train = scale(df_train, fit=True, cols_to_remove=['image_name', 'target'])
df_test = scale(df_test, cols_to_remove=['image_name'])
len(df_train)
!pip install imbalanced-learn
# Over sampling / Under Sampling

X = df_train.drop(['target'], axis=1)
y = df_train['target']

X_resampled, y_resampled = RandomOverSampler(random_state=0).fit_resample(X, y)
X_resampled['target'] = y_resampled

print('Target 1 percentage:',
    (
        X_resampled[X_resampled['target'] == 1].shape[0]/
        X_resampled.shape[0]
    )*100,
    '%',
    'Total:',
    X_resampled.shape[0]
)

df_train = X_resampled
FOLDS = 3

X_train = [None] * FOLDS
X_val = [None] * FOLDS

kfold = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
split = kfold.split(
    np.arange(len(df_train)),
    df_train['target'],
)
for fold, (idx_train, idx_val) in enumerate(split):
    X_train[fold] = df_train.iloc[idx_train]
    X_val[fold] = df_train.iloc[idx_val]
    
X_test = df_test
print('Target 1 percentage:',
    (
        df_train[df_train['target'] == 1].shape[0]/
        df_train.shape[0]
    )*100,
    '%'
)

print(X_train[0][X_train[0]['target'] == 1].shape[0])
print(X_train[1][X_train[1]['target'] == 1].shape[0])
print(X_train[2][X_train[2]['target'] == 1].shape[0])

print(X_val[0][X_val[0]['target'] == 1].shape[0])
print(X_val[1][X_val[1]['target'] == 1].shape[0])
print(X_val[2][X_val[2]['target'] == 1].shape[0])
class Dataset():
    
    def __init__(self, df, training=True, augment=False, batch_size=16, meta=False):
        self.auto = tf.data.experimental.AUTOTUNE
        self.training = training
        self.augment = augment
        
        if meta:
            if training:
                self.ds = tf.data.Dataset.from_tensor_slices((
                    df.drop(['target', 'image_name'], axis=1).values,
                    df['image_name'].values,
                    df['target'].values
                ))
            else:
                self.ds = tf.data.Dataset.from_tensor_slices((
                    df.drop(['image_name'], axis=1).values,
                    df['image_name'].values,
                    [-1] * df.shape[0],
                ))
            self.ds = self.ds.map(
                lambda meta, img, target: (meta, self.load_image(img), target), 
                num_parallel_calls=self.auto
            )

        else:
            if training:
                self.ds = tf.data.Dataset.from_tensor_slices((
                    df['image_name'].values,
                    df['target'].values
                ))
            else:
                self.ds = tf.data.Dataset.from_tensor_slices((
                    df['image_name'].values,
                    [-1] * df.shape[0],
                ))
            self.ds = self.ds.map(
                lambda img, target: (self.load_image(img), target), 
                num_parallel_calls=self.auto
            )

#         self.ds = self.ds.cache()
        self.ds = self.ds.batch(batch_size, drop_remainder=True)
        self.ds = self.ds.prefetch(self.auto)

    def data(self):
        return self.ds
    
    def load_image(self, img):
        size = 256

        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32) / 255.0
        
        if self.augment:
#             img = tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, fill_mode='reflect')(img)
#             img = tf.keras.layers.experimental.preprocessing.RandomZoom(0.2)(img)
            
            img = tf.image.random_brightness(img, 0.2)
            img = tf.image.random_contrast(img, 0.8, 1.2)
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_hue(img, 0.2)
            img = tf.image.random_saturation(img, 0.8, 1.2)
            img = self.dropout(img)
    
        img = tf.reshape(img, [size, size, 3])
        return img

    def dropout(self, image, DIM=256, PROBABILITY = 0.5, CT = 2, SZ = 0.2):
        # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
        # output - image with CT squares of side size SZ*DIM removed

        # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
        P = tf.cast( tf.random.uniform([],0,1)<PROBABILITY, tf.int32)
        if (P==0)|(CT==0)|(SZ==0): return image

        for k in range(CT):
            # CHOOSE RANDOM LOCATION
            x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
            y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
            # COMPUTE SQUARE 
            WIDTH = tf.cast( SZ*DIM,tf.int32) * P
            ya = tf.math.maximum(0,y-WIDTH//2)
            yb = tf.math.minimum(DIM,y+WIDTH//2)
            xa = tf.math.maximum(0,x-WIDTH//2)
            xb = tf.math.minimum(DIM,x+WIDTH//2)
            # DROPOUT IMAGE
            one = image[ya:yb,0:xa,:]
            two = tf.zeros([yb-ya,xb-xa,3]) 
            three = image[ya:yb,xb:DIM,:]
            middle = tf.concat([one,two,three],axis=1)
            image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM,:,:]],axis=0)

        # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR 
        image = tf.reshape(image,[DIM,DIM,3])
        return image
train_ds = Dataset(df_train, augment=True).data()
print( '[Train] Total:', train_ds.cardinality().numpy() )

test_ds = Dataset(df_test, training=False).data()
print( '[Test] Total:', test_ds.cardinality().numpy() )
def show_grid_dataset():
    sample = train_ds.take(100)
    count = 0
    plt.figure(figsize=(18,9))
    for idx, data in enumerate(iter(sample)):
        imgs, targets = data
        if count == 32:
            break
        for i, img in enumerate(imgs):
            plt.subplot(4, 8, count+1, xticks=[], yticks=[])
            img = np.clip(img.numpy() * 255, 0, 255).astype(np.uint8)
            img = Image.fromarray(img)
            plt.xlabel(str(targets[i].numpy()))
            plt.imshow(img)
            count += 1
    plt.show()

if __name__ == '__main__':
    m1 = memory_profiler.memory_usage()
    t1 = time.perf_counter()
    cubes = show_grid_dataset()
    t2 = time.perf_counter()
    m2 = memory_profiler.memory_usage()
    time_diff = t2 - t1
    mem_diff = m2[0] - m1[0]
    print(f"It took {time_diff} Secs and {mem_diff} Mb to execute this method")
from tensorflow.keras.applications import EfficientNetB4
tf.keras.backend.clear_session()
gc.collect()


def get_model():
    efn0_base = EfficientNetB4(include_top=False, weights='imagenet', input_shape=(256, 256, 3))
#     efn0_base.trainable = False
    
    model = Sequential()
    model.add(efn0_base)
    model.add(GlobalAveragePooling2D())
#     model.add(Dense(32, activation='relu'))
#     model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))
    model.summary()
    
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss=BinaryCrossentropy(label_smoothing=0.05), # 
        metrics=['binary_crossentropy', AUC(name='auc')]
    )
    return model

model = get_model()
EPOCHS=3
BATCH_SIZE=32

scores = list()

def train():
    for i in range(1): #FOLDS
        fX_train = X_train[i]
        fX_val = X_val[i]

        train_ds = Dataset(X_train[i], augment=True, batch_size=BATCH_SIZE)
        val_ds = Dataset(X_val[i], batch_size=BATCH_SIZE)
        
        history = model.fit(
            train_ds.data(),
            validation_data=val_ds.data(),
            verbose=1,
            epochs=EPOCHS,
            # steps_per_epoch=math.floor(train_ds.data().cardinality()/BATCH_SIZE),
            batch_size=BATCH_SIZE,
            callbacks=[
                # EarlyStopping(monitor='auc', mode='max', patience=6, verbose=2, restore_best_weights=True),
                ModelCheckpoint(
                    monitor='val_auc', verbose=1, save_best_only=True, mode='max', filepath='{val_auc:.5f}.h5'),
                ReduceLROnPlateau(monitor='val_auc', factor=0.1, patience=3, verbose=1, mode='max', cooldown=1, min_lr=0 ),
            ],
        )
        pprint(history.history)
        
        predicted = model.predict(train_ds.data(), verbose=1)
        y_pred = np.reshape(np.round(predicted), (1, predicted.shape[0]))[0]
        y_true = X_train[i]['target'].iloc[:train_ds.data().cardinality().numpy()*BATCH_SIZE].values
        print( confusion_matrix(y_true, y_pred) )
        
train()
# Get score on the whole data set
train_ds = Dataset(df_train, training=False, batch_size=BATCH_SIZE)
predicted = model.predict(train_ds.data(), verbose=1)

y_pred = np.reshape(np.round(predicted), (1, predicted.shape[0]))[0]
y_true = df_train['target'].iloc[:33120].values

confusion_matrix(y_true, y_pred)
# Get scores for the test set and submit
sub_model = tf.keras.models.load_model('./0.91169_efn_pool_batch_256BD10x1.h5') # 0.9074

# Choose sub_model or model

test_ds = Dataset(X_test, training=False, batch_size=BATCH_SIZE)
predicted = model.predict(test_ds.data(), verbose=1)

sub = pd.DataFrame(dict(
    image_name = np.array([img_name.numpy().decode('utf-8') for img, img_name in iter(ds.stream().unbatch()) ]),
    target = np.reshape(predicted, (1, predicted.shape[0]))[0]
))

sub.to_csv('submission.csv', index=False)
!pip install efficientnet
import efficientnet.tfkeras as efn


import gc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import albumentations


from random import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
%matplotlib inline

print(tf.__version__)
print(tf.keras.__version__)
from sklearn import model_selection
import numpy as np
AUTO = tf.data.experimental.AUTOTUNE
# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)
GCS_DS_PATH = KaggleDatasets().get_gcs_path('siic-isic-224x224-images')
'gs://kds-e1fe477ff4c67fa3f8291e0d9e4c96250efeec91c8d836b1e932f0cd'
GCS_DS_PATH
# create folds
df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")
df["kfold"] = -1    
df = df.sample(frac=1).reset_index(drop=True)
y = df.target.values
kf = model_selection.StratifiedKFold(n_splits=5)

for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
    df.loc[v_, 'kfold'] = f

df.to_csv("train_folds.csv", index=False)
def get_emodel():
     with strategy.scope():
        en =efn.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
        en.trainable = True

        model = tf.keras.Sequential([
            en,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        opt = Adam(learning_rate=1e-4)
        model.compile(optimizer = opt,
            loss = 'binary_crossentropy',
            metrics=['accuracy']
        )
        return model
 
BATCH_SIZE= 64 * strategy.num_replicas_in_sync
img_size = 224
EPOCHS = 10



def data_augment(image, label):    
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.random_saturation(image, 0, 2)
    return image, label   
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_png(bits, channels=3)
    image = tf.cast(image, tf.float32) /255
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label
def cv2_func(image, label):
    print(label)
    image = np.array(image,dtype= np.int64)    
    image = train_aug(image=image)['image']
    image = tf.cast(image, tf.float32)
    return image

def tf_cv2_func(image, label):
    print(type(label))
    [image] = tf.py_function(cv2_func, [image,label], [tf.float32])
    print(image.shape)
    return image, label



# train_dataset = (
# tf.data.Dataset
# .from_tensor_slices((train_images, train_targets))
# .map(decode_image, num_parallel_calls=AUTO) 
# .map(tf_cv2_func, num_parallel_calls=AUTO) 
# .repeat()
# .cache()
# .shuffle(512)
# .batch(BATCH_SIZE)
# .prefetch(AUTO)
# )
# training_data_path = f"{GCS_DS_PATH}/train/"
# df = pd.read_csv("/kaggle/working/train_folds.csv")

# epochs = 5
# train_bs = 32* strategy.num_replicas_in_sync
# valid_bs = 32
# fold = 0

# df_train = df[df.kfold != fold].reset_index(drop=True)


# mean = (0.485, 0.456, 0.406)
# std = (0.229, 0.224, 0.225)
# train_aug = albumentations.Compose(
#     [
# #         albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
#         albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
#         albumentations.Flip(p=0.5)
#     ]
# )



# train_images = df_train.image_name.values.tolist()
# train_images = [os.path.join(training_data_path, i + ".png") for i in train_images]
# train_targets = df_train.target.values

# train_dataset = (
# tf.data.Dataset
# .from_tensor_slices((train_images, train_targets))
# .map(decode_image, num_parallel_calls=AUTO)    
# .map(tf_cv2_func , num_parallel_calls =AUTO) 
# .repeat()
# .cache()
# .shuffle(512)
# .batch(BATCH_SIZE)
# .prefetch(AUTO)
# )

# test_batch = iter(train_dataset)
# image , label = next(test_batch)

def train():
    training_data_path = f"{GCS_DS_PATH}/train/"
    df = pd.read_csv("/kaggle/working/train_folds.csv")
    
    epochs = 5
    train_bs = 32* strategy.num_replicas_in_sync
    valid_bs = 32
    fold = 0

    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    model = get_emodel()

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    train_aug = albumentations.Compose(
        [
#             albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True),
            albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
            albumentations.Flip(p=0.5)
        ]
    )

    valid_aug = albumentations.Compose(
        [
#             albumentations.Normalize(mean, std, max_pixel_value=255.0, always_apply=True)
        ]
    )

    train_images = df_train.image_name.values.tolist()
    train_images = [os.path.join(training_data_path, i + ".png") for i in train_images]
    train_targets = df_train.target.values

    valid_images = df_valid.image_name.values.tolist()
    valid_images = [os.path.join(training_data_path, i + ".png") for i in valid_images]
    valid_targets = df_valid.target.values

    
    train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_images, train_targets))
    .map(decode_image, num_parallel_calls=AUTO)    
    .map(data_augment, num_parallel_calls=AUTO) 
#     .map(tf_cv2_func, num_parallel_calls=AUTO) 
    .repeat()
    .cache()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    )

    valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_images, valid_targets))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
    )

    Checkpoint=tf.keras.callbacks.ModelCheckpoint(f"Enet_model.h5", monitor='val_accuracy', verbose=1, save_best_only=True,
       save_weights_only=True,mode='max')
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', patience=3, verbose=0, mode='auto'
    )
    train_history1 = model.fit(
            train_dataset, 
            validation_data = valid_dataset, 
            steps_per_epoch=train_targets.shape[0] // BATCH_SIZE,            
            validation_steps=valid_targets.shape[0] // BATCH_SIZE,            
            callbacks=[lr_callback, Checkpoint],
            epochs=EPOCHS,
            verbose=1
    )
train()

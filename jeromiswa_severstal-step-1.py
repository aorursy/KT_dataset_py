import os

import cv2
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd  

import keras
import tensorflow as tf
from keras import backend as K
from keras import layers
from keras.applications import DenseNet121
from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.optimizers import Adam, Nadam
from keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
train_folder = os.path.join('/kaggle/input','/severstal-steel-defect-detection/train_images')
test_folder = os.path.join('/kaggle/input','/severstal-steel-defect-detection/test_images')

train_images_set = set()
test_images_set = set()

count=0
for dirname, _, filenames in os.walk('/kaggle/input/severstal-steel-defect-detection'):
    count=0
    for filename in filenames:
        count+=1
        if dirname == '/kaggle/input/severstal-steel-defect-detection/train_images':
            train_images_set.add(filename)
        if dirname == '/kaggle/input/severstal-steel-defect-detection/test_images':
            test_images_set.add(filename)

    if dirname == '/kaggle/input/severstal-steel-defect-detection/train_images':
        no_train_images = count
        print(dirname)
        print("no_train_images >", count)
    if dirname == '/kaggle/input/severstal-steel-defect-detection/test_images':
        no_test_images = count
        print(dirname)
        print("no_test_images >", count)
        
    print(dirname)
train_df = pd.read_csv('../input/severstal-steel-defect-detection/train.csv').sort_values(by=['ImageId'])
submission_df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')

print("train_df Shape (Summation of 'no_defect_class_for_image_i' from image i to j):", train_df.shape, "\nsubmission_df Shape:", submission_df.shape)
display(train_df.head()) # Summation of "no_defect_class_for_image_i" from image i to j
display(submission_df.head())
train_images_have_defects_set = set(train_df['ImageId'].unique())
train_images_no_defects_set = train_images_set - train_images_have_defects_set
no_unique_train_images_no_defects = len(train_images_no_defects_set)
test_images_set = set(submission_df['ImageId'].unique())

print("Number of images in train_df that have defects >>>", len(train_images_have_defects_set), "\nsubmission_df unique >>>", len(test_images_set))
print("\nTotal no of train images >>>", no_train_images)
print("Therefore, Number of images that have no defects >>>", str(no_unique_train_images_no_defects))
train_images_have_defects_set
train_images_no_defects_df = pd.DataFrame({"ImageId" : list(train_images_no_defects_set)})
train_images_no_defects_df['allMissing'] = 1
train_images_have_defects_df = pd.DataFrame({"ImageId" : list(train_images_have_defects_set)})
train_images_have_defects_df['allMissing'] = 0
frames = [train_images_no_defects_df, train_images_have_defects_df]
train_nan_df = pd.concat(frames, ignore_index=True)

print(train_nan_df.shape)
display(train_nan_df)
sub_df = pd.read_csv('../input/severstal-steel-defect-detection/sample_submission.csv')
test_nan_df = pd.DataFrame(sub_df['ImageId'].unique(), columns=['ImageId'])
display(test_nan_df.head())
test_nan_df.shape
def load_img(code, base, resize=True):
    path = f'{base}/{code}'
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize:
        img = cv2.resize(img, (256, 256))
    return img

def validate_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
train_path = '../tmp/train'
validate_path(train_path)

for code in tqdm(train_nan_df['ImageId']):
    img = load_img(
        code,
        base='/kaggle/input/severstal-steel-defect-detection/train_images'
    )
    path = code.replace('.jpg', '')
    cv2.imwrite(f'{train_path}/{path}.png', img)
    
train_nan_df['ImageId'] = train_nan_df['ImageId'].apply(
    lambda x: x.replace('.jpg', '.png')
)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.15

def create_datagen():
    return ImageDataGenerator(
        zoom_range=0.1,  # set range for random zoom
        # set mode for filling points outside the input boundaries
        fill_mode='constant',
        cval=0.,
        rotation_range=10,
        height_shift_range=0.1,
        width_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1/255.,
        validation_split=VALIDATION_SPLIT
    )

def create_test_gen():
    return ImageDataGenerator(rescale=1/255.).flow_from_dataframe(
        test_nan_df,
        directory='../input/severstal-steel-defect-detection/test_images/',
        x_col='ImageId',
        class_mode=None,
        target_size=(256, 256),
        batch_size=BATCH_SIZE,
        shuffle=False
    )

def create_flow(datagen, subset):
    return datagen.flow_from_dataframe(
        train_nan_df, 
        directory=train_path,
        x_col='ImageId', 
        y_col='allMissing', 
        class_mode='raw',
        target_size=(256, 256),
        batch_size=BATCH_SIZE,
        subset=subset
    )

# Using original generator
data_generator = create_datagen()
train_gen = create_flow(data_generator, 'training')
val_gen = create_flow(data_generator, 'validation')
test_gen = create_test_gen()
def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def build_model():
    densenet = DenseNet121(
        include_top=False,
        input_shape=(256,256,3),
        weights='/kaggle/input/keras-pretrain-model-weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
    )
    
    model = Sequential()
    model.add(densenet)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(
        loss='binary_crossentropy',
        optimizer=Nadam(),
        metrics=['accuracy', f1_m, precision_m, recall_m]
    )
    
    return model
model = build_model()
model.summary()
total_steps = train_nan_df.shape[0] / BATCH_SIZE

checkpoint = ModelCheckpoint(
    '../tmp/model.h5', 
    monitor='val_accuracy', 
    verbose=1, 
    save_best_only=True, 
    save_weights_only=False,
    mode='auto'
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=5,
    verbose=1,
    min_lr=0.5e-6
)

history = model.fit_generator(
    train_gen,
    steps_per_epoch=total_steps * 0.85,
    validation_data=val_gen,
    validation_steps=total_steps * 0.15,
    epochs=60,
    callbacks=[checkpoint, reduce_lr]
)
history_df = pd.DataFrame(history.history)
history_df.to_csv('history.csv', index=False)
display(history_df)
history_df.shape
model.save('../tmp/dense121_model.h5')
model.load_weights('../tmp/dense121_model.h5')
y_test = model.predict_generator(
    test_gen,
    steps=len(test_gen),
    verbose=1
)
history_df[['lr']].plot()
plt.title('Model Learning Rate')
plt.ylabel('Learning Rate')
plt.xlabel('Epoch')
plt.show()
history_df[['loss', 'val_loss']].plot()
plt.title('Model loss')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
test_nan_df['allMissing'] = y_test

train_nan_df.to_csv('train_missing_count.csv', index=False)
test_nan_df.to_csv('test_missing_count.csv', index=False)
history_df[['accuracy', 'val_accuracy']].plot()
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
history_df[['f1_m', 'val_f1_m']].plot()
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
history_df[['precision_m', 'val_precision_m']].plot()
plt.title('Model precision')
plt.ylabel('Precision')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
history_df[['recall_m', 'val_recall_m']].plot()
plt.title('Model recall')
plt.ylabel('Recall')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
# All Images will be Rescaled by 1./255. We Apply Data Augmentation Here.
train_datagen = ImageDataGenerator(rotation_range=40,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1./255,
                                   shear_range=0.1,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)
bs = 16 
img_size = (256, 512)

train_gen = train_datagen.flow_from_directory(
    directory=train_folder,
    target_size=img_size,
    batch_size=bs,
    class_mode='binary'
)

test_gen = test_datagen.flow_from_directory(
    directory=val_folder,
    target_size=img_size,
    batch_size=bs,
    class_mode='binary'
)
from keras.applications import DenseNet121
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Dropout, BatchNormalization

def buildModel1():
  dense_net = DenseNet121(
      include_top=False,
      input_shape=(256, 512, 3), # (width, height, colorchannel)
      weights='/kaggle/input/keras-pretrain-model-weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5'
  )

  model = Sequential()
  model.add(dense_net)
  model.add(GlobalAveragePooling2D())
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(512, activation='relu'))
  model.add(BatchNormalization())
  model.add(Dropout(0.5))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(
      loss='binary_crossentropy',
      optimizer='adam',
      metrics=['accuracy', f1_m, precision_m, recall_m]
  )

  return model
history1 = buildModel1().fit_generator(
          train_gen, # train generator has 12568 train images but we are not using all of them
          steps_per_epoch=786, # training 12568 images = 786 steps x 16 images per batch
          epochs=25,
          validation_data=test_gen, # validation generator has 5,000 validation images
          validation_steps=158 # validating on 2514 images = 158 steps x 16 images per batch
)
import matplotlib.pyplot as plt

# Plot training & validation accuracy values
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

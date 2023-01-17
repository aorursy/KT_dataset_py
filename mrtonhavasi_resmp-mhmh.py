import numpy as np
import pandas as pd
import cv2
import keras
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import sys
import csv
import time
from scipy import stats
import shutil
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#number of images and landmarks
train = pd.read_csv("../input/landmark-recognition-2020/train.csv")
print('Train:\t\t', train.shape)
print('Landmarks:\t', len(train['landmark_id'].unique()))
train.head()
#number of images in each of the classes
lm = train[['landmark_id', 'id']].groupby('landmark_id').count().reset_index()
lm = lm.sort_values('id', ascending=False)
lm = lm.rename(columns={'id': 'count'}).reset_index(drop=True)
print(lm)
#Landmark ID distribution
plt.figure(figsize = (10, 8))
plt.title('Landmark Image Distribution')
sns.distplot(train['landmark_id'])
plt.ylabel('ID')
plt.xlabel('Number of Instances')
plt.show()
sns.set()
landmarks_fold_sorted = pd.DataFrame(train['landmark_id'].value_counts())
landmarks_fold_sorted.reset_index(inplace=True)
landmarks_fold_sorted.columns = ['landmark_id','count']
landmarks_fold_sorted = landmarks_fold_sorted.sort_values('landmark_id')
ax = landmarks_fold_sorted.plot.scatter(\
     x='landmark_id',y='count',
     title='Number of images per class(scatter plot)')
locs, labels = plt.xticks()
plt.setp(labels, rotation=30)
ax.set(xlabel="Landmarks", ylabel="Number of images")
print("Number of classes with less than 5 images",(train['landmark_id'].value_counts() < 5).sum())

ival_x = (train['landmark_id'].value_counts() <= 10).sum()
ival_y = (train['landmark_id'].value_counts() <= 4).sum()
ival = ival_x - ival_y
print("Number of classes with images between 5 and 10:" , ival)
#plot random images
import glob
images = glob.glob('../input/landmark-recognition-2020/train/*/*/*/*')
plt.rcParams["axes.grid"] = False
f, axarr = plt.subplots(4, 4, figsize=(24, 22))

current_row = 0
for i in range(16):
    exmpl = cv2.imread(images[i])
    exmpl = exmpl[:,:,::-1]
    
    col = i%4
    axarr[col, current_row].imshow(exmpl)
    if col == 3:
        current_row += 1
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

batch_size = 16
# this is the augmentation configuration used for training
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.1)
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
        "../input/landmark-recognition-2020/train",  #target directory
        target_size=(150, 150),  #images resized to 150x150
        batch_size=batch_size,
        class_mode='binary')  

validation_generator = test_datagen.flow_from_directory(
        "../input/landmark-recognition-2020/test",
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
# this is a similar generator, for validation data
model.fit_generator(
        train_generator,
        steps_per_epoch=5000 // batch_size,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)
model.save_weights('weights.h5')  #saving weights
import csv
# DEBUGGING PARAMS:
NUM_PUBLIC_TRAIN_IMAGES = 1580470 # Used to detect if in session or re-run.
MAX_NUM_EMBEDDINGS = -1  # Set to > 1 to subsample dataset while debugging.


INPUT_DIR = os.path.join('..', 'input')
DATASET_DIR = os.path.join(INPUT_DIR, 'landmark-recognition-2020')
TEST_IMAGE_DIR = os.path.join(DATASET_DIR, 'test')
TRAIN_IMAGE_DIR = os.path.join(DATASET_DIR, 'train')
TRAIN_LABELMAP_PATH = os.path.join(DATASET_DIR, 'train.csv')



def load_labelmap():
  with open(TRAIN_LABELMAP_PATH, mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    labelmap = {row['id']: row['landmark_id'] for row in csv_reader}

  return labelmap

def save_submission_csv(predictions=None):
  """Saves optional `predictions` as submission.csv.

  The csv has columns {id, landmarks}. The landmarks column is a string
  containing the label and score for the id, separated by a ws delimeter.

  If `predictions` is `None` (default), submission.csv is copied from
  sample_submission.csv in `IMAGE_DIR`.

  Args:
    predictions: Optional dict of image ids to dicts with keys {class, score}.
  """

  if predictions is None:
    # Dummy submission!
    shutil.copyfile(
        os.path.join(DATASET_DIR, 'sample_submission.csv'), 'submission.csv')
    return

  with open('submission.csv', 'w') as submission_csv:
    csv_writer = csv.DictWriter(submission_csv, fieldnames=['id', 'landmarks'])
    csv_writer.writeheader()
    for image_id, prediction in predictions.items():
      label = prediction['class']
      score = prediction['score']
      csv_writer.writerow({'id': image_id, 'landmarks': f'{label} {score}'})


def main():
  labelmap = load_labelmap()
  num_training_images = len(labelmap.keys())
  print(f'Found {num_training_images} training images.')

  if num_training_images == NUM_PUBLIC_TRAIN_IMAGES:
    print(
        f'Found {NUM_PUBLIC_TRAIN_IMAGES} training images. Copying sample submission.'
    )
    save_submission_csv()
    return

  _, post_verification_predictions = get_predictions(labelmap)
  save_submission_csv(post_verification_predictions)


if __name__ == '__main__':
  main()
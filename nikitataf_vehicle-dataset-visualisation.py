import os

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import pathlib

import numpy as np

import tensorflow as tf



# Might help: find . -name '.DS_Store' -type f -delete
data_dir = '../input/train/train'

data = []



for category in sorted(os.listdir(data_dir)):

    for file in sorted(os.listdir(os.path.join(data_dir, category))):

        data.append((category, os.path.join(data_dir, category, file)))



df = pd.DataFrame(data, columns=['class', 'file_path'])

len_df = len(df)

print(f"There are {len_df} images")



print(df['class'].value_counts())
df['class'].value_counts().plot(kind='bar')

plt.title('Class counts')
sns.countplot(y=df['class'])

plt.title('Class counts')
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))



CLASS_NAMES = np.array([item.name for item in data_dir.glob('*') if item.name != ".DS_Store"])



image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

BATCH_SIZE = 32

IMG_HEIGHT = 224

IMG_WIDTH = 224

STEPS_PER_EPOCH = np.ceil(image_count / BATCH_SIZE)



train_data_gen = image_generator.flow_from_directory(directory=str(data_dir),

                                                     batch_size=BATCH_SIZE,

                                                     shuffle=True,

                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),

                                                     classes=list(CLASS_NAMES))



image_batch, label_batch = next(train_data_gen)

plt.figure(figsize=(10, 10))

for n in range(25):

    ax = plt.subplot(5, 5, n + 1)

    plt.imshow(image_batch[n])

    plt.title(CLASS_NAMES[label_batch[n] == 1][0].title())

    plt.axis('off')
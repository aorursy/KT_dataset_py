import os
import zipfile 
import random
import numpy as np 
import pandas as pd 
import seaborn as sns

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
np.random.seed(9)
tf.random.set_seed(9)
print(os.listdir("../input/dogs-vs-cats"))
with zipfile.ZipFile("../input/dogs-vs-cats/train.zip","r") as z:
    z.extractall("../kaggle/working/train_unzip")
    
print(f"We have total {len(os.listdir('../kaggle/working/train_unzip/train'))} images in our training data.")
print(f"First 12 filenames: \n {os.listdir('../kaggle/working/train_unzip/train')[:12]}")
train_path = '../kaggle/working/train_unzip/train/'
filenames = os.listdir(train_path)

labels, heights, widths, channels, filesize = [], [], [], [], []

for fname in filenames:
    labels.append(str(fname)[:3])
    img_shape = mpimg.imread(train_path+fname).shape
    heights.append(img_shape[0])
    widths.append(img_shape[1])
    channels.append(img_shape[2])
    filesize.append(os.path.getsize(train_path+fname))

train_df = pd.DataFrame({'filename': filenames, 'label': labels, 'height': heights, 'width': widths, 'channels': channels, 'filesize': filesize})
train_df.head()
print((train_df['label']).value_counts())
dogsVScats_count = train_df['label'].value_counts().plot.bar(title='Number of Dog vs Cat Images in Training Data')
nrows = 3
ncols = 3

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)

for i in range(nrows*ncols):
    sample = np.random.choice(filenames)
    img_path = "../kaggle/working/train_unzip/train/"+sample
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis('Off')
    img = mpimg.imread(img_path)
    plt.imshow(img)
    plt.title(sample[:3])

plt.show()
plt.figure(figsize=(9, 5))

plt.subplot(1, 2, 1)
sns.distplot(train_df['height'], kde=False)
plt.title('Distribution of Image HEIGHTs\nthroughout training data')

plt.subplot(1, 2, 2)
sns.distplot(train_df['width'], kde=False)
plt.title('Distribution of Image WIDTHs\nthroughout training data')

plt.tight_layout()
plt.show()
plt.figure(figsize=(9, 5))

plt.subplot(1, 2, 1)
sns.distplot(train_df[train_df['label']=='dog']['height'], label='dog')
sns.distplot(train_df[train_df['label']=='cat']['height'], label='cat')
plt.title('Cats vs Dogs image \nHEIGHT distribution')
plt.legend()

plt.subplot(1, 2, 2)
sns.distplot(train_df[train_df['label']=='dog']['width'], label='dog')
sns.distplot(train_df[train_df['label']=='cat']['width'], label='cat')
plt.title('Cats vs Dogs image \nWIDTH distribution')
plt.legend()

plt.tight_layout()
plt.show()
train_df.describe()
train_set_df, dev_set_df = train_test_split(train_df[['filename', 'label']], test_size=0.3, random_state = 42, shuffle=True, stratify=train_df['label'])
print(train_set_df.shape, dev_set_df.shape)
print(train_set_df['label'].value_counts())
train_set_plot = train_set_df['label'].value_counts().plot.bar(title='Number of Dog vs Cat Images in train set')
print(dev_set_df['label'].value_counts())
dev_set_plot = dev_set_df['label'].value_counts().plot.bar(title='Number of Dog vs Cat Images in dev set')
train_datagen = ImageDataGenerator( rescale = 1.0/255,
                                    rotation_range=40,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest' )

validation_datagen  = ImageDataGenerator( rescale = 1.0/255 )
sample = train_df.sample(n=1)
sample_generator = train_datagen.flow_from_dataframe(
    sample, 
    "../kaggle/working/train_unzip/train/", 
    x_col='filename',
    y_col='label',
    target_size=(150,150),
    class_mode='categorical'
)

plt.figure(figsize=(10, 10))
for i in range(0, 9):
    plt.subplot(3, 3, i+1)
    for x_batch, y_batch in sample_generator:
        image = x_batch[0]
        plt.imshow(image)
        plt.axis('Off')
        break
plt.tight_layout()
plt.show()
train_generator = train_datagen.flow_from_dataframe(
    train_set_df, 
    directory="../kaggle/working/train_unzip/train/", 
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=32,
    validate_filenames=False 
)

validation_generator = validation_datagen.flow_from_dataframe(
    dev_set_df, 
    directory="../kaggle/working/train_unzip/train/", 
    x_col='filename',
    y_col='label',
    target_size=(150, 150),
    class_mode='binary',
    batch_size=32,
    validate_filenames=False 
)
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1, activation='sigmoid')  
])

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics = ['accuracy'])
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=100,
                    epochs=40,
                    validation_steps=50
                   )
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs   = range(len(acc))

plt.plot(epochs, acc, label="Training accuracy")
plt.plot(epochs, val_acc, label="Validation accuracy")
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, label="Training loss")
plt.plot(epochs, val_loss, label="Validation loss")
plt.title('Training and validation loss')
plt.legend()
plt.show()
loss, accuracy = model.evaluate_generator(validation_generator)
print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))
dev_true = dev_set_df['label'].map({'dog': 1, "cat": 0})
dev_predictions =  model.predict_generator(validation_generator)
dev_set_df['pred'] = np.where(dev_predictions>0.5, 1, 0)
dev_pred = dev_set_df['pred']
dev_set_df.head()
dev_set_predictions_plot = dev_set_df['pred'].value_counts().plot.bar(title='Predicted number of Dog vs Cat Images in dev set')
confusion_mtx = confusion_matrix(dev_true, dev_pred) 

f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Blues",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
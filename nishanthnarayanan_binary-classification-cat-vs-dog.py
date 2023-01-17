import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np 
import pandas as pd 
import random

import matplotlib.pyplot as plt

import os
import zipfile

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
# Extracting the zip file

local_zip = '/kaggle/input/dogs-vs-cats/train.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()
base_dir = '/tmp/train'
img_names = os.listdir(os.path.join(base_dir))
img_names[:10]
sample = random.choice(img_names)
image = load_img("/tmp/train/"+sample)
plt.imshow(image)
plt.show()
# Let's assign the label Dog and Cat for the images

label = []

for i in img_names:
    if(i.split(".")[0] == "dog"):
        label.append("Dog")
    else:
        label.append("Cat")

label[:10]
df = pd.DataFrame({"Image" : img_names, "Label" : label})
df.head()
df.shape
df.Label.value_counts().plot.bar(color = ['red','blue'])
plt.show()
train_df, validate_df = train_test_split(df, test_size = 0.2)
train_df = train_df.reset_index(drop = True)
validate_df = validate_df.reset_index(drop = True)
model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
model.compile(loss = 'binary_crossentropy',
              optimizer = 'SGD',
              metrics = ['accuracy'])
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "/tmp/train/", 
    x_col='Image',
    y_col='Label',
    target_size = (150, 150),
    class_mode = 'binary',
    batch_size = 20
)
validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "/tmp/train/", 
    x_col='Image',
    y_col='Label',
    target_size = (150, 150),
    class_mode = 'binary',
    batch_size = 20
)
history = model.fit_generator(
      train_generator,
      steps_per_epoch = np.ceil(20000/20),  # 20000 images = batch_size * steps
      epochs = 10,
      validation_data=validation_generator,
      validation_steps = np.ceil(5000/20),  # 5000 images = batch_size * steps
      verbose = 1)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(7,7))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure(figsize=(7,7))

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
# Extracting the zip file

local_zip = '/kaggle/input/dogs-vs-cats/test1.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('/tmp')
zip_ref.close()

test_dir = '/tmp/test1/'
test_img = os.listdir(os.path.join(test_dir))
test_img[:10]

test_df = pd.DataFrame({'Image': test_img})
test_gen = ImageDataGenerator(rescale=1./255)

test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "/tmp/test1/", 
    x_col = 'Image',
    y_col = None,
    class_mode = None,
    target_size = (150, 150),
    batch_size = 20,
    shuffle = False
)
predict = model.predict_generator(test_generator, steps = np.ceil(12500/20))
predict
def label(predict):
    if(predict > 0.5):
        return "Dog"
    else:
        return "Cat"
test_df['Label'] = predict
test_df['Label'] = test_df['Label'].apply(label)
test_df.head()
test_df.Label.value_counts()
test_df.Label.value_counts().plot.bar(color = ['red','blue'])
plt.show()
v = random.randint(0, 12000)

sample_test = test_df.iloc[v:(v+18)].reset_index(drop = True)
sample_test.head()

plt.figure(figsize=(12, 24))
for index, row in sample_test.iterrows():
    filename = row['Image']
    category = row['Label']
    img = load_img("/tmp/test1/" + filename, target_size = (150, 150))
    plt.subplot(6, 3, index + 1)
    plt.imshow(img)
    plt.xlabel(filename + ' ( ' + "{}".format(category) + ' )' )
plt.tight_layout()
plt.show()
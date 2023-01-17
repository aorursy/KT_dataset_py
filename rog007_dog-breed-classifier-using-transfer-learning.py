# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import seaborn as sns
import pickle
import cv2
import os

import matplotlib.pyplot as plt
%matplotlib inline
image_path = '../input/stanford-dogs-dataset/images/Images/'

breeds = [str(f.split('-')[1]) for f in os.listdir(image_path)]
image_size = 224

train_datagen = ImageDataGenerator(
                        rescale=1./255,
                        validation_split=0.2,
                        horizontal_flip=True,
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.2,
                        rotation_range=40,
                        fill_mode='nearest'
                        )

train_generator = train_datagen.flow_from_directory(
                        image_path, 
                        target_size=(image_size,image_size),
                        subset='training',
                        shuffle=True,
                        batch_size=24,
                        class_mode='categorical'
                        )

val_datagen = ImageDataGenerator(
                        validation_split=0.2,
                        rescale=1./255
                        )

val_generator = val_datagen.flow_from_directory(
                        image_path, 
                        target_size=(image_size,image_size),
                        subset='validation',
                        shuffle=False,
                        batch_size=24,
                        class_mode='categorical'
                        )
labels = train_generator.class_indices
labels = {v:k for k,v in labels.items()}
#transfer learning

from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras import layers

inception_base = InceptionV3(include_top=False, weights='imagenet', input_shape=(224,224,3))

for layer in inception_base.layers:
    layer.trainable = False

top = inception_base.output
x = layers.GlobalAveragePooling2D()(top)
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dense(units=1024, activation='relu')(x)
x = layers.Dense(units=512, activation='relu')(x)
x = layers.Dense(units=len(breeds), activation='softmax')(x)

model = Model(inputs=inception_base.input, outputs=x)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
plot_model(model, to_file='model_chart.png',show_layer_names=True,show_shapes=True)
hist = model.fit_generator(train_generator,
                           steps_per_epoch=688,
                           epochs=15,
                           validation_data=val_generator,
                           validation_steps=170)
model.save('/kaggle/working/breed.h5')
def plt_model(history):
    
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,4))
    fig.suptitle('Model Accuracy and Loss')

    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.title.set_text('Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train','Valid'],loc=4)

    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.title.set_text('Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train','Valid'],loc=1)

    fig.show()
    
plt_model(hist)
#prediction

val_generator.reset()

predictions = model.predict_generator(val_generator, steps=len(val_generator))
y = np.argmax(predictions, axis=1)
print('Classification report:')
print(classification_report(val_generator.classes, y, target_names=val_generator.class_indices))
print('Confusion matrix:')
conf_mat = confusion_matrix(val_generator.classes, y)
df = pd.DataFrame(conf_mat, columns=val_generator.class_indices)
plt.figure(figsize=(100, 100))
sns.heatmap(df, annot=True)
def test_image(image):
    image = preprocess_input(image)
    image = cv2.resize(image, (image_size, image_size))
    predicted = model.predict(np.expand_dims(image, axis=0))
    index = np.argmax(predicted, axis=1)[0]
    predicted_breed = labels[index].split('-')[1]
    return str(predicted_breed.replace('_', ' '))
img_path = '../input/stanford-dogs-dataset/images/Images/n02085620-Chihuahua/n02085620_2887.jpg'
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
plt.imshow(img)
plt.show()
preds = test_image(img)
print(preds)
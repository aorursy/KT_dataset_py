
import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image # biblioteca para o processamento de imagens
from tqdm import tqdm_notebook as tqdm # biblioteca para exibição de barra de progresso
from pandas.plotting import scatter_matrix
p = sns.color_palette()

from collections import OrderedDict
import cv2
import keras
# For one-hot-encoding
from keras.utils import np_utils
# For creating sequenttial model
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
# For saving and loading models
from keras.models import load_model
import tensorflow as tf

import random

!ls -lh ../input/vehicle/
!ls -lh ../input/vehicle/train/train
!ls -lh ../input/vehicle/test/testset | head -5
root = '../input/vehicle/train/train/'
data = []
for category in sorted(os.listdir(root)):
    for file in sorted(os.listdir(os.path.join(root, category))):
        data.append((category, os.path.join(root, category,  file)))

df = pd.DataFrame(data, columns=['class', 'file_path'])
print(df.info())
df.head()
print(df.info())
print(df.describe())
fig = plt.figure(figsize=(25, 16))
for num, category in enumerate(sorted(df['class'].unique())):
    for i, (idx, row) in enumerate(df.loc[df['class'] == category].sample(4).iterrows()):
        ax = fig.add_subplot(17, 4, num * 4 + i + 1, xticks=[], yticks=[])
        im = Image.open(row['file_path'])
        plt.imshow(im)
        ax.set_title(f'Class: {category}')
fig.tight_layout()
plt.show()
print(df.groupby('class').size())
df['class'].value_counts().plot(kind='bar');
root = '../input/vehicle/test/testset/'
data = []
for category in sorted(os.listdir(root)):
    for file in sorted((os.path.join(root, category))):
        data.append((category, os.path.join(root, category,  file)))
        

dt = pd.DataFrame(data, columns=['class', 'file_path'])
dt.head()
print(df.info())
print(dt.describe())

data = []
labels = []

cols = []
col_imgs = []
cols = sorted(cols)

# Creating trainable 224x224 images
#                    -------
for vehicle_class in cols:
    print(vehicle_class + " started .....")
    for filename in df[vehicle_class]:
        try:
            # for empty cols
            if filename == None:
                pass
            else:
                image = cv2.imread("/kaggle/input/vehicle/train/train/{}/".format(vehicle_class) + filename)
                image_from_numpy_array = Image.fromarray(image, "RGB")
                resized_image = image_from_numpy_array.resize((224, 224))
                data.append(np.array(resized_image))

                if vehicle_class == 'Ambulance':
                    labels.append(0)
                elif vehicle_class == 'Barge':
                    labels.append(1)
                elif vehicle_class == 'Bicycle':
                    labels.append(2)
                elif vehicle_class == 'Boat':
                    labels.append(3)
                elif vehicle_class == 'Bus':
                    labels.append(4)
                elif vehicle_class == 'Car':
                    labels.append(5)
                elif vehicle_class == 'Cart':
                    labels.append(6)
                elif vehicle_class == 'Caterpillar':
                    labels.append(7)
                elif vehicle_class == 'Helicopter':
                    labels.append(8)
                elif vehicle_class == 'Limousine':
                    labels.append(9)
                elif vehicle_class == 'Motorcycle':
                    labels.append(10)
                elif vehicle_class == 'Segway':
                    labels.append(11)
                elif vehicle_class == 'Snowmobile':
                    labels.append(12)
                elif vehicle_class == 'Tank':
                    labels.append(13)
                elif vehicle_class == 'Taxi':
                    labels.append(14)
                elif vehicle_class == 'Truck':
                    labels.append(15)
                elif vehicle_class == 'Van':
                    labels.append(16)
                else:
                    print("Something is wrong.")
                
        except AttributeError:
            print("Attribute error occured for "+filename)
vehicle_images_224x224 = np.array(data)
labels_224x224 = np.array(labels)

# save
np.save("all-vehicle-224x224-images-as-arrays", vehicle_images_224x224)
np.save("corresponding-labels-for-all-224x224-images", labels_224x224)
print(vehicle_images_224x224.shape)
print(labels_224x224.shape)
print(np.unique(labels_224x224))
# Move images to `test` and `train` dir
import shutil
import os

os.mkdir("/kaggle/working/data")
os.mkdir("/kaggle/working/data/test")
os.mkdir("/kaggle/working/data/train")

classes = []

for dir in ["test", "train"]:
    for _class in classes:
        os.mkdir("/kaggle/working/data/{}/{}".format(dir, _class))

for _class in classes:
    images = os.listdir("/kaggle/input/vehicle/train/train/{}".format(_class))

    test = images[:300]
    
    # downsample to 1.5k images
    if len(images) < 1500:
      train = images[300:]
    else:
      train = images[300:1500]

    # move images to test-set folder
    for image in test:
        shutil.copy("/kaggle/input/vehicle/train/train/{}/{}".format(_class, image), "/kaggle/working/data/test/{}/{}".format(_class, image))

    # move images to train-set folder
    for image in train:
        shutil.copy("/kaggle/input/vehicle/train/train/{}/{}".format(_class, image), "/kaggle/working/data/train/{}/{}".format(_class, image))
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from PIL import Image
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json
input_path = "/kaggle/working/data/"
train_datagen = ImageDataGenerator(
    shear_range=10,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    input_path + 'train',
    batch_size=32,
    #class_mode='binary',
    target_size=(224,224))

validation_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

validation_generator = validation_datagen.flow_from_directory(
    input_path + 'test',
    shuffle=False,
    #class_mode='binary',
    target_size=(224,224))
conv_base = ResNet50(
    include_top=False,
    weights='imagenet')

for layer in conv_base.layers:
    layer.trainable = False

x = conv_base.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x) 
predictions = layers.Dense(7, activation='softmax')(x)
model = Model(conv_base.input, predictions)

# Note sgd 
optimizer = keras.optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2/60)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=347 // 32, 
                              epochs=60,
                              validation_data=validation_generator,
                              validation_steps=10  
                             )
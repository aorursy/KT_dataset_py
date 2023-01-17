# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
monkey_species = pd.read_csv('../input/10-monkey-species/monkey_labels.txt')
monkey_species
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(20, 10))
for i in range(10):
    plt.subplot(2, 5, i+1)
    class_name = 'n' + str(i)
    plt.imshow(plt.imread('../input/10-monkey-species/validation/validation/' + class_name + '/' + class_name + '00.jpg'))
    plt.xlabel(class_name)
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

random_seed = 777
num_classes = 10
# the default input of ResNet50 model is 224x224
image_size = 224

# preprocessing_function could not be used currently, https://github.com/keras-team/keras/issues/9624
# data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
data_generator = ImageDataGenerator()

train_directory = '../input/10-monkey-species/training/training'
validation_directory = '../input/10-monkey-species/validation/validation'

train_generator = data_generator.flow_from_directory(
    directory=train_directory,
    batch_size=20,
    seed=random_seed,
    target_size=(image_size, image_size),
    class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    directory=validation_directory,
    seed=random_seed,
    target_size=(image_size, image_size),
    shuffle=False,
    class_mode='categorical')
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

resnet_weights_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

monkey_model = Sequential()
monkey_model.add(ResNet50(include_top=False, pooling='avg', weights=resnet_weights_path))
monkey_model.add(Dense(num_classes, activation='softmax'))

monkey_model.layers[0].trainable = False

monkey_model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
monkey_model.fit_generator(
    generator=train_generator,
    steps_per_epoch=55,
    epochs=3,
    validation_data=validation_generator,
    validation_steps=1)
preds = monkey_model.predict_generator(generator=validation_generator, steps=1)
top3_idx = np.argsort(preds, axis=1)[:, -1:-4:-1]
top3_idx[:5]
from IPython.display import Image, display

for row_idx, row in enumerate(top3_idx[:5]):
    filename = validation_generator.filenames[row_idx]
    display(Image(validation_directory + '/' + filename, width=500))
    print([(filename, class_idx, preds[row_idx, class_idx]) for col_idx, class_idx in enumerate(row)])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# modules from Neural nets Keras API
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# With Augmentation
ds = ImageDataGenerator(
                       rotation_range=40,
                       width_shift_range=0.2,
                       height_shift_range=0.2,
                       shear_range=0.2,
                       zoom_range=0.2,
                       horizontal_flip=True,
                       rescale = 1/255, 
                       validation_split = 0.3
)

# Without Augmentation
# ds = ImageDataGenerator(rescale = 1/255,
#                         validation_split = 0.3)
img_height = 120
img_width = 210
data_dir_name = "/kaggle/input/fruits-vegetables/fruits/"

train_dataset = ds.flow_from_directory(directory = data_dir_name,
                                      subset='training',
                                      target_size = (img_height ,img_width),
#                                           shuffle = False,
                                      class_mode = 'sparse'
                                      )

validation_dataset = ds.flow_from_directory(directory = data_dir_name,
                                          subset='validation',
                                          target_size = (img_height ,img_width),
#                                           shuffle = False,
                                          class_mode = 'sparse'
                                           )
class_names = list(train_dataset.class_indices)
num_classes = len(set(class_names))

print(f'class names:{class_names}')
print(f'num of classes:{num_classes}')
class_names[0]
(a, b) = train_dataset[0]
a[0]
b
# Let's check the photos are really fits to the labels (and not shuffled)
def validate_not_shuffled():
    example_idx = np.random.randint(low=0, high=32) # between 0 -31, because batch size is 32
    example_class_idx = int(b[example_idx])
    example_class_name = class_names[example_class_idx]
    print('Example: this is suppose to be an image of ' + example_class_name)
    plt.imshow(a[example_idx], interpolation='nearest')
    plt.show()
    
validate_not_shuffled()
validate_not_shuffled()
i = Input(shape=a[0].shape)
# x = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)(i) # standartization layer
x = Conv2D(32, (3, 3), activation='relu', padding='same')(i)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.2)(x)

x = GlobalMaxPooling2D()(x)

x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(num_classes, activation='softmax')(x)

model = Model(i, x)
model.compile(
  optimizer='RMSprop',
  loss='sparse_categorical_crossentropy',
  metrics=['accuracy'])
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    batch_size = 32,
    epochs=5
)
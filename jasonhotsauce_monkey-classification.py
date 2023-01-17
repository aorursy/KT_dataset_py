import numpy as np
import pandas as pd

def _extract_labels():
    label_path = '../input/10-monkey-species/monkey_labels.txt'
    data = pd.read_csv(label_path, sep=',', header=0, index_col=False).rename(columns=lambda x: x.strip())
    labels = data['Common Name']
    return labels
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dense, Flatten, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import keras.backend as K

K.set_image_data_format('channels_last')
IMAGE_SIZE = 200
datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = datagen.flow_from_directory('../input/10-monkey-species/training/training', 
                                                      target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                                      batch_size=10)
val_generator = datagen.flow_from_directory('../input/10-monkey-species/validation/validation', 
                                                    target_size=(IMAGE_SIZE, IMAGE_SIZE))
resnet_weight_path = '../input/resnet50/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
model = Sequential()
model.add(ResNet50(include_top=False, pooling='max', weights=resnet_weight_path, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
model.add(BatchNormalization())
model.add(Dense(10, activation='softmax'))
model.layers[0].trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-3, decay=1e-4), metrics=['accuracy'])
model.fit_generator(train_generator, steps_per_epoch=110, epochs=5, validation_data=val_generator, validation_steps=1, workers=5)
model.save('model.h5')
from PIL import Image 

input_img_path ='../input/test-monky/img_5326.jpg'
img = Image.open(input_img_path)
resized_img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC)
img_array = np.array(resized_img, dtype='float32')
(img_row, img_column, img_channels) = img_array.shape
input_img = img_array.reshape(1, img_row, img_column, img_channels)
index = model.predict_classes(input_img)[0]
labels = _extract_labels()
print(labels[index])

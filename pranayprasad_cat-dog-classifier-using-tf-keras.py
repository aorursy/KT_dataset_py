import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
tf.__version__
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)
training_set = train_datagen.flow_from_directory('../input/dogs-cats-images/dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('../input/dogs-cats-images/dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')
cnn = tf.keras.models.Sequential()
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
cnn.fit(x = training_set, validation_data = test_set, epochs = 25)
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('../input/single-predictions/cat_or_dog1.jpg')
test_image
# 1. Reassigning test_image to the same target_size as the other trained images.
# 2. Converting the image to numpy array 
# 3. Adding the batch dimension to the data as while training we had that dimension too.

test_image = image.load_img('../input/single-predictions/cat_or_dog1.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = cnn.predict(test_image)
training_set.class_indices # This will show us the indexes of cat and dog
if result[0][0] == 1: # First '[0]' represents the batch number. Since there is only 1 batch therefore 0.
                      # Second '[0]' represents the image number. Since there is only 1 image therefore 0.
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)
test_image2 = image.load_img('../input/single-predictions/cat_or_dog2.jpg')
test_image2
test_image2 = image.load_img('../input/single-predictions/cat_or_dog2.jpg', target_size = (64, 64))
test_image2 = image.img_to_array(test_image2)
test_image2 = np.expand_dims(test_image2, axis = 0)

result = cnn.predict(test_image2)

if result[0][0] == 1: 
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)
test_image3 = image.load_img('../input/single-predictions/cat_or_dog3.jpg')
test_image3
test_image3 = image.load_img('../input/single-predictions/cat_or_dog3.jpg', target_size = (64, 64))
test_image3 = image.img_to_array(test_image3)
test_image3 = np.expand_dims(test_image3, axis = 0)

result = cnn.predict(test_image3)

if result[0][0] == 1: 
  prediction = 'dog'
else:
  prediction = 'cat'

print(prediction)
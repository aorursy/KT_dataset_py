test_dir="/kaggle/input/dog vs cat/dataset/test_set"
train_dir="/kaggle/input/dog vs cat/dataset/training_set"
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)

batch_size=64

training_set = train_datagen.flow_from_directory(train_dir,
target_size = (100, 100),
batch_size = batch_size,
color_mode='rgb',
class_mode = 'binary',
shuffle=True)

test_set = test_datagen.flow_from_directory(test_dir,
target_size = (100, 100),
batch_size = batch_size,
color_mode='rgb',
class_mode = 'binary')
import keras
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense
from keras.optimizers import adam
import numpy as np
# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape = (100, 100, 3)))
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size = (3, 3)))
classifier.add(Conv2D(64, (3, 3), input_shape = (100, 100, 3)))
classifier.add(Activation("relu"))
classifier.add(MaxPooling2D(pool_size = (3, 3)))

classifier.add(Flatten())

classifier.add(Dense(64))
classifier.add(Activation("relu")) 
classifier.add(Dense(128))
classifier.add(Activation("relu")) 
classifier.add(Dense(activation = 'sigmoid', units=1))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
classifier.fit_generator(training_set,
                        steps_per_epoch=np.ceil(training_set.samples / batch_size),
                        epochs=20,
                        validation_steps=np.ceil(test_set.samples / batch_size),
                         validation_data=test_set
                        )
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
test_image = image.load_img("/kaggle/input/dog vs cat/dataset/training_set/dogs/dog.1056.jpg", target_size = (100, 100)) 
plt.imshow(test_image)
plt.grid(None) 
plt.show()
res_list= ["It's a cat !","It's a dog !"]
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
print(res_list[int(classifier.predict(test_image))])
classifier.predict(test_image)
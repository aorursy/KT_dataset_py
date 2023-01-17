import os


BASE_DIR = '../input/fruits/fruits-360/'
TRAIN_DIR = os.path.join(BASE_DIR, 'Training')
TEST_DIR = os.path.join(BASE_DIR, 'Test')

CATEGORIES = os.listdir(TRAIN_DIR)

print('Categories: ', len(CATEGORIES))
import matplotlib.pyplot as plt


images_in_one_category = []

for category in CATEGORIES:
    path = os.path.join(TRAIN_DIR, category)
    images_in_one_category.append(len(os.listdir(path)))

    
plt.plot(images_in_one_category, 'b-', label='Images in category')

plt.grid()
plt.legend()
plt.show()
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical


train_data = ImageDataGenerator(
    rescale = 1. / 255,
).flow_from_directory(
    TRAIN_DIR,
    target_size = (100, 100),
    batch_size = 64,
    class_mode = 'categorical',
)

test_data = ImageDataGenerator(
    rescale = 1. / 255,
).flow_from_directory(
    TEST_DIR,
    target_size = (100, 100),
    batch_size = 64,
    class_mode = 'categorical',
)
import tensorflow as tf
from tensorflow.keras import losses, optimizers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


model = tf.keras.models.Sequential([
    Conv2D(32, (3, 3), activation=tf.nn.relu, input_shape=(100, 100, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation=tf.nn.relu),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation=tf.nn.relu),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dropout(0.2),
    Dense(256, activation=tf.nn.relu),
    Dense(len(CATEGORIES), activation=tf.nn.softmax),
])


model.compile(
    optimizer = optimizers.RMSprop(lr=0.001),
    loss = losses.categorical_crossentropy,
    metrics = ['accuracy'],
)

model.summary()
history = model.fit_generator(
    train_data,
    steps_per_epoch = 256,
    epochs = 10,
)
plt.plot(history.history['accuracy'], 'r-', label='Train accuracy')

plt.grid()
plt.legend()
plt.show()
loss, accuracy = model.evaluate_generator(test_data)
print('Test accuracy: ', round(accuracy * 100, 2), '%')
print('Test loss: ', loss)

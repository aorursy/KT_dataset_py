import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    count = len([filename for filename in os.listdir(dirname) if os.path.isfile(os.path.join(dirname,filename)) ])
    if count > 0:
        print(dirname.split('/')[-2:], str(count)) 
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import preprocessing
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
train_path = '/kaggle/input/intel-image-classification/seg_train/seg_train/'
valid_path = '/kaggle/input/intel-image-classification/seg_test/seg_test/'
test_path = '/kaggle/input/intel-image-classification/seg_pred/'
img=mpimg.imread(train_path+'/glacier/10.jpg')
imgplot = plt.imshow(img)
plt.show()
img.shape
# Using Preprocessing Techniques From VGG16 model
datagen = preprocessing.image.ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)
train_generator = datagen.flow_from_directory(train_path, color_mode='rgb', batch_size=32, class_mode='categorical', target_size=(150, 150), shuffle=True, seed=42)
valid_generator = datagen.flow_from_directory(valid_path, color_mode='rgb', batch_size=32, class_mode='categorical', target_size=(150, 150), shuffle=True, seed=42)
test_generator = datagen.flow_from_directory(test_path, color_mode="rgb", class_mode='categorical', target_size=(150, 150), batch_size=10, shuffle=False, seed=42)
train_generator.n//train_generator.batch_size
train_generator.fit(images, augment=True, seed=42)
# train_generator.image_shape
imgs, labels = next(train_generator)
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
plotImages(imgs)
print(labels)
input_shape = (150, 150, 3)

model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(6, activation="softmax"),
    ]
)
# loss = tf.keras.backend.sparse_categorical_crossentropy(target, output, from_logits=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics = ['accuracy'])
model.summary()
model.fit(x=train_generator, steps_per_epoch=len(train_generator), validation_data=valid_generator, validation_steps=len(valid_generator), epochs=10, verbose=2)
model.fit(x=train_generator, steps_per_epoch=len(train_generator), validation_data=valid_generator, validation_steps=len(valid_generator), epochs=10, verbose=2)
history = model.fit(x=train_generator, steps_per_epoch=len(train_generator), validation_data=valid_generator, validation_steps=len(valid_generator), epochs=2, verbose=2)
model.evaluate(test_generator)
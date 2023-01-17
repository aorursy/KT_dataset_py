import tensorflow as tf
import numpy as np
import cv2
import glob,os
tf.__version__
from tensorflow.keras import datasets, layers, models
train_data_path = "../input/fruits/fruits-360/Training/*"
test_data_path = "../input/fruits/fruits-360/Test/*"
image_height, image_width = 80, 80 #Real Image Size 100x100 that takes more ram that is unavailable 
train_images = []
train_labels = [] 
for dir_path in glob.glob(train_data_path):
    label = dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (image_height, image_width))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image = getSixChannel(image)
        
        train_images.append(image)
        train_labels.append(label)

train_images = np.array(train_images)
train_labels = np.array(train_labels)
train_image_size = len(train_images)
train_image_size
test_images = []
test_labels = [] 
for dir_path in glob.glob(test_data_path):
    label = dir_path.split("/")[-1]
    for image_path in glob.glob(os.path.join(dir_path, "*.jpg")):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.resize(image, (image_height, image_width))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        #image = getSixChannel(image)
        
        test_images.append(image)
        test_labels.append(label)

test_images = np.array(test_images)
test_labels = np.array(test_labels)
test_image_size = len(test_images)
test_image_size
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer()
train_image_labels = label_binarizer.fit_transform(train_labels)
test_image_labels = label_binarizer.fit_transform(test_labels)

n_classes = len(label_binarizer.classes_)

print(label_binarizer.classes_)
train_images = np.array(train_images, dtype=np.float16) / 225.0
test_images = np.array(test_images, dtype=np.float16) / 225.0
model = models.Sequential()

model.add( layers.Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(image_width, image_height, 3) ) )
model.add( layers.BatchNormalization(axis=-1))
model.add( layers.MaxPooling2D((2, 2)))
model.add( layers.Dropout(0.25) )

model.add( layers.Conv2D(64, (3, 3), activation='relu', padding="same" ) )
model.add( layers.BatchNormalization(axis=-1))
model.add( layers.MaxPooling2D((2, 2)))
model.add( layers.Dropout(0.25) )

model.add( layers.Conv2D(128, (3, 3), activation='relu', padding="same" ) )
model.add( layers.BatchNormalization(axis=-1))
model.add( layers.MaxPooling2D((2, 2)))
model.add( layers.Dropout(0.25) )

model.add( layers.Flatten() )

model.add(layers.Dense(300, activation='relu'))

model.add( layers.BatchNormalization(axis=-1))
model.add( layers.Dropout(0.25) )

model.add(layers.Dense(n_classes, activation='softmax'))

model.summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer=opt,
              loss="categorical_crossentropy",
              metrics=['accuracy'])

checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath = '../input/fruits/fruits_cnn.hdf5', verbose = 1, save_best_only = True)

batch_size = 32


history = model.fit(train_images, 
                    train_image_labels, 
                    epochs=10, 
                    validation_data=(test_images, test_image_labels), 
                    batch_size = batch_size, 
                    callbacks = [checkpointer],
                    steps_per_epoch
                    
                    =len(train_images) // batch_size 
                   )
score = model.evaluate(test_images, test_image_labels, verbose=0)
print('\n', 'Test accuracy:', score[1])

#96.7% Accuracy With 80x80 Images

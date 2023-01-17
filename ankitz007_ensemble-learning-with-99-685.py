import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
df_train = pd.read_csv("../input/digit-recognizer/train.csv")
df_test = pd.read_csv("../input/digit-recognizer/test.csv")
df_train.head()
train_labels = df_train['label']
del df_train['label']
df_train.head()
train_set = df_train.values
train_set = train_set.reshape(42000,28,28)
test_set = df_test.values
test_set = test_set.reshape(28000,28,28)
print(train_set.shape, test_set.shape)
# Create an ImageDataGenerator and do Image Augmentation
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=10,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1)

validation_datagen = ImageDataGenerator(rescale=1.0/255)
num = 10
model = [0]*num
from tensorflow.keras import regularizers
# from tensorflow.keras.callbacks import ReduceLROnPlateau
# ankitz = ReduceLROnPlateau(monitor='val_accuracy', patience = 3, verbose=1,factor=0.5, min_lr=0.00001)

for i in range(num):
    model[i] = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1),padding='same'),
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2,2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2),strides=(2,2)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')])
    model[i].compile(optimizer=tf.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model[0].summary()
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split

epochs = 25
batch_size = 86
history = [0]*num
for i in range(num):
    random_seed = 7
    training_set, validation_set, training_labels, validation_labels = train_test_split(train_set, train_labels, test_size = 0.10, random_state=random_seed)
    training_images = np.expand_dims(training_set, axis=-1)
    testing_images = np.expand_dims(test_set, axis=-1)
    validation_images = np.expand_dims(validation_set, axis=-1)
    ankitz = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.7, min_lr=0.00001)
    history[i] = model[i].fit_generator(train_datagen.flow(training_images, training_labels, batch_size=batch_size),
                              epochs = epochs,
                              steps_per_epoch=training_images.shape[0] // batch_size,
                              validation_data=validation_datagen.flow(validation_images, validation_labels),
                             callbacks = [ankitz])
    print("Model "+str(i+1)+" has finished training..!!")
def predict(X_data):
    results = np.zeros((X_data.shape[0],10))
    for j in range(num):
        results = results + model[j].predict(X_data)
    return results
results = predict(testing_images)
results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)
print(submission[:40])
submission.to_csv("my_results_ensemble.csv",index=False)
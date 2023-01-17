import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
print(os.listdir("../input/digit-recognizer/"))
device_name = tf.test.gpu_device_name()
print(device_name)
batch_size=32
epochs=60
train_data = pd.read_csv("../input/digit-recognizer/train.csv")
test_data = pd.read_csv("../input/digit-recognizer/test.csv")
test_data.head()
def showImage(train_image,label,index):
    image_mtx = train_image.values.reshape(28,28)
    plt.subplot(4,5,index+1)
    plt.imshow(image_mtx , cmap='gray')
    plt.title(label)
    
plt.figure(figsize=(20,10))

first_images = train_data.sample(20).reset_index(drop=True)

for index,row in first_images.iterrows():
    label = row['label']
    image_mtx = row.drop('label')
    showImage(image_mtx,label,index)
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

x = train_data.drop(columns=['label']).values.reshape(train_data.shape[0],28,28,1)
y = to_categorical(train_data['label'])
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rotation_range=10,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
train_datagen.fit(x_train)
train_generator = train_datagen.flow(
    x_train,
    y_train,
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)
train_datagen.fit(x_test)

validation_generator = validation_datagen.flow(
    x_test,
    y_test
    
)
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)

def standardize(x): 
    return (x-mean_px)/std_px
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=10, verbose=1),
    ReduceLROnPlateau(factor=0.3, patience=3, verbose=1),
    ModelCheckpoint('model.h5', verbose=1, save_best_only=True, save_weights_only=True)
]
epochs = 30
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, BatchNormalization

with tf.device('/gpu:0'):
    
    model=Sequential()
 
    model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu", input_shape=(28,28,1)))
    #model.add(Conv2D(filters=64, kernel_size = (3,3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))
    #model.add(Conv2D(filters=128, kernel_size = (3,3), activation="relu"))

    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(BatchNormalization())    
    model.add(Conv2D(filters=256, kernel_size = (3,3), activation="relu",strides=2))

    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512,activation="relu"))
    model.add(Dense(10,activation="softmax"))
    
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    history = model.fit_generator(train_generator, 
                                    steps_per_epoch=len(x_train) // batch_size, 
                                    validation_data=validation_generator,
                                    validation_steps=len(x_test) // batch_size,
                                    epochs=epochs,
                                    callbacks = callbacks)
x_test_recaled = (x_test.astype("float32") / 255)
scores = model.evaluate(x_test_recaled, y_test, verbose=0)
print("{} : {}".format(model.metrics_names[1], scores[1]*100))
print("{} : {}".format(model.metrics_names[0], scores[0]*100))
import seaborn as sns
his_dict = history.history
fig = plt.figure(figsize=(20, 15))
x_range = range(len(history.history['loss']))
sns.set_style('darkgrid')

fig.add_subplot(2,1,1)
sns.lineplot(x=x_range , y=his_dict["val_loss"],label='Validation Loss')
sns.lineplot(x=x_range , y=his_dict["loss"],label='Training Loss')

fig.add_subplot(2,1,2)
sns.lineplot(x=x_range , y=his_dict["val_accuracy"],label='Validation Accuracy')
sns.lineplot(x=x_range , y=his_dict["accuracy"],label='Training Accuracy')
test_digit_data = test_data.values.reshape(test_data.shape[0],28,28,1).astype("float32") / 255
predictions = model.predict(test_digit_data)
results = np.argmax(predictions, axis = 1) 
test_data.head()
plt.figure(figsize=(20, 10))
sample_test = test_data.head(20)
for index, image_pixels in sample_test.iterrows():
    label = results[index]
    showImage(image_pixels, label, index)
from sklearn.metrics import confusion_matrix
preds = model.predict(x_test)
y_preds = np.argmax(preds, axis = 1)
y_test_dec = np.argmax(y_test ,axis=1)

plt.figure(figsize=(10,10))
sns.heatmap(confusion_matrix(y_test_dec,y_preds),cmap='OrRd',annot = True)
submissions = pd.read_csv("../input/digit-recognizer/sample_submission.csv")
submissions['Label'] = results
submissions.to_csv('/kaggle/working/submission.csv', index = False)

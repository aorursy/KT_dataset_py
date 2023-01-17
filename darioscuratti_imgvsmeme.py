import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os, random ,cv2, glob
import keras
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, Activation, MaxPool2D
from keras.optimizers import Adam, RMSprop
from keras.losses import binary_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import keras.preprocessing.image
import matplotlib.pyplot as plt
memes_path = "/kaggle/input/reddit*/memes/memes/*"
memes = glob.glob(memes_path)

print("Memes: ",len(memes))

photo_path = "/kaggle/input/mso*/*/*/*"
photos = glob.glob(photo_path)

print("Photos: ",len(photos))
def prep_data(memes, photos):
    x=[]
    y=[]
    for i in memes:
        image = keras.preprocessing.image.load_img(i, color_mode = "rgb", target_size = (rows,cols))
        image_arr = keras.preprocessing.image.img_to_array(image, data_format = "channels_last")
        x.append(image_arr)
        y.append(1)
        
    for i in photos:
        image = keras.preprocessing.image.load_img(i, color_mode = "rgb", target_size = (rows,cols))
        image_arr = keras.preprocessing.image.img_to_array(image, data_format = "channels_last")
        x.append(image_arr)
        y.append(0)
    return x,y
#define image dimensions 
rows = 150
cols = 150
channels = 3

X, y = prep_data(memes, photos)
#split X,y into a train and validation data sets
X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=(0.2), random_state=1)
#import VGG16
from keras.applications import VGG19
#creating an object of vgg19 model and discarding the top layer
model_vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(rows,cols,channels))
#model_vgg16.summary()
#copy vgg19 layers into our model
model = Sequential()
for layer in model_vgg19.layers:
    model.add(layer)
#freezing vgg19 layers (saving its original weights)
for i in model.layers:
    i.trainable = False
#add top layer for fine-tune VGG19
model.add(Flatten())
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1, activation='sigmoid'))
#check layers trainability
num_layers = len(model.layers)
for x in range(0,num_layers):
    print(model.layers[x])
    print(model.layers[x].trainable)
model.compile(optimizer='Adam', metrics=['accuracy'], loss='binary_crossentropy')
#create a data generator object with some image augmentation specs
datagen = ImageDataGenerator(
    rotation_range = 40,
    rescale=1./ 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
train_gen = datagen.flow(x=np.array(X_train), y=y_train, batch_size=50)
valid_gen = datagen.flow(x=np.array(X_val), y=y_val, batch_size=50)
#train/validate model
history = model.fit(train_gen, steps_per_epoch=100, epochs=22, verbose=1, validation_data=valid_gen, validation_steps=30)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Train', 'Test'], loc='upper left')
plt.title('Model Accuracy')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train', 'Test'], loc='upper left')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
test_path = glob.glob("/kaggle/input/test-images/imgs/*.jpg")

test = []
for i in test_path:
    try:
        image = keras.preprocessing.image.load_img(i, color_mode = "rgb", target_size = (rows,cols))
    except:
        print("Couldn't load Image")
    image_arr = keras.preprocessing.image.img_to_array(image, data_format = "channels_last")
    test.append(image_arr)
test_gen = datagen.flow(x = np.array(test), batch_size=50)
pred = model.predict(test_gen)
print(pred)
ids = [x.rstrip(".jpg").lstrip("/kaggle/input/test-images/imgs/") for x in test_path]
preds = [x[0] for x in pred]

print(ids[0])
df = {'ids': ids, 'predictions': preds}

dataset = pd.DataFrame(df)

print(dataset.head())
dataset.to_csv("predictions.csv")
model.save_weights("ImgMemeWeights.h5")
modelJson = model.to_json()
import json

with open("config.json", "w") as file:
    json.dump(modelJson, file)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import layers
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D,MaxPool2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.models import Model,Sequential

        
import keras.backend as K
K.set_image_data_format('channels_last')

from matplotlib.pyplot import imshow
from keras.preprocessing import image
from keras import applications
from keras.models import Sequential
import os,sys
import warnings
warnings.simplefilter("ignore")

import cv2
from keras.preprocessing.image import ImageDataGenerator
os.listdir('../input/s')
img = cv2.imread('../input/stanford-dogs-dataset-traintest/cropped/test/n02086240-Shih-Tzu/n02086240_11551.jpg')
print(img.shape)
img1 = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img1)
train_dir = '../input/stanford-dogs-dataset-traintest/cropped/train/'
test_dir = '../input/stanford-dogs-dataset-traintest/cropped/test/'
# Training generator
datagen = ImageDataGenerator( 
    rescale=1./255,
)

test_datagen = ImageDataGenerator(rescale=1./255,    validation_split=0.33)


train_generator = datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True,
    seed=42, 
)

# Valid generator

valid_generator = test_datagen.flow_from_directory(
    directory=train_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=64,
    class_mode="categorical",
    shuffle=True,
    seed=42, 
    subset="validation"
)

# Test generator

test_generator = test_datagen.flow_from_directory(
    directory=test_dir,
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
    seed=42
)
from keras import optimizers
from keras.callbacks import ReduceLROnPlateau

optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)
trdata = ImageDataGenerator()
traindata = trdata.flow_from_directory(directory=train_dir,target_size=(224,224))
tsdata = ImageDataGenerator()
testdata = tsdata.flow_from_directory(directory=test_dir, target_size=(224,224))
model = Sequential()

model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))

model.add(Dense(units=120, activation="softmax"))



from keras.optimizers import Adam
model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
hist = model.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=100,callbacks=[checkpoint,early])
import matplotlib.pyplot as plt
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()
from keras.applications import VGG16
img_width, img_height=(224,224)

batch_size = 48*3
nb_train_samples = 12000
nb_validation_samples = ( 8000 // batch_size ) * batch_size
epochs = 6

datagen = ImageDataGenerator(
    horizontal_flip=True,
    shear_range=0.2,
    rescale=1. / 255)

vdatagen = ImageDataGenerator(rescale=1./255)

traingen = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    follow_links=True,
    shuffle=True)

valgen = vdatagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    follow_links=True,
    shuffle=True)

vgg_model = VGG16(input_shape=(224,224,3), weights="imagenet", include_top=False)
for layer in vgg_model.layers:
    layer.trainable = False
model = Sequential()
model.add(vgg_model)
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(120, activation='softmax'))
model.compile(optimizer=Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(traingen,
          epochs=epochs,
          steps_per_epoch=nb_train_samples // batch_size,
          validation_data=valgen,
          validation_steps=nb_validation_samples // batch_size)
import matplotlib.pyplot as plt
plt.plot(history.history["acc"])
plt.plot(history.history['val_acc'])
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("model accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Accuracy","Validation Accuracy","loss","Validation Loss"])
plt.show()
pred=vgg_model.predict(np.expand_dims(img,0))
len(pred[0][0][0])
plt.imshow(l)
xception = Xception(input_shape=(224,224,3), weights="imagenet", include_top=False)
for layer in xception.layers:
    layer.trainable = False
model = Sequential()
model.add(xception)
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(120, activation='softmax'))
model.compile(optimizer=Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])






from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')



fit_stats = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=25,
                    verbose=1,
                    callbacks=[learning_rate_reduction]
)


from keras.applications.xception import Xception
base_model = Xception(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(120, activation='softmax')(x)


for layer in base_model.layers:
    layer.trainable = False
    
xception = Model(inputs=base_model.input, outputs=predictions)
optimizer = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

xception.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
from keras.callbacks import ModelCheckpoint, EarlyStopping
checkpoint = ModelCheckpoint("dog_class.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')

fit_stats = xception.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=25,
                    verbose=1,
                    callbacks=[learning_rate_reduction,checkpoint,early]
)
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size
loss, acc = xception.evaluate_generator(generator=test_generator, steps=STEP_SIZE_TEST, verbose=0)
print(loss, acc)
plt.plot(fit_stats.history['acc'])
pred=xception.predict_generator(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)

labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]
predictions
def get_pred(pred):
    predicted_class_indices=np.argmax(pred,axis=1)
    labels = (traingen.class_indices)
    labels = dict((v,k) for k,v in labels.items())
    print(predicted_class_indices)
    return labels[predicted_class_indices[0]]
img = img/255
plt.imshow(img)
pred=xception.predict(np.expand_dims(img,0))
pred[0][4]*100
np.argmax(pred)
a = np.argsort(-pred)
a
inv_map = {v: k for k, v in train_generator.class_indices.items()}
inv_map[a[0][0]]
b = train_generator.class_indices
from PIL import Image
import requests
from io import BytesIO
def load_img(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img

pred=xception.predict(np.expand_dims(img,0))

pred = pred*100
h=np.argmax(pred,1)
pred
train_generator.class_indices
url = 'https://cdn3-www.dogtime.com/assets/uploads/2020/03/akita-pit-mixed-dog-breed-pictures-1-1442x958.jpg'

response = requests.get(url)
img = Image.open(BytesIO(response.content))

def predict(url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    img=np.array(img)
    plt.imshow(img)

    img= cv2.resize(img,(224,224))
    pred = xception.predict(np.expand_dims(img,0))
    return get_pred(pred)
    
predict(url)
predict('https://i.pinimg.com/236x/44/9a/cd/449acd48951b5a2370b7e2d125030a33.jpg')
model = Model()
model.load_weights('dog_class.h5')
import pickle

# obj0, obj1, obj2 are created here...

# Saving the objects:
with open('classes.pkl', 'wb') as f:  # Pyth
    pickle.dump(b, f)


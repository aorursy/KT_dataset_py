import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras.models import Sequential

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import LearningRateScheduler

from tensorflow.keras.applications import EfficientNetB3


train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
train.head()
train.shape
Y_train = train["label"]

X_train = train.drop(labels = ["label"],axis = 1)

X_train = X_train / 255.0

X_test = test / 255.0

X_train = X_train.values.reshape(-1,28,28,1)

X_test = X_test.values.reshape(-1,28,28,1)

Y_train = to_categorical(Y_train, num_classes = 10)
import matplotlib

import matplotlib.pyplot as plt

# EXTRA

def plot_digits(instances, images_per_row=10, **options):

    size = 28

    images_per_row = min(len(instances), images_per_row)

    images = [instance.reshape(size,size) for instance in instances]

    n_rows = (len(instances) - 1) // images_per_row + 1

    row_images = []

    n_empty = n_rows * images_per_row - len(instances)

    images.append(np.zeros((size, size * n_empty)))

    for row in range(n_rows):

        rimages = images[row * images_per_row : (row + 1) * images_per_row]

        row_images.append(np.concatenate(rimages, axis=1))

    image = np.concatenate(row_images, axis=0)

    plt.imshow(image, cmap = matplotlib.cm.binary, **options)

    plt.axis("off")

    

plt.figure(figsize=(9,9))

example_images = np.r_[X_train[:12000:600], X_train[13000:30600:600], X_train[30600:60000:590]]

plot_digits(example_images, images_per_row=10)

plt.show()
datagen = ImageDataGenerator(

        rotation_range=10,  

        zoom_range = 0.10,  

        width_shift_range=0.1, 

        height_shift_range=0.1)
# PREVIEW AUGMENTED IMAGES

X_train3 = X_train[9,].reshape((1,28,28,1))

Y_train3 = Y_train[9,].reshape((1,10))

plt.figure(figsize=(15,4.5))

for i in range(30):  

    plt.subplot(3, 10, i+1)

    X_train2, Y_train2 = datagen.flow(X_train3,Y_train3).next()

    plt.imshow(X_train2[0].reshape((28,28)),cmap=plt.cm.binary)

    plt.axis('off')

    if i==9: X_train3 = X_train[11,].reshape((1,28,28,1))

    if i==19: X_train3 = X_train[18,].reshape((1,28,28,1))

plt.subplots_adjust(wspace=-0.1, hspace=-0.1)

plt.show()
X_train2.shape
#https://www.kaggle.com/ateplyuk/tpu-tensor-processing-unit-mnist-efficientnet

enet = EfficientNetB3(input_shape=(32, 32, 3), weights='imagenet',include_top=False) 
X_train = np.pad(X_train, ((0,0), (2,2), (2,2), (0,0)), mode='constant')

X_train.shape
X_train = np.squeeze(X_train, axis=-1)

X_train = stacked_img = np.stack((X_train,)*3, axis=-1)

X_train.shape
nets = 2

model = [0] *nets

for j in range(nets):

    model[j] = Sequential(enet)

    model[j].add(Flatten())

    model[j].add(Dense(units=1024, use_bias=True, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Dense(units=512, use_bias=True, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(.4))

    model[j].add(Dense(units=256, use_bias=True, activation='relu'))

    model[j].add(BatchNormalization())

    model[j].add(Dropout(.4))

    model[j].add(Dense(units=10, use_bias=True, activation='softmax'))

    model[j].compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
annealer = LearningRateScheduler(lambda x: 1e-3 * 0.95 ** x)

history = [0] * nets

epochs = 45

for j in range(nets):

    X_train2, X_val2, Y_train2, Y_val2 = train_test_split(X_train, Y_train, test_size = 0.1)

    history[j] = model[j].fit_generator(datagen.flow(X_train2,Y_train2, batch_size=64),

        epochs = epochs, steps_per_epoch = X_train2.shape[0]//64,  

        validation_data = (X_val2,Y_val2), callbacks=[annealer], verbose=0)

    print("EffNet {0:d}: Epochs={1:d}, Train accuracy={2:.5f}, Validation accuracy={3:.5f}".format(

        j+1,epochs,max(history[j].history['accuracy']),max(history[j].history['val_accuracy']) ))
X_test = np.pad(X_test, ((0,0), (2,2), (2,2), (0,0)), mode='constant')

X_test = np.squeeze(X_test, axis=-1)

X_test = stacked_img = np.stack((X_test,)*3, axis=-1)

X_test.shape
results = np.zeros( (X_test.shape[0],10) ) 

for j in range(nets):

    results = results + model[j].predict(X_test)

results = np.argmax(results,axis = 1)

results = pd.Series(results,name="Label")

submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("MNIST_EffNet_Ensemble.csv",index=False)
X_test1a = test / 255.0

X_test1a = X_test1a.values.reshape(-1,28,28,1)
plt.figure(figsize=(15,6))

for i in range(40):  

    plt.subplot(4, 10, i+1)

    plt.imshow(X_test1a[i].reshape((28,28)),cmap=plt.cm.binary)

    plt.title("predict=%d" % results[i],y=0.9)

    plt.axis('off')

plt.subplots_adjust(wspace=0.3, hspace=-0.1)

plt.show()
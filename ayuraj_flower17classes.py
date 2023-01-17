import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt



plt.style.use('seaborn-whitegrid')



from sklearn.preprocessing import MinMaxScaler



from keras.models import Model

from keras.layers import Input, Conv2D, MaxPool2D, Dense

from keras.layers import Flatten, Dropout

from keras.callbacks import EarlyStopping

from keras.preprocessing.image import ImageDataGenerator



import os

print(os.listdir("../input/flowerdataset/flowerDataset/"))
train_df = pd.read_csv("../input/flowerdataset/flowerDataset/flowers17_training.csv", header=None)

test_df = pd.read_csv("../input/flowerdataset/flowerDataset/flowers17_testing.csv", header=None)
train_df.head()
y_train = train_df[0]

X_train = train_df.drop(columns=[0])

y_test = test_df[0]

X_test = test_df.drop(columns=[0])
def get_sample_images(X_train, y_train):

    image_data = []

    labels = []

    print("Loading images for: ", end =" ")

    samples = np.random.choice(len(X_train), 16)

    for sample in samples:

        print("{} |".format(y_train.iloc[sample]), end=" ")

        img = X_train.iloc[sample].values.reshape((64,64,3))

        img = np.flip(img, 2)

        image_data.append(img)

        labels.append(y_train.iloc[sample])

        

    return np.array(image_data), labels
def show_images(images, cols = 1, titles = None):

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)

    if titles is None: print('Serial title'); titles = ['Image (%d)' % i for i in range(1,n_images + 1)]

    fig = plt.figure()

    for n, (image, title) in enumerate(zip(images, titles)):

        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)

        if image.ndim == 2:

            plt.gray()

        plt.imshow(image, cmap=None)

        a.set_title(title, fontsize=50)

        a.grid(False)

        a.axis("off")

    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)

plt.show()
images, labels = get_sample_images(X_train, y_train)
show_images(images, 4, titles=labels)
y_train = pd.get_dummies(y_train)

y_test = pd.get_dummies(y_test)
scaler = MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
X_train = X_train.reshape((X_train.shape[0], 64,64,3))

X_test = X_test.reshape((X_test.shape[0], 64, 64, 3))
aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15, width_shift_range=0.2, height_shift_range=0.2,

                         shear_range=0.15, horizontal_flip=True, fill_mode="nearest")
def modelCNN(inputShape, classes):    

    inputX = Input(inputShape)



    l1 = Conv2D(32, (3, 3), padding="same", activation='relu')(inputX)

    l1 = Conv2D(32, (3, 3), padding="same", activation='relu')(l1)

    l1 = MaxPool2D(pool_size=(2, 2))(l1)

#     l1 = Dropout(0.25)(l1)



    # second CONV => RELU => CONV => RELU => POOL layer set

    l2 = Conv2D(64, (3, 3), padding="same", activation='relu')(l1)

    l2 = Conv2D(64, (3, 3), padding="same", activation='relu')(l2)

    l2 = MaxPool2D(pool_size=(2, 2))(l2)

#     l2 = Dropout(0.25)(l2)



    # third CONV => RELU => CONV => RELU => POOL layer set

    l3 = Conv2D(128, (3, 3), padding="same", activation='relu')(l2)

    l3 = Conv2D(128, (3, 3), padding="same", activation='relu')(l3)

    l3 = MaxPool2D(pool_size=(2, 2))(l3)

#     l3 = Dropout(0.25)(l3)



    # forth CONV => RELU => CONV => RELU => POOL layer set

    l4 = Conv2D(512, (3, 3), padding="same", activation='relu')(l3)

    l4 = Conv2D(512, (3, 3), padding="same", activation='relu')(l4)

    l4 = MaxPool2D(pool_size=(2, 2))(l4)



    # first (and only) set of FC => RELU layers

    l5 = Flatten()(l4)

    l5 = Dropout(0.5)(l5)

    l5 = Dense(512, activation="relu")(l5)



    # softmax classifier

    predictions = Dense(classes, activation="softmax")(l5)

    

    modelCNN = Model(inputs=inputX, outputs=predictions)

    

    return modelCNN
model = modelCNN((64,64,3), 17)
model.summary()
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
earlyStopper = EarlyStopping(monitor='acc', patience=1, restore_best_weights=True)
hist = model.fit_generator(aug.flow(X_train, y_train, batch_size=32),validation_data=(X_test, y_test),

                          steps_per_epoch=len(X_train) // 32, epochs=100)
# plot the training loss and accuracy

plt.figure(figsize=(8,5))

plt.plot(hist.history["loss"], label="train_loss")

plt.plot(hist.history["val_loss"], label="val_loss")

plt.plot(hist.history["acc"], label="train_acc")

plt.plot(hist.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy on Dataset")

plt.xlabel("Epoch #")

plt.ylabel("Loss/Accuracy")

plt.legend(loc="center right");
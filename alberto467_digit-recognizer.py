import pandas as pd



train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
from keras.utils.np_utils import to_categorical



# Separate training X's and Y's

X_train = train.drop(labels=["label"], axis=1)

Y_train = train['label']

del train



# Normalize pixel values

X_train = X_train / 255.

test = test / 255.



# Reshape X's into 28x28 images

X_train = X_train.values.reshape(-1, 28, 28, 1)

test = test.values.reshape(-1, 28, 28, 1)



# One-hot encode labels

#Y_train = to_categorical(Y_train, num_classes=10)
from sklearn.model_selection import train_test_split



X_train, X_val, Y_train, Y_val = train_test_split(

    X_train,

    Y_train,

    test_size=0.1,

    random_state=42

)
from keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    rotation_range=10,

    zoom_range=(1.15, 0.95),

    width_shift_range=0.1,

    height_shift_range=0.1,

    horizontal_flip=False,

    vertical_flip=False,

    #brightness_range=[0.7,1.1],

    shear_range=5

)
import matplotlib.pyplot as plt



fig, axs = plt.subplots(5,10)



imgs = X_train[:25]

imgs_aug = datagen.flow(imgs, batch_size=25, shuffle=False).next()



i = 0



for x in range(5):

    for y in range(5):

        axs[y,x*2].imshow(1-imgs[i][:,:,0], cmap='gray')

        axs[y,x*2].axis('off')

        axs[y,x*2+1].imshow(1-imgs_aug[i][:,:,0], cmap='gray')

        axs[y,x*2+1].axis('off')

        i += 1

        

plt.subplots_adjust(wspace=0, hspace=0)
from keras import Sequential

from keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, LeakyReLU

from keras.initializers import RandomNormal



init = RandomNormal(stddev=0.02)



model = Sequential([

    Conv2D(32, 3, input_shape=(28, 28, 1), activation='relu', kernel_initializer=init),

    BatchNormalization(),

    

    Conv2D(32, 3, activation='relu', kernel_initializer=init),

    BatchNormalization(),

    

    Conv2D(32, 5, strides=2, padding='same', activation='relu', kernel_initializer=init),

    BatchNormalization(),

    Dropout(0.4),

    

    Conv2D(64, 3, activation='relu', kernel_initializer=init),

    BatchNormalization(),

    

    Conv2D(64, 3, activation='relu', kernel_initializer=init),

    BatchNormalization(),

    

    Conv2D(64, 5, strides=2, padding='same', activation='relu', kernel_initializer=init),

    BatchNormalization(),

    Dropout(0.4),

    

    Conv2D(128, 4, activation='relu', kernel_initializer=init),

    BatchNormalization(),

    

    Flatten(),

    Dropout(0.4),

    Dense(10, activation='softmax')

])



model.summary()
from keras.optimizers import Adam



model.compile(

    optimizer=Adam(lr=1e-3),

    loss='sparse_categorical_crossentropy',

    metrics=['sparse_categorical_accuracy']

)
from keras.callbacks import ReduceLROnPlateau



learning_rate_reduction = ReduceLROnPlateau(

    monitor='val_sparse_categorical_accuracy', 

    patience=3, 

    verbose=1, 

    factor=0.5, 

    min_lr=0.00001

)
history = model.fit(

    datagen.flow(X_train, Y_train, batch_size=64),

    epochs=45,

    validation_data=(X_val, Y_val),

    callbacks=[learning_rate_reduction],

    use_multiprocessing=True

)
import matplotlib.pyplot as plt



fig, ax = plt.subplots(2,1)

ax[0].plot(history.history['loss'], color='b', label="Training loss")

ax[0].plot(history.history['val_loss'], color='r', label="validation loss")

legend = ax[0].legend(loc='best')



ax[1].plot(history.history['sparse_categorical_accuracy'], color='b', label="Training accuracy")

ax[1].plot(history.history['val_sparse_categorical_accuracy'], color='r',label="Validation accuracy")

legend = ax[1].legend(loc='best')
import numpy as np



results = model.predict(test)

results = np.argmax(results, axis=1)



submission = pd.concat([

    pd.Series(range(1,28001), name="ImageId"),

    pd.Series(results, name="Label")

], axis=1)



submission.to_csv("submission.csv", index=False)
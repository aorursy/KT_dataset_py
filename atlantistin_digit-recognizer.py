from sklearn.model_selection import train_test_split

from tensorflow import keras

import matplotlib.pyplot as plt

import pandas as pd

import numpy as np



%matplotlib inline

print(pd.__version__, np.__version__, keras.__version__)
data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")

data.describe()
X, y = data.iloc[:, 1:] / 255, data.iloc[:, 0]

X, y = X.values, y.values

print(X.dtype, y.dtype)

X.resize(X.shape[0], 28, 28, 1)

# to_categorical

from keras.utils import to_categorical

y = to_categorical(y, num_classes=10)

# shape

print(type(X), X.shape)

print(type(y), y.shape)
# help(train_test_split)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=3)

print(X_train.shape, y_train.shape)

print(X_val.shape, y_val.shape)
from tensorflow.keras.preprocessing.image import ImageDataGenerator



datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range=0.1,

    shear_range=0.1,

    zoom_range=0.1

)



# datagen.fit

datagen.fit(X_train)
for i, img in zip(range(27), datagen.flow(X_train, batch_size=1, shuffle=True)):

    img.resize(28, 28)

    plt.subplot(3, 9, i+1)

    plt.imshow(img, cmap="gray")

    plt.axis('off')
# help(keras.layers.Conv2D)

regularizer=10E-4



def res_block(x, i, regularizer=regularizer):

    x1 = keras.layers.Conv2D(pow(2, i), (3, 3), activation='relu', 

                             padding='same', kernel_initializer="he_normal", 

                             kernel_regularizer=keras.regularizers.l2(regularizer))(x)

    x2 = keras.layers.Conv2D(pow(2, i), (3, 3), activation='relu', 

                             padding='same', kernel_initializer="he_normal",

                             kernel_regularizer=keras.regularizers.l2(regularizer))(x1)

    x2 = keras.layers.Conv2D(pow(2, i), (3, 3), activation='relu', 

                             padding='same', kernel_initializer="he_normal",

                             kernel_regularizer=keras.regularizers.l2(regularizer))(x2)

    x = keras.layers.concatenate([x1, x2])

    return x



input_X = keras.layers.Input(shape=(28, 28, 1))

x = input_X

for i in range(3, 7):

    x = res_block(x, i)

    x = keras.layers.MaxPooling2D(2, 2)(x)

    x = keras.layers.BatchNormalization()(x)

x = keras.layers.Flatten()(x)

y = keras.layers.Dense(units=10, activation='softmax')(x)

model = keras.models.Model(inputs=input_X, outputs=y)



# ------------------------------------------------------------------------------------

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

model.summary()



# plot_model

from tensorflow.keras.utils import plot_model

plot_model(model, show_shapes=True)



    
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

history = model.fit_generator(

    datagen.flow(X_train, y_train, batch_size=64),

    epochs=32,

    verbose=1,

    validation_data=(X_val, y_val),

    shuffle=True,

    callbacks=[es]

)
test_data = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")

test_imgs = test_data / 255

test_imgs = test_imgs.values

test_imgs.resize(test_imgs.shape[0], 28, 28, 1)

print(test_imgs.shape)
predictions = model.predict(test_imgs).argmax(axis=1)

print(predictions.shape)

print(predictions)
results = pd.DataFrame({"ImageID": range(1, len(test_imgs) + 1),

                        "Label": predictions})

print(results)

results.to_csv("sample_submission.csv", header=True, index=False)
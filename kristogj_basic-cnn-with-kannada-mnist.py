import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



from keras.models import Sequential

from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten, BatchNormalization

from keras.utils import np_utils, plot_model

from keras import regularizers, optimizers
df_train = pd.read_csv("/kaggle/input/Kannada-MNIST/train.csv")

df_test = pd.read_csv("/kaggle/input/Kannada-MNIST/test.csv")

df_test = df_test.drop(columns="id")



# Shuffle

df_train = df_train.sample(frac=1).reset_index(drop=True)
def normalize(X):

    X = X.astype("float")

    mean = X.mean()

    std = X.std() + 10**(-8)

    X -= mean

    X /= std

    return X



def preprocess(df):

    X = df.drop(columns="label").to_numpy()

    # Normalize

    X = normalize(X)

    # Reshape to fit in Conv2D

    X = X.reshape(X.shape[0], 28, 28, 1)

    # One hot encode labels

    y = np_utils.to_categorical(df["label"].to_numpy())

    return X,y



Xtrain, ytrain = preprocess(df_train)



Xtest = df_test.to_numpy()

Xtest = normalize(Xtest)

Xtest = Xtest.reshape(Xtest.shape[0], 28, 28, 1)
img = Xtrain[0]

img = np.array(img, dtype='float')

pixels = img.reshape((28, 28))

plt.imshow(pixels, cmap='gray')

plt.show()

print("Label: {}".format(ytrain[0]))
# Model Architecture

img_width, img_height = Xtrain.shape[1], Xtrain[2]

n_classes = ytrain.shape[1]

model = Sequential()

model.add(Conv2D(64,# Number of filters

                (3,3), # Kernel Size

                input_shape=(img_width,img_width,1),

                activation="relu",

                padding="same"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dropout(0.4))

model.add(Dense(50, activation="relu"))

model.add(Dropout(0.4))

model.add(Dense(n_classes, activation="softmax"))





model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.summary()
plot_model(model)
history = model.fit(Xtrain, ytrain, validation_split=0.2, batch_size=64, epochs=10)
# Plot training & validation accuracy values

plt.plot(history.history['accuracy'])

plt.plot(history.history['val_accuracy'])

plt.title('Model accuracy')

plt.ylabel('Accuracy')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()



# Plot training & validation loss values

plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model loss')

plt.ylabel('Loss')

plt.xlabel('Epoch')

plt.legend(['Train', 'Validation'], loc='best')

plt.show()
predictions = model.predict(Xtest)

predictions = np.argmax(predictions,axis=1)

submission = pd.read_csv('/kaggle/input/Kannada-MNIST/sample_submission.csv')

submission['label'] = predictions

submission.to_csv("submission.csv", index=False)
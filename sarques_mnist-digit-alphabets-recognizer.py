import pandas as pd

import numpy as np

import os



from sklearn.model_selection import train_test_split

from sklearn import metrics



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set(style = "darkgrid")



import gc



import keras

from keras.utils.np_utils import to_categorical

from keras import backend as K

from keras.models import Sequential

from keras.layers import Dense, Flatten, Dropout, MaxPool2D, Conv2D

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.optimizers import Adam

from keras.layers.normalization import BatchNormalization

from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model



import tensorflow as tf



import warnings

warnings.filterwarnings("ignore")



import cv2
df_num = pd.read_csv("../input/train-digit-recognition-mnist/mnist_train.csv")

df_alph = pd.read_csv("../input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data/A_Z Handwritten Data.csv")

df_alph["0"] += 10
df_num.shape, df_alph.shape
df_num.head(3)
df_alph.head(3)
pixel_array = ["Label"]

for i in range(1, 785):

    pixel_array.append(f"pixel_{i}")

df_num.columns = pixel_array

df_alph.columns = pixel_array

del pixel_array

gc.collect()
df = pd.concat([df_num, df_alph], axis = 0)

df.shape
SEED = 42

np.random.seed(SEED)
plt.rcParams["figure.figsize"] = [10, 8]

df.Label.value_counts().plot(kind = "bar")

plt.title("Target value distribution")
a, b = df.Label[df.Label == 24].value_counts().sum(), df.Label[df.Label == 18].value_counts().sum()

print(f"Maximum and Minimum frequency for any target value in the data: {a, b}")
X = df.drop(["Label"], axis = 1)

y = df["Label"]

X.shape
X_reshaped = X.values.astype("float32").reshape(X.shape[0], 28, 28)

y_int = y.values.astype("int32")

print(X.shape, "***", X_reshaped.shape)
X_reshaped = X_reshaped.reshape(-1, 28, 28, 1)



X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_int, test_size = 0.3, stratify = y)



data_to_predict = X_test.reshape(-1, 28, 28)
def plot_grid(pred = False):

    fig=plt.figure(figsize=(8, 8))

    columns = 3

    rows = 3

    for i in range(1, columns*rows +1):

        index = np.random.randint(data_to_predict.shape[0])

        fig.add_subplot(rows, columns, i)

        plt.imshow(data_to_predict[index], cmap = plt.get_cmap("gray"))

        plt.xticks([])

        plt.yticks([])

        plt.xlabel(f"Label: {y_test[index]}")

    plt.tight_layout()

    plt.show()
plot_grid()
X_train_mean = X_train.mean().astype(np.float32)

X_train_std = X_train.std().astype(np.float32)

X_test_mean = X_test.mean().astype(np.float32)

X_test_std = X_test.std().astype(np.float32)



X_train = (X_train - X_train_mean)/X_train_std

X_test = (X_test - X_test_mean)/X_test_std



y_train = to_categorical(y_train, num_classes = 36)

y_test = to_categorical(y_test, num_classes = 36)
del X_train_mean, X_train_std, df_num, df_alph, X_reshaped, y_int, X, y, X_test_mean, X_test_std, data_to_predict, a, b

gc.collect()
# create CNN model for layers

input_shape = (28, 28, 1)

num_classes = 36



model = Sequential()

model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu", padding = "Same", input_shape = input_shape))

model.add(Conv2D(64, kernel_size = (3, 3), activation = "relu", padding = "Same"))

model.add(MaxPool2D(pool_size = (3, 3)))

model.add(Dropout(0.25))



model.add(Conv2D(128, kernel_size = (3, 3), activation = "relu", padding = "Same"))

model.add(Conv2D(128, kernel_size = (3, 3), activation = "relu", padding = "Same"))

model.add(MaxPool2D(pool_size = (3, 3)))

model.add(Dropout(0.40))



model.add(Flatten())

model.add(Dense(150, activation = "relu"))

model.add(Dropout(0.30))

model.add(Dense(36, activation = "softmax"))

model.summary()
optimizer = Adam(lr = .0005, beta_1 = .9, beta_2 = .999, epsilon = 1e-07, decay = 0, amsgrad = False)
# Compile the model

model.compile(optimizer = optimizer, loss = "categorical_crossentropy", metrics = ["categorical_accuracy", tf.keras.metrics.AUC()])



# learning rate annealer

learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_acc', patience = 3, verbose = 1, factor = .5, min_lr = .00001)



# EarlyStopping

es = EarlyStopping(monitor='val_categorical_accuracy', patience = 4)
epochs = 80

batch_size = 128
# Data Augmentation

datagen = ImageDataGenerator(featurewise_center = False, samplewise_center = False, 

                            featurewise_std_normalization = False, samplewise_std_normalization = False,

                            zca_whitening = False, rotation_range = 10, zoom_range = .1, 

                            width_shift_range = .1, height_shift_range = .1, horizontal_flip = True, 

                            vertical_flip = False)

train_batches = datagen.flow(X_train, y_train, batch_size = batch_size)

val_batches = datagen.flow(X_test, y_test, batch_size = batch_size)
# Fitting the model

history = model.fit_generator(generator = train_batches, steps_per_epoch = train_batches.n//batch_size, epochs=epochs, 

                    validation_data = val_batches, validation_steps = val_batches.n//batch_size, verbose = 0,

                    callbacks = [learning_rate_reduction, es])
model.save("model_0-10_a-z.h5")
print(f"Total number of epochs for which the model trained: {len(history.history['loss'])}")
plt.rcParams['figure.figsize'] = [10, 8]

plt.plot(history.history['categorical_accuracy'], "b--")

plt.plot(history.history['val_categorical_accuracy'], "r-")

plt.title("Training vs Validation accuracy")

plt.legend(["Training", "Validation"])

plt.xlabel("Epochs")

plt.ylabel("Accuracy")
print(f"Maximum Training Accuracy: {max(history.history['categorical_accuracy'])}, Maximum Validation Accuracy: {max(history.history['val_categorical_accuracy'])}")
plt.rcParams['figure.figsize'] = [10, 8]

plt.plot(history.history['auc'], "b--")

plt.plot(history.history['val_auc'], "r-")

plt.title("Training vs Validation AUC score")

plt.legend(["Training", "Validation"])

plt.xlabel("Epochs")

plt.ylabel("AUC score")
print(f"Maximum Training AUC: {max(history.history['auc'])}, Maximum Validation AUC: {max(history.history['val_auc'])}")
plt.rcParams['figure.figsize'] = [10, 8]

plt.plot(history.history['loss'], "b--")

plt.plot(history.history['val_loss'], "r-")

plt.title("Training vs Validation Loss")

plt.legend(["Training", "Validation"])

plt.xlabel("Epochs")

plt.ylabel("Loss value")
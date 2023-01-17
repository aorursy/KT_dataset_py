import tensorflow as tf



print(f"TF Version: {tf.__version__}")

print("Is using GPU", "Yes" if tf.config.list_physical_devices('GPU') else "No")
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import pandas as pd



np.random.seed(21)

sns.set(rc={'figure.figsize': (12,8)})



from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix



from tensorflow.keras.models import Sequential

from tensorflow.keras.datasets import fashion_mnist

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
X_train = train.drop('label', axis=1)

y_train = train['label']

X_test = test.drop('label', axis=1)

y_test = test['label']
X_train.shape, y_train.shape, X_test.shape, y_test.shape
y_train[:10]
sns.countplot(y_train)

# Check the data

X_train.isna().sum()
X_train, X_test = X_train / 255., X_test / 255.
X_train.max()
X_train = X_train.to_numpy()

X_test = X_test.to_numpy()

y_train = y_train.to_numpy()

y_test = y_test.to_numpy()
X_train = X_train.reshape(-1, 28, 28, 1)

X_test = X_test.reshape(-1, 28, 28, 1)
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal','Shirt', 'Sneaker', 'Bag', 'Ankle boot']
y_train, y_test = to_categorical(y_train, 10), to_categorical(y_test, 10)
y_train[:5]
X_train, X_val, y_train, y_val = train_test_split(X_train, 

                                                  y_train, 

                                                  test_size=0.2,

                                                  random_state=21)
len(X_train), len(X_val), len(y_train), len(y_val)
def visualize_data(data, labels):

    fig, ax = plt.subplots(nrows=5, ncols=5, figsize=(10,10))

    for i, a in enumerate(ax.ravel()):

        a.imshow(data[i][:,:,0], cmap='binary')

        a.set(title=classes[labels[i].argmax()])

        a.axis('off')

visualize_data(X_train, y_train)
model = Sequential([

    Conv2D(16, (3,3), activation='relu', input_shape=(28,28,1)),

    Conv2D(16, (3,3), activation='relu'),

    MaxPooling2D(),



    Conv2D(32, (3,3), activation='relu'),

    Conv2D(32, (3,3), activation='relu'),

    MaxPooling2D(),



    Flatten(),

    Dense(256, activation='relu'),

    Dropout(0.25),

    Dense(len(classes), activation='softmax')

])

model.summary()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# Callbacks

early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
datagen = ImageDataGenerator(

    rotation_range=10,

    width_shift_range=0.1,

    height_shift_range = 0.1,

    zoom_range=0.1,

    horizontal_flip=False,

    vertical_flip=False

)

datagen.fit(X_train)
history = model.fit_generator(datagen.flow(X_train, y_train),

                    epochs=30,

                    validation_data=(X_val, y_val),

                    callbacks=[early_stop])
model.evaluate(X_test, y_test)
# Plotting the learning graph

acc = history.history['accuracy']

loss = history.history['loss']

val_acc = history.history['val_accuracy']

val_loss = history.history['val_loss']



epochs = range(len(acc))



plt.plot(epochs, acc, "r", label = "Training Accuracy")

plt.plot(epochs, val_acc, "b", label = "Validation Accuracy")

plt.legend(loc='upper right')

plt.title("Training and Validation Accuracy")

plt.figure()



plt.plot(epochs, loss, "r", label = "Training Loss")

plt.plot(epochs, val_loss, "b", label = "Validation Loss")

plt.legend(loc='upper right')

plt.title("Training and Validation Loss")

plt.figure()
# Submitting Predictions to Kaggle

preds = model.predict_classes(X_test)



def write_preds(preds, fname):

    pd.DataFrame({"ImageId": list(range(1,len(preds)+1)), "Label": preds}).to_csv(fname, index=False, header=True)



write_preds(preds, "cnn_mnist_datagen.csv")
import tensorflow as tf    
import keras
from keras.utils import to_categorical
from keras import layers, models, callbacks
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

SEED = 1
np.random.seed(SEED)
sns.set(style="white", context="notebook", palette="deep")
train = pd.read_csv("../input/digit-recognizer/train.csv")
test = pd.read_csv("../input/digit-recognizer/test.csv")
y_train = train["label"]
X_train = train.drop(labels=["label"], axis=1)

sns.countplot(y_train)
print(X_train.isnull().any().any(), y_train.isnull().any().any(), test.isnull().any().any())
X_train /= 255
test /= 255
X_train = X_train.to_numpy().reshape(-1, 28, 28, 1)
test = test.to_numpy().reshape(-1, 28, 28, 1)
print(X_train.shape, test.shape)
y_train = to_categorical(y_train)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, random_state=SEED, train_size=0.9)
print(X_train.shape, X_valid.shape)
plt.figure(figsize=(10, 6))
for i in range(15):
    plt.subplot(3, 5, i + 1)
    plt.imshow(X_train[i,:,:,0])
    plt.xticks([])
    plt.yticks([])
model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="same", input_shape=(28, 28, 1)))
model.add(layers.Conv2D(filters=32, kernel_size=(5, 5), activation="relu", padding="same"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu", padding="same"))
model.add(layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu", padding="same"))
model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(10, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="rmsprop", metrics=["acc"])
learning_rate_reduction = callbacks.ReduceLROnPlateau(monitor='val_acc', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
# model_checkpoints = callbacks.ModelCheckpoint("./checkpoints/weights{epoch:03d}.h5", save_weights_only=True)
epochs = 20
batch_size = 86
datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1)
history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size), 
                             epochs=epochs,
                             validation_data=(X_valid, y_valid),
                             steps_per_epoch=len(X_train) // batch_size,
                             callbacks=[learning_rate_reduction])
plt.subplots(figsize=(10, 12))

plt.subplot(211)
plt.title("Loss")
loss = history.history["loss"]
plt.plot(range(1, len(loss) + 1), loss, "bo-", label="Training Loss")
loss = history.history["val_loss"]
plt.plot(range(1, len(loss) + 1), loss, "ro-", label="Validation Loss")
plt.xticks(range(1, len(loss) + 1))
plt.grid(True)
plt.legend()

plt.subplot(212)
plt.title("Accuracy")
acc = history.history["acc"]
plt.plot(range(1, len(loss) + 1), acc, "bo-", label="Training Acc")
acc = history.history["val_acc"]
plt.plot(range(1, len(loss) + 1), acc, "ro-", label="Validation Acc")
plt.xticks(range(1, len(loss) + 1))
plt.grid(True)
plt.legend()
pred = model.predict(X_valid)
pred_classes = np.argmax(pred, axis=1)
pred_true = np.argmax(y_valid, axis=1)
confusion_mtx = confusion_matrix(pred_true, pred_classes)
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap=plt.cm.Blues)
results = model.predict(test)
results = np.argmax(results, axis=1)
results = pd.concat([pd.Series(range(1, 28001), name="ImageId"),
                     pd.Series(results, name="Label")],
                    axis=1)
results.shape
results.to_csv("out.csv", index=False)
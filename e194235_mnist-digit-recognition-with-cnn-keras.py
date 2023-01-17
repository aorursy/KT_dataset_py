import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv("../input/digit-recognizer/train.csv")

test = pd.read_csv("../input/digit-recognizer/test.csv")
train.shape
test.shape
train.head()
train.info()
train.isnull().sum().any()
sns.countplot(train["label"])
labels = list(range(10))
labels
f, ax = plt.subplots(2,5, figsize=(15,15))
l = 0


for i in range(2):
    for j in range(5):
        img = train.loc[train["label"] == l].sample().values[0][1:].reshape(28,28)
        ax[i,j].imshow(img, cmap="gray")
        l += 1
    plt.tight_layout()
X = train.drop("label", axis=1).values
y = train["label"].values
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=10)
X_test = test.values
y_train.shape
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val_true = y_val.copy()
y_val = to_categorical(y_val)
y_train.shape
X_train.max()
X_train.min()
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255
X_train.shape
X_train = X_train.reshape(-1, 28, 28, 1)
X_val = X_val.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)
X_train.shape
X_val.shape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
image_gen = ImageDataGenerator(rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               shear_range=0.1,
                               zoom_range=0.1,
                               horizontal_flip=False,
                               vertical_flip=False,
                              )
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
model = Sequential()


model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(28, 28, 1), padding="same", activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=32, kernel_size=(5,5), input_shape=(28, 28, 1), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))


model.add(Conv2D(filters=64, kernel_size=(5,5), input_shape=(28, 28, 1), padding="same", activation="relu"))
model.add(BatchNormalization())

model.add(Conv2D(filters=64, kernel_size=(5,5), input_shape=(28, 28, 1), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))


model.add(Conv2D(filters=128, kernel_size=(5,5), input_shape=(28, 28, 1), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))


model.add(Flatten())

model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
#model.add(Dropout(0.25))

model.add(Dense(10, activation="softmax"))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
check_point = ModelCheckpoint("best_model.h5", monitor="val_accuracy", verbose=1, save_best_only=True)
reduce_lr = ReduceLROnPlateau(monitor="val_accuracy", patience=3, verbose=1, factor=0.5, min_lr=0.0001)
#model.fit(X_train, y_train, epochs=100, validation_data=(X_val,y_val), callbacks=[check_point,reduce_lr])
history = model.fit_generator(image_gen.flow(X_train,y_train, batch_size=64),
                              epochs = 50, validation_data = (X_val,y_val),
                              verbose = 1,
                              callbacks=[check_point, reduce_lr])
losses = pd.DataFrame(model.history.history)
losses.head()
losses[["accuracy", "val_accuracy"]].plot()
losses[["loss", "val_loss"]].plot()
print("Accuracy on validation data: {:.4f}".format(losses["val_accuracy"].max()))
from keras.models import load_model
saved_model = load_model('best_model.h5')
predictions = saved_model.predict_classes(X_test)
submission = pd.Series(predictions, name="Label")
submission = pd.concat([pd.Series(range(1,28001), name ="ImageId"), submission], axis = 1)
submission.to_csv("submission.csv", index=False)
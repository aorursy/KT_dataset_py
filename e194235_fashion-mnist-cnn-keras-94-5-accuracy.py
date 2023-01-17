import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
train = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")

test = pd.read_csv("../input/fashionmnist/fashion-mnist_test.csv")
train.shape
test.shape
train.head()
labels = {0: "T-shirt/top",
          1: "Trouser",
          2: "Pullover",
          3: "Dress",
          4: "Coat",
          5: "Sandal",
          6: "Shirt",
          7: "Sneaker",
          8: "Bag",
          9: "Ankle boot"
        }
f, ax = plt.subplots(2,5, figsize=(15,15))
l = 0


for i in range(2):
    for j in range(5):
        img = train.loc[train["label"] == l].iloc[0][1:].values.reshape(28,28)
        ax[i,j].imshow(img, cmap="gray")
        label = labels[l]
        ax[i,j].set_title(label)
        l += 1
    plt.tight_layout()
X = train.drop("label", axis=1).values
y = train["label"].values
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/6, random_state=10)
X_test = test.drop("label", axis=1).values
y_test = test["label"].values
y_test.shape
from keras.utils import to_categorical
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test_true = y_test.copy()
y_test = to_categorical(y_test)
y_test.shape
X_train.max()
X_train.min()
X_train = X_train/255
X_val = X_val/255
X_test = X_test/255
X_train.shape
X_train = X_train.reshape(50000, 28, 28, 1)
X_val = X_val.reshape(10000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)
X_train.shape
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
model = Sequential()

model.add(Conv2D(filters=64, kernel_size=(4,4), input_shape=(28, 28, 1), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(4,4), input_shape=(28, 28, 1), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64, kernel_size=(4,4), input_shape=(28, 28, 1), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Dense(10, activation="softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
model.summary()
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
check_point = ModelCheckpoint("best_model.h5", monitor="val_accuracy", verbose=1, save_best_only=True)
#reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.001)
model.fit(x=X_train, y=y_train, 
         epochs=200, 
         validation_data=(X_val, y_val), 
         callbacks=[check_point])
losses = pd.DataFrame(model.history.history)
losses.head()
losses[["accuracy","val_accuracy"]].plot()
losses[["loss","val_loss"]].plot()
print("Accuracy on validation data: {:.4f}".format(losses["val_accuracy"].max()))
from keras.models import load_model
saved_model = load_model('best_model.h5')
from sklearn.metrics import classification_report,confusion_matrix
predictions = saved_model.predict_classes(X_test)
eval = saved_model.evaluate(X_test,y_test,verbose=0)
print("Accuracy on test data: {:.4f}".format(eval[1]))
print("Loss on test data: {:.4f}".format(eval[0]))
print(classification_report(y_test_true,predictions))
confusion_matrix(y_test_true,predictions)




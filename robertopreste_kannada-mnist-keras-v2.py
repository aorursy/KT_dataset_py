import numpy as np 

import pandas as pd 

import os



from keras.models import Sequential

from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization

from keras.callbacks import EarlyStopping

from keras.utils import to_categorical



from IPython.display import Markdown as md

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from mlxtend.plotting import plot_confusion_matrix
data = '/kaggle/input/kannada-mnist/kannada_mnist_datataset_paper/Kannada_MNIST_datataset_paper/Kannada_MNIST_npz/'



X_train = np.load(os.path.join(data, "Kannada_MNIST/X_kannada_MNIST_train.npz"))["arr_0"]

X_test = np.load(os.path.join(data, "Kannada_MNIST/X_kannada_MNIST_test.npz"))["arr_0"]

X_dig = np.load(os.path.join(data, "Dig_MNIST/X_dig_MNIST.npz"))["arr_0"]



y_train = np.load(os.path.join(data, "Kannada_MNIST/y_kannada_MNIST_train.npz"))["arr_0"]

y_test = np.load(os.path.join(data, "Kannada_MNIST/y_kannada_MNIST_test.npz"))["arr_0"]

y_dig = np.load(os.path.join(data, "Dig_MNIST/y_dig_MNIST.npz"))["arr_0"]
X_train = X_train.astype(np.float32) / 255.0

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)



X_test = X_test.astype(np.float32) / 255.0

X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)



X_dig = X_dig.astype(np.float32) / 255.0

X_dig = X_dig.reshape(X_dig.shape[0], 28, 28, 1)
y_train_raw = y_train.copy()

y_test_raw = y_test.copy()

y_dig_raw = y_dig.copy()



y_train = to_categorical(y_train)

y_test = to_categorical(y_test)

y_dig = to_categorical(y_dig)
np.random.seed(420)

model = Sequential()



model.add(Conv2D(32, input_shape=(28, 28, 1), kernel_size=(3, 3), padding="same", activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(0.2))



model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation="relu"))

model.add(BatchNormalization())

model.add(MaxPooling2D())

model.add(Dropout(0.2))



model.add(Flatten())

model.add(Dense(64, activation="relu"))

model.add(Dropout(0.2))

model.add(Dense(10, activation="softmax"))
model.summary()
early_stop = EarlyStopping(monitor="loss", patience=3, restore_best_weights=True)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

history = model.fit(X_train, y_train, epochs=20, batch_size=32, callbacks=[early_stop], 

                    verbose=0)
plt.rcParams["figure.figsize"] = (20, 10)

plt.rcParams["figure.facecolor"] = "white"



plt.subplot(1, 2, 1)

plt.plot(history.history['acc'])

# plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.ylabel('accuracy')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')



plt.subplot(1, 2, 2)

plt.plot(history.history['loss'])

# plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.ylabel('loss')

plt.xlabel('epoch')

plt.legend(['train'], loc='upper left')



plt.show()
y_pred = model.predict(X_test)

y_pred_raw = np.argmax(y_pred, axis=1)



y_pred_dig = model.predict(X_dig)

y_pred_dig_raw = np.argmax(y_pred_dig, axis=1)
cm = confusion_matrix(y_test_raw, y_pred_raw)

c_report = classification_report(y_test_raw, y_pred_raw, digits=4)

acc = accuracy_score(y_test_raw, y_pred_raw)

plot_confusion_matrix(cm, show_absolute=True, show_normed=True, 

                      class_names=range(10), colorbar=True)

plt.title("Confusion Matrix (Kannada dataset)", fontdict={"fontsize": 22})

plt.show()

print(c_report)
cm_dig = confusion_matrix(y_dig_raw, y_pred_dig_raw)

c_report_dig = classification_report(y_dig_raw, y_pred_dig_raw, digits=4)

acc_dig = accuracy_score(y_dig_raw, y_pred_dig_raw)

plot_confusion_matrix(cm_dig, show_absolute=True, show_normed=True, 

                      class_names=range(10), colorbar=True)

plt.title("Confusion Matrix (Dig dataset)", fontdict={"fontsize": 22})

plt.show()

print(c_report_dig)
md(f"- **{round(acc, 4) * 100}% accuracy on the Kannada test set** (vs 96.85% in the original paper)")
md(f"- **{round(acc_dig, 4) * 100}% accuracy on the Dig test set** (vs 76.17% in the original paper)")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPool2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
np.random.seed(42)
def doSubmission(y_pred):
    test_Id = np.arange(1, y_pred.size+1, dtype=np.int)
    
    pred_dict = {"ImageId": test_Id, "Label": y_pred}
    df = pd.DataFrame(pred_dict)
    df.to_csv("sample_submission.csv", index=False, index_label=False)
df_train = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
df_test = pd.read_csv("/kaggle/input/digit-recognizer/test.csv")
y = df_train.label.to_numpy() # transforming into numpy array

X = df_train.drop(columns=["label"]).to_numpy(np.float64)
X /= 255.0 #normalizing to improve the model learning
X_totrain = X.reshape(X.shape[0], 28, 28, 1) #a complete data base to train the model for prediction
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
y_cat = to_categorical(y, 10)
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
test = df_test.to_numpy(np.float64)
test = test.reshape(test.shape[0], 28, 28, 1)
test /= 255.0
def convNeuralNetwork(filters=256, kernel_size=(3, 3), pool_size=(2, 2), units=128, dropout=0.2):
    cnn = Sequential()
    
    cnn.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1), 
                   input_shape=(28, 28, 1), activation="relu", padding="same"))
    cnn.add(MaxPool2D(pool_size=pool_size, padding="same"))

    cnn.add(Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1),
                   activation="relu", padding="same"))
    cnn.add(MaxPool2D(pool_size=pool_size, padding="same"))
    
    cnn.add(Flatten())
            
    cnn.add(Dense(units=units, activation="relu"))
    cnn.add(Dropout(dropout))
            
    cnn.add(Dense(units=units, activation="relu"))
    cnn.add(Dropout(dropout))
            
    cnn.add(Dense(units=10, activation="softmax"))
    
    cnn.compile(optimizer="adamax", loss="categorical_crossentropy", metrics=["accuracy"])
    
    return cnn
early_stopping = EarlyStopping(monitor="val_loss", patience=10, verbose=1, 
                               restore_best_weights=True)

cnn = convNeuralNetwork(filters=2048, units=1024)
cnn_hist = cnn.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), 
                   epochs=50, batch_size=256, callbacks=[early_stopping])
accuracy = cnn_hist.history["accuracy"]
val_accuracy = cnn_hist.history["val_accuracy"]

plt.plot(accuracy, "o-", label="Accuracy")
plt.plot(val_accuracy, "o-", label="Val Accuracy")

plt.legend(loc="best")
plt.grid()
plt.show()
loss = cnn_hist.history["loss"]
val_loss = cnn_hist.history["val_loss"]

plt.plot(loss, "o-", label="Loss")
plt.plot(val_loss, "o-", label="Val Loss")

plt.legend(loc="best")
plt.grid()
plt.show()
early_stopping = EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(filepath="./", monitor="loss", verbose=1,
                                   save_best_only=True, save_weights_only=True)

model = convNeuralNetwork(filters=2048, units=1024)
model_hist = model.fit(X_totrain, y_cat, batch_size=256, epochs=50, 
                       callbacks=[early_stopping, model_checkpoint])
accuracy = model_hist.history["accuracy"]

plt.plot(accuracy, "o-", label="Accuracy")
plt.legend(loc="best")
plt.grid()
plt.show()
accuracy = model_hist.history["loss"]

plt.plot(accuracy, "o-", label="Loss")
plt.legend(loc="best")
plt.grid()
plt.show()
y_pred = model.predict(test).argmax(1)

doSubmission(y_pred)

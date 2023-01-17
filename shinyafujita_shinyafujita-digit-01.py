import os
import csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
df_train_tmp = pd.read_csv("../input/train.csv")
df_test_tmp = pd.read_csv("../input/test.csv")
def show(image):
    pylab.gray()
    pylab.imshow(image.reshape(28, 28))
tmp_data = df_train_tmp.iloc[:,1:].values.reshape(len(df_train_tmp), 28, 28, 1).astype("float32") / 255
tmp_label = keras.utils.to_categorical(df_train_tmp["label"].values, 10)

train_data = tmp_data[:40000]
train_label = tmp_label[:40000]
valid_data = tmp_data[40000:]
valid_label = tmp_label[40000:]

test_data = df_test_tmp.values.reshape(len(df_test_tmp), 28, 28, 1).astype("float32") / 255
i = 40
show(train_data[i])
print(np.argmax(train_label[i]))
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])
history = model.fit(train_data, train_label, batch_size=128, epochs=20, verbose=1, validation_data=(valid_data, valid_label))
#Accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
#loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
test_predicted = np.argmax(model.predict(test_data), axis=1)
with open("predict_result_data.csv", "w") as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerow(["ImageId", "Label"])
    for i in range(len(test_predicted)):
        writer.writerow([i + 1, test_predicted[i]])
i = 17
show(test_data[i])
test_predicted[i]

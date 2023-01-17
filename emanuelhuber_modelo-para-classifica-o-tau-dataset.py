import os
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from PIL import Image # biblioteca para o processamento de imagens
from tqdm import tqdm_notebook as tqdm # biblioteca para exibição de barra de progresso
root = '../input/vehicle/train/train'
data = []
for category in sorted(os.listdir(root)):
    for file in sorted(os.listdir(os.path.join(root, category))):
        data.append((category, os.path.join(root, category,  file)))

df = pd.DataFrame(data, columns=['class', 'file_path'])
df.head()
from sklearn.model_selection import train_test_split

X = df["file_path"].values
classes = dict()
for i, _class in enumerate(df["class"].unique()):
    classes[_class] = i
y = df["class"].apply(lambda x: classes[x]).values

X_train, X_test, Y_train, Y_test = train_test_split( \
     X, y, test_size=0.33, random_state=42)
X_train.shape, X_test.shape, Y_train.shape
from tensorflow import keras
from tensorflow.keras import layers

# Model / data parameters
image_size = (320, 320)
num_classes = len(classes)
input_shape = image_size + (3,)


# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()
batch_size = 128

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Executando uma época 

for i in range(int(X_train.shape[0]/batch_size)):
    x_train = np.array([np.array(tf.keras.preprocessing.image.load_img(x, target_size=image_size)) for x in X_train[i*batch_size:(i+1)*batch_size]])
    y_train = Y_train[i*batch_size:(i+1)*batch_size]
        
    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    
    ret = model.train_on_batch(x_train, y_train)
    print(f"Train loss: {ret[0]} \t\t accuracy:{ret[1]}")

# model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
test_loss = []
test_acc = []
for i in range(int(X_test.shape[0]/batch_size)):

    x_test = np.array([np.array(tf.keras.preprocessing.image.load_img(x, target_size=image_size)) for x in X_test[i*batch_size:(i+1)*batch_size]])
    x_test = x_test.astype("float32") / 255
    y_test = Y_test[i*batch_size:(i+1)*batch_size]
    score = model.evaluate(x_test, y_test, verbose=0)
    test_loss.append(score[0])
    test_acc.append(score[1])
print(f"Test loss: {np.mean(test_loss)}\t\t accuracy:{np.mean(test_acc)}")
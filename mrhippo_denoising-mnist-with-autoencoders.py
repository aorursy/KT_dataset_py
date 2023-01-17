# data processing
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

# keras
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Input
from keras.models import Sequential, Model

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
def show_samples(data):

    fig, ax = plt.subplots(4, 5, figsize = (15, 8))
    index = 1

    for column in range(0, 5):
        for row in range(0, 4):
            ax[row, column].imshow(data[index].reshape(28,28), cmap = "Blues")
            ax[row, column].axis(False)
            index += 200

    plt.show()
def show_predicted_samples(model, data):
    
    predicted = model.predict(data)
    
    fig, ax = plt.subplots(4, 5, figsize = (15, 8))
    index = 1
    x = 0
    
    for row in range(0, 2):
        for column in range(0, 5):
            ax[row + x, column].imshow(data[index].reshape(28,28), cmap = "Blues")
            ax[row + x, column].axis(False)            
            ax[row + x + 1, column].imshow(predicted[index].reshape(28,28), cmap = "Blues")
            ax[row + x + 1, column].axis(False)
        
            index += 200
            
        x += 1

    plt.show()
data = pd.read_csv("/kaggle/input/digit-recognizer/train.csv")
data.head()
data_target = data["label"]
data = data.drop(["label"], axis = 1) # drop label
train, test, _, _ = train_test_split(data, data_target, test_size = 0.2, random_state = 42)
train = train.to_numpy()
test = test.to_numpy()
train = train / 255
test = test / 255
show_samples(train)
noise = 0.4

train_noise = train + noise*np.random.normal(0, 1, size = train.shape)
test_noise = test + noise*np.random.normal(0, 1, size = test.shape)

train_noise = np.clip(train_noise, 0, 1)
test_noise = np.clip(test_noise, 0, 1)
show_samples(train_noise)
# Input
input_img = Input(shape = (784, )) 

# Encoding
encoded = Dense(64, activation = "relu")(input_img) 
encoded = Dense(32, activation = "relu")(encoded)

# Bottleneck
encoded = Dense(16, activation = "relu")(encoded)

# Decoding
decoded = Dense(32, activation = "relu")(encoded)
decoded = Dense(64, activation = "relu")(decoded)

# Output
decoded = Dense(784, activation = "sigmoid")(decoded)

autoencoder = Model(input_img, decoded) # Autoencoder

autoencoder.compile(loss = "binary_crossentropy",
                   optimizer = "rmsprop")

autoencoder.summary()
hist = autoencoder.fit(train_noise, train, epochs = 40, shuffle = True, batch_size = 256, validation_data = (test_noise, test))
plt.figure(figsize = (15, 6))
plt.plot(hist.history["loss"], label = "Train Loss")
plt.plot(hist.history["val_loss"], label = "Test Loss")
plt.title("Losses")
plt.legend()
plt.show()
show_predicted_samples(autoencoder, test_noise)
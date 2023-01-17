from PIL import Image
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import img_to_array, array_to_img
import os
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
f = pd.read_csv("../input/fashionmnist/fashion-mnist_train.csv")
df = pd.DataFrame(f)

print("Length of the dataframe : " + str(len(df)))
for i in range(0,1):
    pixel_values = []
    for j in range(1,len(df.iloc[0])):  # The first index is the label
        pixel_values.append(df.iloc[i][j])

    print(len(pixel_values))    # Should be 784 since 784 pixel values have been used to represent an image
    pixel_values = np.array(pixel_values)
    pixel_values = pixel_values.reshape(28,28)
    print(pixel_values.shape)
    # print(pixel_values)
    plt.imshow(pixel_values)
    plt.show()
X = np.load("../input/numpyfiles/X_RNN.npy")
Y_final = np.load("../input/numpyfiles/Y_RNN.npy")
Y_final = np.asarray(Y_final)

print("Shapes ----- ")
    
print(X.shape)
print(X[0].shape)
print(Y_final.shape)
print(Y_final[0].shape)

X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y_final, test_size=0.3, random_state=42)

print(X_train.shape)
print(Y_train.shape)

# Model formation
model = Sequential()

no_of_units = 10
no_of_steps = 28
no_of_inputs = 28
no_of_outputs = 10
batch_size = 1
no_of_epochs = 10

model.add(LSTM(128, input_shape=(no_of_steps, no_of_inputs)))
model.add(Dense(no_of_outputs, activation='softmax'))   # For the output class
model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=no_of_epochs, shuffle=False)
test_loss = model.evaluate(X_valid, Y_valid)
print(test_loss)
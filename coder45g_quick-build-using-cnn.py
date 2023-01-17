import os

import numpy as np

from numpy import load

from tensorflow import keras

from matplotlib import image, pyplot as plt
def create_array(name):

    base = "./dataset"

    subfolder = ("NORMAL", "PNEUMONIA")



    x = []

    y = []

    for cat in subfolder:

        ext = os.path.join(base, name, cat)

        for file in os.listdir(ext):

            x.append(np.resize(image.imread(os.path.join(ext, file)), (200, 200, 1)))

            if cat == "NORMAL":

                y.append(1)

            else:

                y.append(0)

    return np.asarray(x),np.asarray(y)
# X_train, y_train = create_array("train")
X_train = load('../input/dataset-in-npzformat/xtrain.npz')['arr_0']

y_train = load('../input/dataset-in-npzformat/ytrain.npz')['arr_0']

X_test = load('../input/dataset-in-npzformat/xtest.npz')['arr_0']

y_test = load('../input/dataset-in-npzformat/ytest.npz')['arr_0']

X_val = load('../input/dataset-in-npzformat/xval.npz')['arr_0']

y_val = load('../input/dataset-in-npzformat/yval.npz')['arr_0']
print("Training data shape:", X_train.shape)

print("Training target shape:", y_train.shape)

print("Test data shape:", X_test.shape)

print("Test target shape:", y_test.shape)

print("Validation data shape:", X_val.shape)

print("Validation target shape:", y_val.shape)
base = "../input/chest-xray-pneumonia/chest_xray"

folder = ["train", "test", "val"]

subfolder = ("NORMAL", "PNEUMONIA")

x = []
for name in folder:

    for cat in subfolder:

        ext = os.path.join(base, name, cat) #Just change the "val" with "train", "test" to

        x.append(len(os.listdir(ext)))       #get the count of images in subfolder

print(x)
plt.figure(figsize=(5, 10))

plt.subplot(3,1,1)

plt.bar(["Normal", "Pneumonia"], x[:2])



plt.subplot(3,1,2)

plt.bar(["Normal", "Pneumonia"], x[2:4])



plt.subplot(3,1,3)

plt.bar(["Normal", "Pneumonia"], x[4:])

plt.show()
arr1 = np.random.permutation(X_train.shape[0])

X_train = X_train/255

y_train = y_train/255



arr2 = np.random.permutation(X_test.shape[0])

arr3 = np.random.permutation(X_val.shape[0])

X_test = X_test/255

y_test = y_test/255

X_val = X_val/255

y_val = y_val/255
def PModel(input_shape):

    X_input = keras.layers.Input(input_shape)  

    X = keras.layers.Conv2D(7, (3, 3), strides=(1,1), padding='same', kernel_initializer='glorot_uniform', activation='relu', name = 'conv0')(X_input)

    X = keras.layers.Conv2D(14, (3, 3), strides=(1,1), padding='valid', kernel_initializer='glorot_uniform', activation='relu', name = 'conv1')(X)

    X = keras.layers.Conv2D(28, (2, 2), strides=(1,1), padding='valid', name='conv2')(X)

    X = keras.layers.BatchNormalization(axis=3, name='bn0')(X)

    X = keras.layers.Activation('relu', name='ac0')(X)

    X = keras.layers.MaxPooling2D((2,2), name='avg_pool')(X)

    X = keras.layers.Flatten()(X)

    X = keras.layers.Dense(1, activation='sigmoid', name='fc')(X)

   

    model = keras.Model(inputs = X_input, outputs = X, name='PModel')

  

    return model
pmodel = PModel(X_train.shape[1:])
pmodel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
pmodel.fit(x=X_train[arr1], y=y_train[arr1], epochs=2, batch_size=64)
performance = pmodel.evaluate(x=X_test[arr2], y=y_test[arr2])

print("Loss: ", performance[0])

print("Accuracy: ", performance[1])
print(pmodel.summary())
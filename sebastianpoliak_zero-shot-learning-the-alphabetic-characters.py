from keras.models import Sequential, Model
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils, to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import string
data = pd.read_csv('/kaggle/input/az-handwritten-alphabets-in-csv-format/A_Z Handwritten Data.csv')
print(data.shape)
data.rename(columns={'0':'label'}, inplace=True)
data["label"] = data.apply(lambda x: string.ascii_uppercase[x["label"]], axis=1)
data["label"].value_counts()
zero_shot_categories = ["J", "D", "H", "R", "Z"]
category_vectors = {
    "A": [4,8,10,12],
    "B": [1,5,11,15],
    "C": [11,13],
    "D": [1,5,11],
    "E": [1,4,5,6],
    "F": [1,4,6],
    "G": [4,11,13,14],
    "H": [1,3,4],
    "I": [2],
    "J": [3,11],
    "K": [1,8,12],
    "L": [1,7],
    "M": [1,3,8,12],
    "N": [1,3,8],
    "O": [5,11,13,14],
    "P": [1,5,15],
    "Q": [5,8,13,14],
    "R": [1,5,8,15],
    "S": [8,11,13],
    "T": [2,6],
    "U": [1,3,11,14],
    "V": [8,9,12],
    "W": [8,9,10,12],
    "X": [8,12],
    "Y": [2,8,12],
    "Z": [6,7,12]
}
for d in category_vectors.keys():
    category_vectors[d] = sum(to_categorical([v-1 for v in category_vectors[d]], num_classes=15))
data["category_vector"] = data.apply(lambda x: category_vectors[x["label"]], axis=1)
data_X = data[~data["label"].isin(zero_shot_categories)].drop(["label", "category_vector"],axis=1)
data_Y = np.asarray(list(data[~data["label"].isin(zero_shot_categories)]["category_vector"]))
zero_X = data[data["label"].isin(zero_shot_categories)].drop(["label", "category_vector"],axis=1)
zero_Y = np.asarray(list(data[data["label"].isin(zero_shot_categories)]["category_vector"]))
train_X, test_X, train_Y, test_Y = train_test_split(data_X, data_Y, random_state=42)
standard_scaler = MinMaxScaler()

train_X = standard_scaler.fit_transform(train_X)
test_X = standard_scaler.transform(test_X)
zero_X = standard_scaler.transform(zero_X)

train_X = train_X.reshape((train_X.shape[0], 28, 28, 1)).astype('float32')
test_X = test_X.reshape((test_X.shape[0], 28, 28, 1)).astype('float32')
zero_X = zero_X.reshape((zero_X.shape[0], 28, 28, 1)).astype('float32')

plt.imshow(train_X[0].reshape((28,28)))
print("Train X:", train_X.shape)
print("Test X:", test_X.shape)
print("Train Y:", train_Y.shape)
print("Test Y:", test_Y.shape)
model = Sequential()
model.add(Conv2D(16, (3, 3), padding="same", activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(32, (3, 3), padding="same", activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(15, activation='sigmoid'))
model.compile(optimizer="Adam", loss='binary_crossentropy', metrics=['binary_accuracy'])
model.summary()
model.fit(train_X, train_Y, epochs=50, batch_size=256, validation_data=(test_X, test_Y))
def nearest_category(vector, only_zero_shot=False):
    nearest = ""
    value = -1
    if only_zero_shot:
        categories = zero_shot_categories
    else:
        categories = category_vectors.keys() #all categories
    for k in categories:
        d = distance.euclidean(vector, category_vectors[k])
        if d < value or value == -1:
            nearest = k
            value = d
    return nearest
def hard_decision(predictions, only_zero_shot=False):
    output = []
    for pred in predictions:
        output.append(nearest_category([round(p) for p in pred], only_zero_shot))
    return output
train_pred = model.predict(train_X)
test_pred = model.predict(test_X)
zero_pred = model.predict(zero_X)
accuracy_score(hard_decision(test_Y), hard_decision(test_pred))
accuracy_score(hard_decision(zero_Y), hard_decision(zero_pred, only_zero_shot=True))
accuracy_score(hard_decision(zero_Y), hard_decision(zero_pred))
def plot_feature_maps(feature_maps):
    plt.figure(figsize=(8, 8))
    square = 4
    ix = 1
    for _ in range(square):
        for _ in range(square):
            ax = plt.subplot(square, square, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
            ix += 1
    # show the figure
    plt.show()
fm_model = Sequential()
fm_model.add(model.layers[0])
#fm_model.add(model.layers[1])
#fm_model.add(model.layers[2])
plot_feature_maps(fm_model.predict(np.array([zero_X[14000]])))

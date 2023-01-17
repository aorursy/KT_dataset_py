import pandas as pd

from keras.utils import to_categorical



# Loading and pre-processing training dataset

train_data = pd.read_csv("../input/fashion-mnist_train.csv")

train_label = pd.DataFrame(train_data[["label"]].copy(deep=False)) # Seperate labels (y) from inputs (X)

train_input = pd.DataFrame(train_data.drop("label", 1, inplace=False))

del train_data



# Convert labels to dummies (one-hot encoding) so that they can be used in the output layer

train_label = to_categorical(train_label)



print(train_input.describe())
# Check the distribution of labels

import pandas as pd

print(pd.DataFrame(train_label).describe())
# Normalize the inputs

train_means = train_input.mean(axis=0) # Keep these for test too

train_stds  = train_input.std(axis=0)

print("Means:")

print(train_means.head(5))

print("Stds:")

print(train_stds.head(5))



train_input = train_input - train_means # Zero mean

train_input = train_input / train_stds # 1 standard deviation

# Let us visualize some of the data points

import matplotlib.pyplot as plt

import numpy as np



def visualize_img(img_vec, title=""):

    plt.imshow(img_vec.values.reshape((28,28)), cmap="hot")

    plt.title(title)

    plt.show()



random_indices = np.random.randint(0, train_input.shape[0], 10)

for idx in random_indices:

    visualize_img(train_input.iloc[idx, :], title=str(idx))

    

from keras.models import Sequential

from keras.layers import Dense, Dropout



# Set-up the network

model = Sequential()

model.add(Dense(units=500, input_dim=train_input.shape[1],

                activation="relu",

                 kernel_initializer="random_uniform",

                 bias_initializer="zeros"))

model.add(Dropout(0.30))

model.add(Dense(units=300, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))

model.add(Dropout(0.25))

model.add(Dense(units=200, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))

model.add(Dropout(0.20))

model.add(Dense(units=100, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))

model.add(Dropout(0.15))

model.add(Dense(units=50, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))

model.add(Dropout(0.10))

model.add(Dense(units=25, activation="relu", kernel_initializer="random_uniform", bias_initializer="zeros"))

model.add(Dropout(0.05))

model.add(Dense(units=10, activation="softmax"))



# Print out the network configuration

print(model.summary())
from keras.optimizers import RMSprop



# Train the network

model.compile(loss='categorical_crossentropy', 

              optimizer="RMSprop",#lr=0.0001),

              metrics=['accuracy'])

model.fit(train_input.as_matrix(), train_label, epochs=20, batch_size=6000)
# Loading and pre-processing testing dataset

test_data = pd.read_csv("../input/fashion-mnist_test.csv") # Load the csv from file

test_label = pd.DataFrame(test_data[["label"]].copy(deep=False)) # Seperate labels (y) from inputs (X)

test_input = pd.DataFrame(test_data.drop("label", 1, inplace=False))

del test_data



# Convert labels to dummies (one-hot encoding) so that they can be used in the output layer

test_label = to_categorical(test_label)



print(pd.DataFrame(test_label).describe())



# Apply normalization

test_input = test_input - train_means # Zero mean

test_input = test_input / train_stds # 1 standard deviation
# Evaluate the model

test_loss_and_metrics = model.evaluate(test_input.as_matrix(), test_label)

train_loss_and_metrics = model.evaluate(train_input.as_matrix(), train_label)

print("")

print("Test Accuracy:" + str(test_loss_and_metrics[1]))

print("Train Accuracy:" + str(train_loss_and_metrics[1]))
# Let us visualize the false predictions

num_to_vis = 10



# Check which instances are falsely or truely classified

test_preds = model.predict(test_input.as_matrix())

pred_mat = pd.DataFrame(np.argmax(test_preds, axis=1), columns=["pred"])

pred_mat["actual"] = np.argmax(test_label, axis=1)

pred_mat["is_T"] = pred_mat.actual == pred_mat.pred



# Select and visualize random false predictions

f_mat = pred_mat[(pred_mat.is_T == False)]

f_mat = f_mat.iloc[np.random.randint(0, f_mat.shape[0], num_to_vis), :]

for f_mat_idx in  range(f_mat.shape[0]):

    title = "Actual: " + str(f_mat.iloc[f_mat_idx].actual) + " - Pred: " + str(f_mat.iloc[f_mat_idx].pred)

    visualize_img(test_input.iloc[f_mat.index[f_mat_idx], :], title=title)



# Select and visualize random true predictions

f_mat = pred_mat[(pred_mat.is_T == True)]

f_mat = f_mat.iloc[np.random.randint(0, f_mat.shape[0], num_to_vis), :]

for f_mat_idx in  range(f_mat.shape[0]):

    title = "Actual: " + str(f_mat.iloc[f_mat_idx].actual) + " - Pred: " + str(f_mat.iloc[f_mat_idx].pred)

    visualize_img(test_input.iloc[f_mat.index[f_mat_idx], :], title=title)

    

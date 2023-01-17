"""Classification problem organized in a way that pleases my brain."""
# Reference: Pierre Kiefer's kernel linked below under an Apache-2.0 license.
# https://www.kaggle.com/pierrek20/multiclass-iris-prediction-with-tensorflow-keras

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential 
from keras.utils.np_utils import to_categorical as one_hot
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
data = (
    pd
    .read_csv('../input/Iris.csv')
    .drop(['Id'], axis=1)
)

print(data[:3])  # Read data and print a small sample
def reindex(data):
    """Reset the index value on the data."""
    return data.reset_index(drop=True)

def input_data(data, label):
    """Create an unlabeled array."""
    return np.array(data.drop([label], axis=1))

def output_labels(data, label):
    """Create binary output labels for training data."""
    labels = data[label]
    encoder = LabelEncoder()
    return encoder, one_hot(encoder.fit(labels).transform(labels))

def split_training_data(inputs, labels, test_size=0.1, random_state=0):
    """Split a training set into train/test parts."""
    train_x, test_x, train_y, test_y = train_test_split(
        inputs,
        labels,
        test_size=test_size,
        random_state=random_state
    )
    return train_x, test_x, train_y, test_y

def fit_model(train_x, test_x, train_y, test_y, epochs=10, batch_size=10):
    """Fit the model using train and test data over some epochs."""
    model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size)
    scores = model.evaluate(test_x, test_y)
    print("\n{}: {:.2f}%".format(model.metrics_names[1], scores[1] * 100))

def test_model(inputs, truths, encoder):
    """Test the model by decoding back to the true label names."""
    perception = model.predict_classes(inputs)
    prediction = encoder.inverse_transform(np.argmax(one_hot(perception), axis=1))
    for guess, truth in zip(prediction, truths):
        print("{} thinks {} is {}".format("✔" if guess == truth else "✖", truth, guess))

LABEL = 'Species'

dataset = shuffle(reindex(data)[::2])  # let's take half the data and randomize it.
inputs = input_data(dataset, label=LABEL)
encoder, labels = output_labels(dataset, label=LABEL)
truths = dataset[LABEL]  # What we predict at the end

model = Sequential()
model.add(Dense(8, input_dim=len(data.columns) - 1, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

train_inputs, test_inputs, train_labels, test_labels = \
    split_training_data(inputs, labels)

fit_model(
    train_inputs,
    test_inputs,
    train_labels,
    test_labels,
    epochs=10,
    batch_size=10
)

test_model(inputs, dataset.Species, encoder)
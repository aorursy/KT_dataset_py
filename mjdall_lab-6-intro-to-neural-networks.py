import numpy as np

import pandas as pd

import keras

import os

import matplotlib.pyplot as plt
# CONSTS

RANDOM_STATE = 1313361

RELU = 'relu'
from sklearn.preprocessing import OneHotEncoder

from sklearn.impute import SimpleImputer



pd.options.mode.chained_assignment = None



def encode_categorical(df_col):

    '''

    Encodes a categorical dataframe column into numeric values.

    

    Args:

        df_col (DataFrame): A dataframe column which is categorical.

    '''

    onehot_encoder = OneHotEncoder(sparse=False)

    onehot_encoded = onehot_encoder.fit_transform(df_col)

    return pd.DataFrame(onehot_encoded)



def impute_values(df):

    '''Imputes missing values for passed in DataFrame'''

    imputer = SimpleImputer(strategy='median')

    imputer.fit(df)

    df_transformed = imputer.transform(df)

    return pd.DataFrame(df_transformed, columns=df.columns)

housing_raw = pd.read_csv('../input/housing.csv')



housing_raw['ocean_proximity'] = encode_categorical(housing_raw[['ocean_proximity']])

housing = impute_values(housing_raw)



X = housing.drop('median_house_value', axis=1)

y = housing['median_house_value'].copy()

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler



# split into train/validation/test

X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, test_size=0.1875, random_state=RANDOM_STATE)



print(X.shape)

print(X_train_full.shape)

print(X_train.shape)

print(X_valid.shape)



# scale the data

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_valid_scaled = scaler.transform(X_valid)

X_test_scaled = scaler.transform(X_test)

seq_nn_mod = keras.models.Sequential([

    keras.layers.Dense(300, activation=RELU, input_shape=X_train_scaled.shape[1:]),

    keras.layers.Dense(300, activation=RELU),

    keras.layers.Dense(300, activation=RELU),

    keras.layers.Dense(300, activation=RELU),

    keras.layers.Dense(100, activation=RELU),

    keras.layers.Dense(100, activation=RELU),

    keras.layers.Dense(100, activation=RELU),

    keras.layers.Dense(1)

])



seq_nn_mod.compile(

    loss='mean_squared_error',

    optimizer='adam',

    metrics=['mae']

)

history = seq_nn_mod.fit(X_train_scaled, y_train, epochs=30, validation_data=(X_valid_scaled, y_valid))
def plot_training(history):

    pd.DataFrame(history.history).plot(figsize=(8, 5))

    plt.grid(True)

    # plt.gca().set_ylim(0, 1)

    plt.show()



def plot_loss(history):

    # Get training and test loss histories

    training_loss = history.history['loss']

    test_loss = history.history['val_loss']



    # Create count of the number of epochs

    epoch_count = range(1, len(training_loss) + 1)



    # Visualize loss history

    plt.plot(epoch_count, training_loss, 'r--')

    plt.plot(epoch_count, test_loss, 'b-')

    plt.legend(['Training Loss', 'Test Loss'])

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.show()



def plot_accuracy(history):

    # Plot training & validation accuracy values

    plt.plot(history.history['mean_absolute_error'])

    plt.plot(history.history['val_mean_absolute_error'])

    plt.title('Model accuracy')

    plt.ylabel('Accuracy')

    plt.xlabel('Epoch')

    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

plot_training(history)
plot_loss(history)

plot_accuracy(history)
final_mae = history.history['mean_absolute_error'][29]

final_val_mae = history.history['val_mean_absolute_error'][29]

print('final mae: %.2f, final validation mae: %.2f' % (final_mae, final_val_mae))
from sklearn.metrics import mean_absolute_error



y_pred = seq_nn_mod.predict(X_test_scaled)

test_mae = mean_absolute_error(y_test, y_pred)



print('Test Set MAE: %.2f' % test_mae)
y_pred = seq_nn_mod.predict(X_test_scaled)



under_15 = 0

over_500 = 0



for pred, actual in zip(y_pred, y_test):

    if pred > 500000:

        over_500 += 1

    if pred < 15000:

        under_15 += 1



print('Less than 15,000: %d, Greater than 500,000: %d' % (under_15, over_500))
from keras.datasets import cifar10



# Import cifar data

(X_train_conv, y_train_conv), (X_test_conv, y_test_conv) = cifar10.load_data()
from keras.utils import to_categorical



# categorise the labels

y_train_conv_categorical = to_categorical(y_train_conv)

y_test_conv_categorical = to_categorical(y_test_conv)



print(y_train_conv_categorical)
X_train_conv_scaled = X_train_conv / 255

X_test_conv_scaled = X_test_conv / 255



# scale the X data

print(X_train_conv_scaled.shape)

print(X_test_conv_scaled.shape)
from functools import partial    



# model from the lecture slides

DefaultConv2D = partial(keras.layers.Conv2D, kernel_size=3, activation=RELU, padding='SAME')

conv_model = keras.models.Sequential([

    DefaultConv2D(filters=64, kernel_size=7, input_shape=[32, 32, 3]),

    keras.layers.MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=128),

    DefaultConv2D(filters=128),

    keras.layers.MaxPooling2D(pool_size=2),

    DefaultConv2D(filters=256),

    DefaultConv2D(filters=256),

    keras.layers.MaxPooling2D(pool_size=2),

    keras.layers.Flatten(),

    keras.layers.Dense(units=128, activation=RELU),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(units=64, activation=RELU),

    keras.layers.Dropout(0.5),

    keras.layers.Dense(units=10, activation='softmax')

])



conv_model.compile(

    loss='categorical_crossentropy',

    optimizer=keras.optimizers.SGD(lr=0.032),

    metrics=['accuracy']

)
conv_history = conv_model.fit(X_train_conv_scaled, y_train_conv_categorical, epochs=30, validation_split=0.15)
plot_training(conv_history)
from sklearn.metrics import accuracy_score



# predict X_test

y_pred_conv = conv_model.predict(X_test_conv_scaled)



# so we can compare pred and test

normalized_y_pred = np.argmax(y_pred_conv, axis=1)

normalized_y_test = np.argmax(y_test_conv_categorical, axis=1)



# get the accuracy of the model

cnn_acc = accuracy_score(normalized_y_test, normalized_y_pred)



print('model accuracy: %.2f%%' % (cnn_acc * 100))
"""Find the missclassifications"""



# from https://www.cs.toronto.edu/~kriz/cifar.html

CIFAR_LABELS = {

    0: 'airplane',

    1: 'automobile',

    2: 'bird',

    3: 'cat',

    4: 'deer',

    5: 'dog',

    6: 'frog',

    7: 'horse',

    8: 'ship',

    9: 'truck',

}



IDX = 'idx'

DIFFERENCE = 'difference'

EXPECTED_LABEL = 'expected_label'

PREDICTED_LABEL = 'predicted_label'



DEFAULT = {

    IDX: -1,

    DIFFERENCE: -1,

    EXPECTED_LABEL: None,

    PREDICTED_LABEL: None,

}



wrong_predictions = {}



idx = 0

for cnn_pred, actual in zip(normalized_y_pred, normalized_y_test):

    # use absolute to get missclassification in any direction

    dif = abs(cnn_pred - actual)



    # make sure this entry is set

    wrong_predictions.setdefault(actual, dict.copy(DEFAULT))



    # if we've found a new maximum

    if wrong_predictions[actual][DIFFERENCE] < dif:

        # set the occurence

        wrong_predictions[actual][IDX] = idx

        wrong_predictions[actual][DIFFERENCE] = dif

        wrong_predictions[actual][EXPECTED_LABEL] = CIFAR_LABELS[actual]

        wrong_predictions[actual][PREDICTED_LABEL] = CIFAR_LABELS[cnn_pred]

    idx += 1



print(wrong_predictions)
def visualize_image(X, title, idx):

    img = X[idx]

    plt.imshow(img)

    plt.title(title)

    plt.show()





for label_idx in wrong_predictions:

    pred_detials = wrong_predictions[label_idx]

    actual_label = pred_detials[EXPECTED_LABEL]

    predicted_label = pred_detials[PREDICTED_LABEL]

    title = '%s missclassified as a %s, dif: %d' % (

        actual_label, predicted_label, pred_detials[DIFFERENCE]

    )

    visualize_image(X_test_conv_scaled, title, pred_detials[IDX])
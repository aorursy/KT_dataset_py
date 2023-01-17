import numpy as np

import pandas as pd

from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score, log_loss

import keras

from keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, Dropout

from keras.layers.normalization import BatchNormalization

from keras.models import Sequential
train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

y = train['label'].values

X = train.drop(['label'], axis=1).values
X.shape
y.shape
def normalise_data(x):

    return x / 255.0
def reshape(x, model='cnn'):

    if model == 'mlp':

        return x.reshape(x.shape[0], -1)  # -1 flattens rest of dimensions: no change

    else:

        # model == 'cnn'

        return x.reshape(x.shape[0], 28, 28, 1)  # 1 for greyscale, 3 for rgb
def one_hot(y):

    return keras.utils.to_categorical(y, np.max(y) + 1)
def model_performance(model, x_train, x_test, y_test):

    predictions = model.predict(x_test)  # same as predict_proba in softmax output

    y_pred = np.argmax(np.round(predictions), axis=1)

    y_test_og = np.argmax(y_test, axis=1)



    accuracy = accuracy_score(y_test_og, y_pred)

    balanced_accuracy = balanced_accuracy_score(y_test_og, y_pred)

    # auc = roc_auc_score(y_test_og, predictions)  # need 1-vs-all approach for auc-roc curve

    loss = log_loss(y_test_og, predictions)

    report = classification_report(y_test_og, y_pred)

    matrix = confusion_matrix(y_test_og, y_pred)



    tp = sum(np.diagonal(matrix))

    fp = np.sum(matrix, axis=0) - tp

    tn = 0  # must be computed per class

    fn = np.sum(matrix, axis=1) - tp



    print(f'training cases={x_train.shape[0]}, test cases={y_test.shape[0]}, possible outcomes={y_test.shape[1]}')

    print(f'accuracy={accuracy:.2f}%, balanced_accuracy={balanced_accuracy:.2f}%, loss={loss:.3f}')

    # print(f'auc={auc:.3f}')

    print(report)
def mlp(x, y):

    """Multilayer Perceptron"""

    # Initialise MLP

    # Input layer with nodes=number of features in the dataset

    model = Sequential()

    # Hidden layer, one hidden layer is sufficient for the large majority of problems

    model.add(Dense(512, activation='relu', input_shape=(x.shape[1],)))

    model.add(Dropout(0.2))  # apply dropout to input, randomly setting a fraction rate of input units to 0 at each

    # update during training time, which helps prevent overfitting

    # Hidden layer, size between input and output layers

    model.add(Dense(512, activation='relu'))

    model.add(Dropout(0.2))

    # Output layer, one node unless 'softmax' in multi-class problems

    model.add(Dense(y.shape[1], activation='softmax'))

    # Compile

    model.compile(loss=keras.losses.categorical_crossentropy,  # 'sparse_categorical_crossentropy' doesn't require oh

                  optimizer=keras.optimizers.RMSprop(),

                  metrics=['accuracy'])  # more metric history available https://keras.io/metrics/

    return model
def cnn(x, y):

    """Convolutional Neural Network"""

    # Initialise CNN

    # Input layer

    model = Sequential()

    # Hidden layer

    model.add(Conv2D(64, (3, 3), input_shape=(x.shape[1], x.shape[2], 1)))  # 64 filters (output space), 3x3 convolution

    # BatchNormalization() aids with overfitting, according to authors and Andrew Ng it should be applied immediately

    # before activation function (non-linearity)

    model.add(BatchNormalization())

    model.add(Activation('relu'))  # rectified linear unit (fire or not)

    model.add(MaxPooling2D(pool_size=(2, 2)))  # maximum value for each patch on feature map reduced by 2x2 pool_size

    # Hidden layer

    model.add(Conv2D(64, (3, 3)))

    model.add(BatchNormalization())

    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Output Layer

    model.add(Flatten())  # flattens the input

    model.add(Dense(64))  # regular densely connected NN layer, no activation function means linear activation

    model.add(Dense(y.shape[1]))  # can also do e.g. model.add(Dense(64, activation='tanh'))

    model.add(BatchNormalization())

    model.add(Activation('softmax'))  # softmax activation function as output, turns into weights that sum to 1

    # Compile

    model.compile(loss=keras.losses.categorical_crossentropy,

                  optimizer=keras.optimizers.Adam(),

                  metrics=['accuracy'])

    return model
def plot_model_history(metric):

    plt.figure()

    plt.plot(history.history[metric])

    plt.plot(history.history[f'val_{metric}'])

    plt.title(f'model {metric}')

    plt.ylabel(f'{metric}')

    plt.xlabel('epoch')

    plt.legend(['train', 'test'], loc='upper left')

    plt.show()
ml_model = cnn
# Load the train test data

train = pd.read_csv('/kaggle/input/digit-recognizer/train.csv')

y = train['label'].values

X = train.drop(['label'], axis=1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=38)



# Normalise and shape data

X_train, X_test = normalise_data(X_train), normalise_data(X_test)

X_train, X_test = reshape(X_train, ml_model.__name__), reshape(X_test, ml_model.__name__)

y_train, y_test = one_hot(y_train), one_hot(y_test)



# Model

model = ml_model(X_train, y_train)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=64, epochs=10)



# Validate

model.summary()

test_loss, test_accuracy = model.evaluate(X_test, y_test, batch_size=64, verbose=1)

print(f'test loss={test_loss}, test accuracy={test_accuracy}')

plot_model_history('loss')

plot_model_history('accuracy')

model_performance(model, X_train, X_test, y_test)
# Load and manipulate data

x_test = pd.read_csv('/kaggle/input/digit-recognizer/test.csv')

x_test = x_test.values

x_test = normalise_data(x_test)

x_test = reshape(x_test, ml_model.__name__)



# Predict

y_test = model.predict(x_test)



# Manipulate submission format

submission = pd.DataFrame({'Label': np.argmax(y_test, axis=1)})

submission['ImageId'] = submission.index + 1

submission = submission[['ImageId', 'Label']]

submission.to_csv('/kaggle/working/digit-recognizer-submission.csv', index=False)
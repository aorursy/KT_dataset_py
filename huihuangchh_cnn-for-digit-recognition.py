import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils.np_utils import to_categorical
# load train data
def load_train(fname):
    print("load train dataset...")
    df = pd.read_csv(fname, index_col=0)
    data = df.values
    data = data.reshape(data.shape[0], 28, 28, 1)
    target = df.index.values
    target = to_categorical(target)
    return data / 255, target

# load test data
def load_test(fname):
    print("load test dataset...")
    df = pd.read_csv(fname)
    data = df.values
    data = data.reshape(data.shape[0], 28, 28, 1)
    return data / 255
def train_and_test():
    batch_size = 200
    nb_epoch = 30

    train_data, train_target = load_train("../input/train.csv")

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1), padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation="relu", padding="same"))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(10, activation='softmax'))

    # sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.9)
    adam = Adam(lr=1e-3)
    model.compile(optimizer=adam, loss="binary_crossentropy", metrics=['accuracy'])

    callback = [EarlyStopping(monitor='val_loss', patience=10, verbose=0)]
    model.fit(train_data, train_target, epochs=nb_epoch, validation_split=0.1,
              batch_size=batch_size, callbacks=callback)
    model.save(os.path.join(RES_DIR, "model.h5"))
    pred_train = model.predict(train_data)

    train_target_class = np.argmax(train_target, axis=1)
    score = log_loss(train_target, pred_train)
    print("accuracy log_loss:", score)
    print("accuracy score:", accuracy_score(train_target_class, model.predict_classes(train_data)))
    # =============================================== #
    # load test
    test_data = load_test("../input/test.csv")
    pred_test = model.predict_classes(test_data, batch_size=100)
    print("shape:", pred_test.shape)
    # create_submission(pred_test)
    ll = len(pred_test)
    index = [i + 1 for i in range(ll)]
    header = ['Label']
    df = pd.DataFrame(data=np.array([pred_test]).T, index=index, columns=header)
    df.index.name = "ImageId"
    df.to_csv("my_submission_" + str(score) + ".csv")
train_and_test()
import pandas as pd
import h5py
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def cat2num_sex(val):
    if "male" == val:
        return 1
    else:
        return 0


def cat2num_embarking(val):
    if "C" == val:
        return 2
    elif "Q" == val:
        return 1
    else:
        return 0


def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)

    # Shuffle
    df = df.sample(frac=1).reset_index(drop=True)

    # Missing Data
    df['Age'] = df['Age'].fillna(value=df.Age.median())

    # Categorical to Numerical
    df['Sex'] = df['Sex'].apply(cat2num_sex)
    df['Embarked'] = df['Embarked'].apply(cat2num_embarking)

    # Normalization
    scaler = MinMaxScaler()
    df['Age'] = scaler.fit_transform(np.array(df['Age']).reshape(-1, 1))
    df['Fare'] = scaler.fit_transform(np.array(df['Fare']).reshape(-1, 1))

    # Columns
    target_cols = ["Survived"]
    features_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

    y_train = df[target_cols]
    X_train = df[features_cols]
    
    # Pairwise Correlation & Covariance
    print(X_train.Fare.corr(y_train.Survived))
    print(X_train.Fare.cov(y_train.Survived))

    return X_train, y_train


def wrap_preprocess():
    X_train, y_train = preprocess_data("../input/train.csv")

    train_size = int(len(y_train) * 0.80)

    with h5py.File("dataset-v4.h5", 'w') as f:
        f.create_dataset("X_train", data=np.array(X_train[:train_size]))
        f.create_dataset('y_train', data=np.array(y_train[:train_size]))
        f.create_dataset("X_val", data=np.array(X_train[train_size:]))
        f.create_dataset("y_val", data=np.array(y_train[train_size:]))


wrap_preprocess()


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU


def init_model():
    model = Sequential()
    model.add(Dense(70, input_dim=7))
    model.add(Dropout(0.2))
    model.add(LeakyReLU())
    model.add(Dense(50, input_dim=10))
    model.add(Dropout(0.1))
    model.add(LeakyReLU())
    model.add(Dense(1, input_dim=10))
    model.add(Activation('sigmoid'))
    return model
from keras.optimizers import SGD
from keras.callbacks import CSVLogger, ModelCheckpoint
import os
import h5py


def mkdir_exists(dir):
    if os.path.exists(dir):
        return
    os.mkdir(dir)


def data_reader():
    with h5py.File(''.join(['dataset-v4.h5']), 'r') as hf:
        X_train = hf['X_train'].value
        y_train = hf['y_train'].value
        X_val = hf['X_val'].value
        y_val = hf['y_val'].value
    return X_train, y_train, X_val, y_val


def train():
    model = init_model()

    sgd = SGD(lr=1e-2, decay=1e-4, momentum=0.9, nesterov=True)

    model.compile(
        loss='mean_squared_error',
        optimizer=sgd,
        metrics=['accuracy']
    )

    X_train, y_train, X_val, y_val = data_reader()

    mkdir_exists("weights")

    # training & validation
    history = model.fit(X_train,
              y_train,
              batch_size=64,
              validation_data=(X_val, y_val),
              epochs=1000,
              verbose=0,
              callbacks=[
                  CSVLogger(
                      'logs.csv',
                      append=True
                  ),
                  ModelCheckpoint(
                      'weights/model-ffn.hdf5',
                      monitor='val_acc',
                      verbose=0,
                      mode='min'
                  )
              ]
              )
    
    return history


history = train()
from matplotlib import pyplot as plt

plt.figure(figsize=(16, 12))
plt.title("Train & Validation Loss")
plt.plot(history.history['loss'], label="Training Loss")
plt.plot(history.history['val_loss'], label="Validation Loss")
plt.legend()
plt.figure(figsize=(16, 12))
plt.title("Train & Validation Accuracy")
plt.plot(history.history['acc'], label="Training Accuracy")
plt.plot(history.history['val_acc'], label="Validation Accuracy")
plt.legend()
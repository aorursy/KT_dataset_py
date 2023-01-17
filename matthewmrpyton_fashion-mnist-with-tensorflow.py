import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow import keras

from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold



import gc

import os

print(os.listdir("../input"))
data_train = pd.read_csv('../input/fashion-mnist_train.csv')

data_test = pd.read_csv('../input/fashion-mnist_test.csv')



print('train:', data_test.shape)

print('test:', data_train.shape)



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



data = np.array(data_train.iloc[:, 1:])

labels = np.array(data_train.iloc[:, 0])

test_data = np.array(data_test.iloc[:, 1:])

test_labels = np.array(data_test.iloc[:, 0])
train_data, val_data, train_labels, val_labels = train_test_split(data, labels, test_size=0.3, random_state=2019)



train_data = train_data.reshape(train_data.shape[0], img_rows, img_cols, 1)

val_data = val_data.reshape(val_data .shape[0], img_rows, img_cols, 1)

test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)



train_data = train_data.astype('float32')

val_data = val_data.astype('float32')

test_data = test_data.astype('float32')

train_data /= 255

val_data /= 255

test_data /= 255



print('train shape:', train_data.shape, train_labels.shape)

print('valid shape:',val_data.shape, val_labels.shape)

print('test shape:', test_data.shape, test_labels.shape)
def display(i):

    plt.imshow(train_data[i, :, :, 0])

    plt.colorbar()

    plt.grid(False)

    plt.title(class_names[train_labels[i]])

    plt.show()



    

display(2)

display(4)

display(5)
def plot_history_vs_epoch(histories, key):

    plt.figure(figsize=(16, 10))



    for name, history in histories:

        val = plt.plot(history.epoch, history.history['val_' + key], '.-', label=name.title()+' - Val')

        plt.plot(history.epoch, history.history[key], '--', color=val[0].get_color(), label=name.title()+' - Train')



    plt.xlabel('Epochs')

    plt.ylabel(key.replace('_', ' ').title())

    plt.legend()



    plt.xlim([0, max(history.epoch)])
# CNN model.

def create_model():

    model = keras.Sequential([

        keras.layers.Conv2D(filters=32, kernel_size=[5, 5], padding="same", activation='relu', input_shape=(28, 28, 1)),

        keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),

        keras.layers.Dropout(0.4),



        keras.layers.Conv2D(filters=64, kernel_size=[5, 5], padding="same", activation='relu'),

        keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),

        keras.layers.Dropout(0.4),



        keras.layers.Flatten(),

        keras.layers.Dense(256, activation='relu'),

        keras.layers.Dropout(0.4),

        keras.layers.Dense(10, activation='softmax')

    ])



    model.compile(

        optimizer='adam',

        loss='sparse_categorical_crossentropy',

        metrics=['accuracy', 'sparse_categorical_crossentropy']

    )

    return model



cnn_dt_model = create_model()

cnn_dt_model.summary()
ckpt = keras.callbacks.ModelCheckpoint(

    f'weights.h5',

    save_best_only=True,

    save_weights_only=True,

    verbose=1,

    monitor='acc',

    mode='max'

)



# Train the model.

cnn_dt_history = cnn_dt_model.fit(

    train_data,

    train_labels,

    validation_data=(val_data, val_labels),

    batch_size=32,

    epochs=60,

    verbose=2,

    # callbacks=[ckpt]

)
summ_list = [

    ('M1', cnn_dt_history)

]



plot_history_vs_epoch(summ_list, key='sparse_categorical_crossentropy')

plot_history_vs_epoch(summ_list, key='acc')



print('Acc:', max(cnn_dt_history.history['val_acc']))
model = cnn_dt_model

val_pred_prob = folds_metrics

val_pred_labels = np.argmax(val_pred_prob, axis=-1)

print('Valid ACC:', accuracy_score(val_labels, val_pred_labels))



# Evaluating test.

test_pred_prob = model.predict(test_data)

test_pred_labels = np.argmax(test_pred_prob, axis=-1)

print(test_pred_labels)

print('Test ACC:', accuracy_score(test_labels, test_pred_labels))
data_ = pd.read_csv('../input/fashion-mnist_train.csv')

data_test_ = pd.read_csv('../input/fashion-mnist_test.csv')



print('train:', data_test.shape)

print('test:', data_train.shape)



class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',

               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']



img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)



data = np.array(data_.iloc[:, 1:])

labels = np.array(data_.iloc[:, 0])

test_data = np.array(data_test_.iloc[:, 1:])

test_labels = np.array(data_test_.iloc[:, 0])



del data_, data_test_

gc.collect()
data = data.reshape(data.shape[0], img_rows, img_cols, 1)

test_data = test_data.reshape(test_data.shape[0], img_rows, img_cols, 1)



data = data.astype('float32')

test_data = test_data.astype('float32')

data /= 255

test_data /= 255



print('train shape:', data.shape, labels.shape)

print('test shape:', test_data.shape, test_labels.shape)
# K-fold Cross-validation.

N_SPLITS = 5

BATCH_SIZE = 32

EPOCHS = 60



data_index = np.array(range(data.shape[0]))

splits = list(KFold(n_splits=N_SPLITS, random_state=0, shuffle=True).split(data_index))



models = []

predicts_prob = []

folds_acc = []



for n, (train_index, val_index) in enumerate(splits):

    print(f'Fold: {n}')

    print("TRAIN:", train_index, "VAL:", val_index)

    train_data, val_data = data[train_index, :], data[val_index, :]

    train_labels, val_labels = labels[train_index], labels[val_index]



    # It is basically an early stopping.

    ckpt = keras.callbacks.ModelCheckpoint(f'weights_fold-{n}.h5', save_best_only=True, save_weights_only=True, verbose=1, monitor='acc', mode='max')

    

    model = create_model()

    

    history = model.fit(

        train_data,

        train_labels,

        validation_data=(val_data, val_labels),

        batch_size=BATCH_SIZE,

        epochs=EPOCHS,

        verbose=2,

        callbacks=[ckpt]

    )



    model.load_weights(f'weights_fold-{n}.h5')

    

    test_pred_prob = model.predict(test_data, batch_size=512)

    predicts_prob.append(test_pred_prob)



    test_pred_labels = np.argmax(test_pred_prob, axis=-1)

    acc = accuracy_score(test_labels, test_pred_labels)

    print(f"Fold's ACC: {acc}")

    folds_acc.append(acc)



# 1) Average ACC and  2) average probabilities and calculate one accuracy in the end. Both might be informative. 



# 1)

print(f'Avg ACC: {np.sum(folds_acc) / len(folds_acc)}')



# 2)

# Average prob and calculate 1 accuracy.



    

# preds_val = np.concatenate(preds_val)[...,0]

# y_val = np.concatenate(y_val)

# preds_val.shape, y_val.shape

# TODO: Data augmentation.